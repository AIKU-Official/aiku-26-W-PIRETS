import torch
from torch.utils.data import Dataset
import random
from src.g2p import G2PConverter, KOREAN_CONFUSION_MAP, ENGLISH_CONFUSION_MAP

class RerankerDataset(Dataset):
    def __init__(self, corpus, hard_negative_cache, is_train=True, num_negatives=4):
        """
        :param corpus: {'id': {'raw': '...', 'g2p': '...'}} 형태의 전체 코퍼스
        :param hard_negative_cache: {'id': ['neg_id1', 'neg_id2', ...]} 형태의 오답 맵핑
        """
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        self.hard_negative_cache = hard_negative_cache
        self.is_train = is_train
        self.num_negatives = num_negatives
        
        # 동적 쿼리 생성을 위한 도구들
        self.g2p = G2PConverter()
        self.min_words = 4
        self.max_words = 12
        self.noise_delete_prob = 0.10
        self.noise_swap_prob = 0.10
        self.noise_random_prob = 0.05
        self.phoneme_vocab = list(set(list(KOREAN_CONFUSION_MAP.keys()) + list(ENGLISH_CONFUSION_MAP.keys())))

    def __len__(self):
        return len(self.doc_ids)

    def _create_query_span(self, text: str, rng: random.Random) -> str:
        """
        [Slicing] rng 객체를 받아서 자름 (Train은 랜덤, Val은 고정)
        """
        words = text.split()
        total_words = len(words)
        if total_words <= self.min_words: return text
            
        current_max = min(total_words, self.max_words)
        target_len = rng.randint(self.min_words, current_max)
        max_start_idx = total_words - target_len
        start_idx = rng.randint(0, max_start_idx)
        
        return " ".join(words[start_idx : start_idx + target_len])

    def _add_phonetic_noise(self, phoneme_str: str, rng: random.Random) -> str:
        """
        [Noise Injection] rng 객체를 받아서 노이즈 추가
        """
        phonemes = phoneme_str.split()
        if len(phonemes) < 2: return phoneme_str 

        noisy_phonemes = []
        
        for ph in phonemes:
            prob = rng.random() #  여기서 전달받은 rng 사용!
            
            # 1. Deletion (10%)
            if prob < self.noise_delete_prob:
                continue 
            
            # 2. Confusion (10%)
            elif prob < (self.noise_delete_prob + self.noise_swap_prob):
                if ph in KOREAN_CONFUSION_MAP:
                    noisy_phonemes.append(rng.choice(KOREAN_CONFUSION_MAP[ph]))
                elif ph.upper() in ENGLISH_CONFUSION_MAP:
                    cand = rng.choice(ENGLISH_CONFUSION_MAP[ph.upper()])
                    if ph.islower(): cand = cand.lower()
                    noisy_phonemes.append(cand)
                else:
                    noisy_phonemes.append(ph)

            # 3. Random Swap (5%)
            elif prob < (self.noise_delete_prob + self.noise_swap_prob + self.noise_random_prob):
                noisy_phonemes.append(rng.choice(self.phoneme_vocab))
            
            # 4. Keep
            else:
                noisy_phonemes.append(ph)
            
        if not noisy_phonemes: return phoneme_str
        return " ".join(noisy_phonemes)

    def __getitem__(self, idx):
        pos_id = self.doc_ids[idx]
        raw_text = self.corpus[pos_id]['raw']
        pos_g2p = self.corpus[pos_id]['g2p'] 
        
        # -------------------------------------------------------
        # 1. [Deterministic] 고정 난수 생성기 (Slicing 전용)
        # -------------------------------------------------------
        # mine_hard_negatives.py에서 사용한 것과 100% 동일한 시드 생성
        seed_val = hash(raw_text)
        fixed_rng = random.Random(seed_val)
        
        # 항상 동일한 위치의 4~12 토큰을 잘라냅니다. (오프라인 캐시와 동기화)
        query_span = self._create_query_span(raw_text, fixed_rng)
        query_phoneme = self.g2p(query_span)

        # -------------------------------------------------------
        # 2. [Stochastic] 동적 난수 생성기 (Noise 전용)
        # -------------------------------------------------------
        if self.is_train:
            dynamic_rng = random # Train: 매 Epoch마다 완전히 새로운 난수 발생
        else:
            dynamic_rng = fixed_rng # Val: 평가의 일관성을 위해 노이즈도 고정

        # 동일한 query_span에 매번 다른 패턴의 G2P 노이즈를 입힙니다.
        query_g2p = self._add_phonetic_noise(query_phoneme, dynamic_rng)

        # -------------------------------------------------------
        # 3. 데이터 조립 (미리 캐싱된 오답 ID 활용)
        # -------------------------------------------------------
        neg_ids = self.hard_negative_cache.get(pos_id, [])
        if len(neg_ids) > self.num_negatives:
            # 오답을 섞을 때도 매번 다른 조합을 보기 위해 dynamic_rng 사용
            neg_ids = dynamic_rng.sample(neg_ids, self.num_negatives) 
            
        neg_g2ps = [self.corpus[n_id]['g2p'] for n_id in neg_ids]

        # 5. Reranker 모델용 Pair 조립
        pairs = []
        labels = []
        
        pairs.append((query_g2p, pos_g2p))
        labels.append(1.0)
        
        for neg_g2p in neg_g2ps:
            pairs.append((query_g2p, neg_g2p))
            labels.append(0.0)
            
        return pairs, labels
    

def reranker_collate_fn(batch, tokenizer, max_length=256):
    """
    DataLoader가 호출하여 Dataset의 결과물들을 하나의 Tensor Batch로 압축하는 함수
    """
    all_queries = []
    all_passages = []
    all_labels = []
    
    # 1. Flattening (1차원 리스트로 펼치기)
    # batch 안에는 여러 개의 (pairs, labels) 튜플이 들어있습니다.
    for pairs, labels in batch:
        for q, p in pairs:
            all_queries.append(q)
            all_passages.append(p)
        all_labels.extend(labels)
        
    # 2. Cross-Encoder 전용 Tokenizing & Dynamic Padding
    # HuggingFace Tokenizer에 두 개의 리스트를 넘기면 자동으로 [CLS] Query [SEP] Passage [SEP] 형태로 묶어줍니다.
    encoded = tokenizer(
        all_queries,
        all_passages,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # 3. Loss 계산을 위한 Label Tensor 변환 (BCE Loss는 float 타입을 요구함)
    encoded['labels'] = torch.tensor(all_labels, dtype=torch.float)
    
    return encoded