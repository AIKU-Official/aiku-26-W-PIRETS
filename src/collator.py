import random
import logging
from typing import List, Dict, Any
from transformers import AutoTokenizer
from src.g2p import G2PConverter, KOREAN_CONFUSION_MAP, ENGLISH_CONFUSION_MAP
import torch

logger = logging.getLogger(__name__)

class RetrievalCollator:
    def __init__(self, config, is_train=True):
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.backbone)
        self.g2p = G2PConverter()
        
        self.max_length = config.data.max_length
        self.is_train = is_train
        
        self.min_words = 4
        self.max_words = 12
        
        #  노이즈 확률 (총 25%)
        self.noise_delete_prob = 0.10
        self.noise_swap_prob = 0.10
        self.noise_random_prob = 0.05
        
        # 전체 음소 풀 (Random Substitution용)
        self.phoneme_vocab = list(set(list(KOREAN_CONFUSION_MAP.keys()) + list(ENGLISH_CONFUSION_MAP.keys())))

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

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        queries = []
        passages = []
        
        for item in batch:
            raw_text = item['raw_text']
            
            # -------------------------------------------------------
            #  RNG(난수 생성기) 결정
            # -------------------------------------------------------
            if self.is_train:
                # Train: 매번 새로운 난수 (다양성)
                rng = random 
            else:
                # Val: 텍스트 해시 기반 고정 난수 (일관성)
                # "거북선" -> 항상 똑같은 "커북선" 노이즈 생성됨
                seed_val = hash(raw_text)
                rng = random.Random(seed_val)

            # -------------------------------------------------------
            # 1. Query Generation (Slicing + G2P + Noise)
            # -------------------------------------------------------
            # (1) Span 추출
            query_span = self._create_query_span(raw_text, rng)
            
            # (2) G2P 변환
            query_phoneme = self.g2p(query_span)
            
            # (3) 노이즈 주입 (Train, Val 모두 적용하되 Val은 고정됨)
            #  이제 Val에서도 노이즈가 들어갑니다! (하지만 고정된 패턴으로)
            query_phoneme = self._add_phonetic_noise(query_phoneme, rng)
                
            queries.append(query_phoneme)

            # -------------------------------------------------------
            # 2. Passage Generation (Clean)
            # -------------------------------------------------------
            if item.get('passage_phoneme'):
                passages.append(item['passage_phoneme'])
            else:
                passages.append(self.g2p(raw_text))

        # Tokenization
        q_encodings = self.tokenizer(queries, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        p_encodings = self.tokenizer(passages, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')

        return {
            'q_input_ids': q_encodings['input_ids'],
            'q_attention_mask': q_encodings['attention_mask'],
            'p_input_ids': p_encodings['input_ids'],
            'p_attention_mask': p_encodings['attention_mask']
        }