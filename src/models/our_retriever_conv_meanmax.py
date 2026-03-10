import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from tqdm import tqdm
import numpy as np
from src.g2p import G2PConverter
from src.models.base import RetrievalModel 

class OurRetrieverConvMeanMax(nn.Module, RetrievalModel):
    def __init__(self, config):
        # RetrievalModel의 __init__이 있으면 호출
        super().__init__() 
        self.config = config
        
        # 1. 모델 로드
        self.backbone_name = config.backbone 
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone_name)
        model_config = AutoConfig.from_pretrained(self.backbone_name)
        
        #  [핵심 수정] "Pooler Layer는 아예 생성하지 마!" 옵션 추가
        # Max Pooling을 쓸 거면 이 옵션을 False로 줘서 파라미터 자체를 없애야 합니다.
        if not config.get('use_pooler', False):
            model_config.add_pooling_layer = False
        
        self.encoder = AutoModel.from_pretrained(self.backbone_name, config=model_config)
        
        #  [추가] Conv1d (3-gram) 레이어 정의
        self.conv1d = nn.Conv1d(
            in_channels=768, 
            out_channels=768, 
            kernel_size=3,   # 3-gram
            padding=1        # 길이 유지
        )
        self.activation = nn.ReLU()
        
        # 2. G2P 변환기
        self.g2p = G2PConverter()
        
        # 3. Projection (옵션)
        # config에 projection_dim이 있으면 Linear 레이어 추가
        self.projection_dim = config.get('projection_dim', None)
        if self.projection_dim:
            self.projector = nn.Linear(model_config.hidden_size, self.projection_dim)
        else:
            self.projector = None
            
        # 4. Temperature 설정 (Loss 계산용)
        # config에서 불러오거나 없으면 0.05 (Contrastive Loss 기본값) 사용
        self.temperature = config.get('temperature', 0.05)
        
        # 5. 검색을 위한 저장소 (Index)
        self.doc_embeddings = None
        self.doc_ids = []
        self.doc_texts = [] 

    def _max_pooling(self, token_embeddings, attention_mask):
        """
        Max Pooling: 문장 내에서 가장 강력한 신호(Peak) 하나만 뽑아냅니다.
        """
        # 마스크 확장: [B, L] -> [B, L, D]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        #  중요: 패딩(Padding) 토큰 처리
        # Mean Pooling은 0으로 만들지만, Max Pooling은 -10억(-1e9)으로 만들어야 
        # 절대 선택되지 않습니다. (0으로 하면 음수 값이 다 무시됨)
        min_value = torch.finfo(token_embeddings.dtype).min
        token_embeddings = token_embeddings.masked_fill(input_mask_expanded == 0, min_value)
        
        # Max 연산 수행 (값만 가져옴)
        return torch.max(token_embeddings, 1)[0]
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """
        토큰 임베딩의 평균을 구하는 함수 (Padding 제외)
        """
        # 삭졔: token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

    def encode(self, input_ids, attention_mask):
        """
        입력을 인코딩하고 정규화된 임베딩을 반환합니다.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state # [B, Seq_Len, 768]
        
        # Conv1d 적용 (N-gram Feature 추출)
        # (1) 차원 뒤집기: [B, L, D] -> [B, D, L]
        token_embeddings = token_embeddings.transpose(1, 2)
        
        # (2) 3-gram Conv + ReLU
        token_embeddings = self.conv1d(token_embeddings)
        token_embeddings = self.activation(token_embeddings)
        
        # (3) 다시 원상복구: [B, D, L] -> [B, L, D]
        #  이제 이 token_embeddings는 Conv가 찾아낸 패턴 덩어리입니다.
        token_embeddings = token_embeddings.transpose(1, 2)
        
        #  [핵심 수정] Pooling 방식: Mean과 Max를 각각 구해서 1:1로 더하기
        if self.config.get('use_pooler', False):
            # pooler를 쓴다면 첫 번째 토큰([CLS])만 가져옴 (현재 설정상 사용 안 함)
            embeddings = token_embeddings[:, 0, :]
        else:
            # 1. 평균(Mean) 풀링 
            mean_emb = self._mean_pooling(token_embeddings, attention_mask)
            
            # 2. 최대(Max) 풀링
            max_emb = self._max_pooling(token_embeddings, attention_mask)
            
            # 3. 1:1 단순 덧셈 (강력한 앙상블 효과)
            embeddings = mean_emb + max_emb
        
        # Projection 적용 (설정된 경우)
        if getattr(self, 'projector', None) is not None:
            embeddings = self.projector(embeddings)
            
        # 정규화 (Cosine Similarity를 위해 필수)
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, batch):
        """
        학습용 Forward (Trainer에서 호출)
        InfoNCE Loss 계산
        """
        # 쿼리와 패시지 인코딩
        q_emb = self.encode(batch['q_input_ids'], batch['q_attention_mask'])
        p_emb = self.encode(batch['p_input_ids'], batch['p_attention_mask'])
        
        # 유사도 점수 계산 (Batch_Size x Batch_Size)
        # temperature로 나누어 점수 분포를 조정
        sim_scores = torch.matmul(q_emb, p_emb.T) / self.temperature
        
        # 정답 라벨 생성 (대각선이 정답: 0, 1, 2...)
        labels = torch.arange(q_emb.size(0), device=q_emb.device, dtype=torch.long)
        
        # Loss 계산
        loss = F.cross_entropy(sim_scores, labels)
        
        return {'loss': loss, 'scores': sim_scores}

    # =========================================================
    #  RetrievalModel 추상 메서드 구현 (Build Index & Search)
    # =========================================================

    @torch.no_grad()
    def build_index(self, corpus_data):
        self.eval()
        
        # 현재 모델이 있는 디바이스 확인
        device = next(self.parameters()).device
        
        doc_embs_list = []
        self.doc_ids = []
        self.doc_texts = []
        
        batch_size = 128 
        
        for i in tqdm(range(0, len(corpus_data), batch_size), desc="Building Index"):
            batch = corpus_data[i : i + batch_size]
            texts = [item['text'] for item in batch]
            ids = [item['passage_id'] for item in batch]
            
            self.doc_ids.extend(ids)
            self.doc_texts.extend(texts)
            
            # 1. G2P
            phonemes = [self.g2p(t) for t in texts]
            
            # 2. Tokenize
            inputs = self.tokenizer(
                phonemes, padding=True, truncation=True, max_length=128, return_tensors='pt'
            ).to(device)
            
            # 3. Encode
            emb = self.encode(inputs['input_ids'], inputs['attention_mask'])
            doc_embs_list.append(emb.cpu()) 
            
        self.doc_embeddings = torch.cat(doc_embs_list, dim=0)
        print(f" Index Built: {self.doc_embeddings.size()}")

    @torch.no_grad()
    def search(self, query_text, top_k=100):
        self.eval()
        device = next(self.parameters()).device
        
        # 1. Query Encoding
        q_phoneme = self.g2p(query_text)
        q_inputs = self.tokenizer(
            [q_phoneme], padding=True, truncation=True, max_length=128, return_tensors='pt'
        ).to(device)
        
        q_emb = self.encode(q_inputs['input_ids'], q_inputs['attention_mask']) 
        
        # 2. Similarity
        doc_embs_gpu = self.doc_embeddings.to(device) 
        scores = torch.matmul(q_emb, doc_embs_gpu.T).squeeze(0) 
        
        # 3. Top-K
        top_scores, top_indices = torch.topk(scores, k=min(top_k, len(scores)))
        
        top_scores = top_scores.cpu().tolist()
        top_indices = top_indices.cpu().tolist()
        
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                'passage_id': self.doc_ids[idx],
                'score': top_scores[rank],
                'text': self.doc_texts[idx]
            })
            
        return results