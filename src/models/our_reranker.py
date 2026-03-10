import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class OurCrossEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 모델 아키텍처 로드
        self.backbone_name = config.get('backbone', 'vinai/xphonebert-base')
        model_config = AutoConfig.from_pretrained(self.backbone_name)
        
        # 2. 트랜스포머 인코더 생성
        self.encoder = AutoModel.from_pretrained(self.backbone_name, config=model_config)
        
        # 3.  [핵심 추가] 과적합 방지용 드롭아웃 (Dropout)
        # 모델의 기본 설정값(주로 0.1)을 가져와 적용합니다.
        self.dropout = nn.Dropout(model_config.hidden_dropout_prob)
        
        # 4. Reranker용 분류기 헤드 (Classifier Head)
        self.classifier = nn.Linear(model_config.hidden_size, 1)

    def forward(self, batch):
        """
        Cross-Encoder의 Forward 연산
        입력: 토크나이저가 [CLS] Query [SEP] Passage [SEP] 형태로 묶어준 텐서들
        출력: Batch 크기만큼의 1차원 점수 (Logits)
        """
        # 1. 트랜스포머 통과 (두 문장이 섞인 채로 Attention 연산 발생)
        outputs = self.encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            # RoBERTa 기반 모델은 token_type_ids를 쓰지 않으므로 에러 방지용 get 사용
            token_type_ids=batch.get('token_type_ids', None) 
        )
        
        # 2. [CLS] 토큰 추출 (전체 문맥을 요약한 첫 번째 토큰)
        # shape: (Batch_Size, Seq_Len, Hidden_Dim) -> (Batch_Size, Hidden_Dim)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # 3.  [핵심 추가] 드롭아웃 적용
        cls_embedding = self.dropout(cls_embedding)
        
        # 4. 1차원 점수로 투영 (Logit)
        # shape: (Batch_Size, Hidden_Dim) -> (Batch_Size, 1) -> (Batch_Size)
        logits = self.classifier(cls_embedding).squeeze(-1)
        
        return logits