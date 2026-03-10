import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from src.g2p import G2PConverter

from src.models.our_retriever_conv_meanmax import OurRetrieverConvMeanMax
from src.models.our_reranker import OurCrossEncoder
import logging

logger = logging.getLogger(__name__)

class ConvMeanMaxWithReranker(nn.Module):
    def __init__(self, retriever: OurRetrieverConvMeanMax, reranker: OurCrossEncoder, max_length=512):
        super().__init__()
        self.retriever = retriever
        self.reranker = reranker
        self.g2p = G2PConverter()
        
        self.tokenizer = AutoTokenizer.from_pretrained(reranker.backbone_name)
        self.max_length = max_length

    # 두 개의 체크포인트를 독립적으로 로드하는 파이프라인 전용 메서드
    def load_checkpoints(self, retriever_path=None, reranker_path=None):
        if retriever_path and os.path.exists(retriever_path):
            logger.info(f"Loading Retriever checkpoint from: {retriever_path}")
            state_dict = torch.load(retriever_path, map_location='cpu')
            state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
            self.retriever.load_state_dict(state_dict, strict=False)
            logger.info("Retriever checkpoint loaded!")
        elif retriever_path:
            logger.warning(f"Retriever checkpoint not found: {retriever_path}")

        if reranker_path and os.path.exists(reranker_path):
            logger.info(f"Loading Reranker checkpoint from: {reranker_path}")
            state_dict = torch.load(reranker_path, map_location='cpu')
            state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
            self.reranker.load_state_dict(state_dict, strict=False)
            logger.info("Reranker checkpoint loaded!")
        elif reranker_path:
            logger.warning(f"Reranker checkpoint not found: {reranker_path}")

    def build_index(self, corpus_data, cache_path=None):
        # 1차 검색기인 Dense Retriever에 인덱싱 위임
        self.retriever.build_index(corpus_data, cache_path=cache_path)

    @torch.no_grad()
    def search(self, query_text, top_k=100):
        # 1. Stage-1 Retrieval: 1차 후보군 추출
        candidates = self.retriever.search(query_text, top_k=top_k)
        
        if not candidates:
            return []

        self.reranker.eval()
        device = next(self.reranker.parameters()).device
        
        # 2. Cross-Encoder용 Input 생성
        q_phoneme = self.g2p(query_text)
        pairs = []
        
        for cand in candidates:
            # Retriever가 캐시에서 가져온 음소를 그대로 사용
            p_phoneme = cand['phoneme'] 
            pairs.append((q_phoneme, p_phoneme))

        # Mini-batching으로 Reranker 추론
        batch_size = 32
        all_scores = []

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            
            # 1. 미니 배치 단위로 Tokenization
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(device)

            # 2. 미니 배치 Forward Pass
            logits = self.reranker(**inputs)
            
            # 3. 점수 추출 및 리스트 누적
            scores = torch.sigmoid(logits).squeeze(-1).cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
                
            all_scores.extend(scores)
            
            # 4. 사용 완료된 텐서 해제
            del inputs, logits

        # 5. 분할 계산된 전체 점수로 Candidate 정보 업데이트
        for idx, cand in enumerate(candidates):
            cand['score'] = all_scores[idx]

        # 6. 최종 점수 기준 내림차순 정렬
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates[:top_k]