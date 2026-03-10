import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from src.g2p import G2PConverter
import logging

logger = logging.getLogger(__name__)

class RetrieverWithReranker(nn.Module):
    """
    1차 검색기(Lexical, Dense, Hybrid 무관)의 결과에 
    Cross-Encoder 기반 Reranker를 부착하여 순위를 재조정하는 범용 파이프라인.
    """
    def __init__(self, retriever, reranker, max_length=512, **kwargs):
        super().__init__()
        # 범용 retriever 수용 (타입 제한 없음)
        self.retriever = retriever
        self.reranker = reranker
        self.g2p = G2PConverter()
        
        # evaluate_final.py 로깅 및 캐시 파일명 생성을 위한 동적 이름 할당
        ret_name = getattr(retriever, 'backbone_name', 'UnknownRetriever')
        self.backbone_name = f"Reranker_over_{ret_name}"
        
        self.tokenizer = AutoTokenizer.from_pretrained(reranker.backbone_name)
        self.max_length = max_length

    def load_checkpoints(self, retriever_path=None, reranker_path=None):
        if retriever_path and os.path.exists(retriever_path):
            logger.info(f"Loading Retriever checkpoint from: {retriever_path}")
            state_dict = torch.load(retriever_path, map_location='cpu')
            state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

            # Lexical 모델은 state_dict가 없으므로 건너뛰기
            if hasattr(self.retriever, 'load_state_dict'):
                self.retriever.load_state_dict(state_dict, strict=False)
                logger.info("Retriever checkpoint loaded!")
            else:
                logger.info("Lexical Retriever bypassed checkpoint loading.")
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

    def build_index(self, corpus_data, cache_path=None, **kwargs):
        # 파이프라인 인자를 하위 retriever에 전달
        if hasattr(self.retriever, 'build_index'):
            self.retriever.build_index(corpus_data, cache_path=cache_path, **kwargs)

    @torch.no_grad()
    def search(self, query_text, top_k=100):
        # 정확히 top_k개만 추출
        candidates = self.retriever.search(query_text, top_k=top_k)
        
        if not candidates:
            return []

        self.reranker.eval()
        device = next(self.reranker.parameters()).device
        
        q_phoneme = self.g2p(query_text)
        pairs = []
        
        for cand in candidates:
            # Phoneme fallback: cand에 없으면 on-the-fly G2P 변환
            p_phoneme = cand.get('phoneme')
            if not p_phoneme:
                p_phoneme = self.g2p(cand.get('text', ''))
            pairs.append((q_phoneme, p_phoneme))

        batch_size = 32
        all_scores = []

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(device)

            logits = self.reranker(**inputs)
            
            # HuggingFace 객체 반환 시 텐서 추출 방어 로직
            if hasattr(logits, 'logits'):
                logits = logits.logits
                
            scores = torch.sigmoid(logits).squeeze(-1).cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
                
            all_scores.extend(scores)
            del inputs, logits

        for idx, cand in enumerate(candidates):
            cand['score'] = all_scores[idx]

        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 최종적으로 요청받은 top_k 개수만큼만 절삭하여 반환
        return candidates[:top_k]