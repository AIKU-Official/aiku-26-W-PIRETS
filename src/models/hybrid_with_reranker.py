import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from src.g2p import G2PConverter
import logging

logger = logging.getLogger(__name__)

class HybridWithReranker(nn.Module):
    """
    Stage 1: BM25 + Dense Hybrid (Lexical & Semantic Ensemble)
    Stage 2: Cross-Encoder (Semantic Reranking)
    """
    def __init__(self, hybrid_retriever, reranker, max_length=512):
        super().__init__()
        # 1차 검색기: BM25와 Dense가 결합된 Hybrid 모델을 통째로 받습니다.
        self.hybrid_retriever = hybrid_retriever
        
        # 2차 검색기: Reranker
        self.reranker = reranker
        self.g2p = G2PConverter()
        
        # Cross-Encoder 입력을 위한 Tokenizer 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(reranker.backbone_name)
        self.max_length = max_length
        
        self.backbone_name = "Hybrid_CrossEncoder_Pipeline"

    # 파이프라인 전용 체크포인트 로드
    def load_checkpoints(self, retriever_path=None, reranker_path=None):
        """
        하이브리드 모델 내부의 Dense Retriever와, Reranker 모델의 가중치를 각각 로드합니다.
        BM25는 가중치 파일이 없으므로 안전하게 패스됩니다.
        """
        # 1. 하이브리드 내부의 Dense Retriever 가중치 로드
        if retriever_path and os.path.exists(retriever_path):
            logger.info(f"Loading Dense Retriever checkpoint from: {retriever_path}")
            state_dict = torch.load(retriever_path, map_location='cpu')
            state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

            # Hybrid 모델 내부의 dense_retriever 속성에 접근하여 로드
            if hasattr(self.hybrid_retriever, 'dense_retriever'):
                self.hybrid_retriever.dense_retriever.load_state_dict(state_dict, strict=False)
                logger.info("Dense Retriever checkpoint loaded successfully!")
            else:
                logger.warning("'dense_retriever' attribute not found in hybrid_retriever.")
        elif retriever_path:
            logger.warning(f"Dense Retriever checkpoint not found: {retriever_path}")

        # 2. Reranker 가중치 로드
        if reranker_path and os.path.exists(reranker_path):
            logger.info(f"Loading Reranker checkpoint from: {reranker_path}")
            state_dict = torch.load(reranker_path, map_location='cpu')
            state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
            self.reranker.load_state_dict(state_dict, strict=False)
            logger.info("Reranker checkpoint loaded successfully!")
        elif reranker_path:
            logger.warning(f"Reranker checkpoint not found: {reranker_path}")

    def build_index(self, corpus_data, cache_path=None):
        logger.info("[Pipeline] Building Hybrid Indices (BM25 + Dense)...")
        # 내부의 Dense Retriever가 cache_path에 .pt 파일을 저장
        self.hybrid_retriever.build_index(corpus_data, cache_path=cache_path)
        logger.info("[Pipeline] Hybrid Indices built successfully.")
        
        # 캐시에서 phoneme 데이터를 로드하여 딕셔너리 구축
        self.pid2phoneme = {}
        if cache_path and os.path.exists(cache_path):
            cache_data = torch.load(cache_path, map_location='cpu')
            cached_ids = cache_data.get('ids', [])
            cached_phonemes = cache_data.get('phonemes', [])
            
            if cached_ids and cached_phonemes:
                self.pid2phoneme = dict(zip(cached_ids, cached_phonemes))
                logger.info(f"Pre-loaded {len(self.pid2phoneme)} phonemes into memory for fast Reranking.")

    @torch.no_grad()
    def search(self, query_text, top_k=100):
        search_k = top_k * 2
        candidates = self.hybrid_retriever.search(query_text, top_k=search_k)
        
        if not candidates: return []

        self.reranker.eval()
        device = next(self.reranker.parameters()).device
        
        q_phoneme = self.g2p(query_text)
        pairs = []
        
        for cand in candidates:
            pid = cand['passage_id']
            # Phoneme fallback: cand -> pid2phoneme 캐시 -> on-the-fly G2P
            p_phoneme = cand.get('phoneme') or getattr(self, 'pid2phoneme', {}).get(pid) or self.g2p(cand['text'])
            
            pairs.append((q_phoneme, p_phoneme))

        # 3. Tokenization: <s> Query </s></s> Passage </s> 포맷 구성
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(device)

        # 4. Stage-2 Reranking
        logits = self.reranker(**inputs)
        scores = torch.sigmoid(logits).squeeze(-1).cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]

        # 5. 1차 점수를 Reranker 점수로 업데이트
        for idx, cand in enumerate(candidates):
            cand['score'] = scores[idx]

        # 6. Reranker 점수 기준 내림차순 정렬 및 Top-K 반환
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_k]