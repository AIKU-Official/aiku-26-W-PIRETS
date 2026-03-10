import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from src.g2p import G2PConverter
import logging

logger = logging.getLogger(__name__)

class BM25WithReranker(nn.Module):
    """
    Stage 1: BM25 (Lexical Retrieval - High Recall)
    Stage 2: Cross-Encoder (Semantic Reranking - High Precision)
    """
    def __init__(self, bm25_model, reranker, max_length=512):
        super().__init__()
        self.bm25_model = bm25_model
        self.reranker = reranker
        self.g2p = G2PConverter()
        
        # Cross-Encoder 입력을 위한 Tokenizer 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(reranker.backbone_name)
        self.max_length = max_length
        
        self.backbone_name = "BM25_CrossEncoder_Pipeline"

    # 파이프라인 전용 체크포인트 로드 메서드
    def load_checkpoints(self, retriever_path=None, reranker_path=None):
        """
        BM25는 가중치 파일이 없으므로, reranker_path만 활용하여 모델을 로드합니다.
        """
        if reranker_path and os.path.exists(reranker_path):
            logger.info(f"Loading Reranker checkpoint from: {reranker_path}")
            state_dict = torch.load(reranker_path, map_location='cpu')
            # DDP 훈련 시 추가되는 'module.' 접두어 제거
            state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
            self.reranker.load_state_dict(state_dict, strict=False)
            logger.info("Reranker checkpoint loaded successfully!")
        elif reranker_path:
            logger.warning(f"Reranker checkpoint not found: {reranker_path}")

    # build_index: BM25 인덱싱 후, Reranker용 캐시를 메모리에 탑재
    def build_index(self, corpus_data, cache_path=None):
        logger.info("[Pipeline] Building BM25 Index...")
        self.bm25_model.build_index(corpus_data)
        logger.info("[Pipeline] BM25 Index built successfully.")
        
        # Reranker용 G2P phoneme 사전 구축
        self.pid2phoneme = {}
        if cache_path and os.path.exists(cache_path):
            cache_data = torch.load(cache_path, map_location='cpu')
            cached_ids = cache_data.get('ids', [])
            cached_phonemes = cache_data.get('phonemes', [])
            
            if cached_ids and cached_phonemes:
                self.pid2phoneme = dict(zip(cached_ids, cached_phonemes))
                logger.info(f"Pre-loaded {len(self.pid2phoneme)} phonemes into memory for fast Reranking.")
        else:
            logger.warning("No cache found. Reranker will compute G2P on the fly (slow).")
            
    @torch.no_grad()
    def search(self, query_text, top_k=100):
        search_k = top_k * 2
        candidates = self.bm25_model.search(query_text, top_k=search_k)
        
        if not candidates:
            return []

        self.reranker.eval()
        device = next(self.reranker.parameters()).device
        
        # 쿼리 G2P 변환
        q_phoneme = self.g2p(query_text)
        pairs = []
        
        for cand in candidates:
            pid = cand['passage_id']
            
            # Phoneme fallback: cand -> pid2phoneme 캐시 -> on-the-fly G2P
            p_phoneme = cand.get('phoneme') or getattr(self, 'pid2phoneme', {}).get(pid) or self.g2p(cand['text'])
            
            pairs.append((q_phoneme, p_phoneme))

        # 3. Tokenization: RoBERTa 규격에 맞게 <s> Query </s></s> Passage </s> 구조 형성
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(device)

        # 4. Stage-2 Reranking: Forward Pass를 통한 정밀 문맥 점수 산출
        logits = self.reranker(**inputs)
        
        # Sigmoid를 통과시켜 0~1 사이의 확률값으로 변환
        scores = torch.sigmoid(logits).squeeze(-1).cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]

        # 5. 기존 BM25 점수를 Reranker 점수로 완전히 덮어쓰기 (Update)
        for idx, cand in enumerate(candidates):
            cand['score'] = scores[idx]

        # 6. Reranker의 최종 점수 기준 내림차순 정렬 및 Top-K 절삭
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_k]