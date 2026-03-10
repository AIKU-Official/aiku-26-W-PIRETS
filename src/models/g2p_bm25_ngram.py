import numpy as np
import logging
from rank_bm25 import BM25Okapi
from .base import RetrievalModel

# G2PConverter 임포트
from src.g2p import G2PConverter

logger = logging.getLogger(__name__)

class BM25NGramModel(RetrievalModel):
    def __init__(self, n=3, **kwargs):
        super().__init__()
        self.n = n
        self.bm25 = None
        self.passages = []
        
        # G2P 변환기 초기화
        logger.info("Initializing G2PConverter inside BM25NGramModel...")
        self.g2p = G2PConverter()

    def _tokenize(self, text_or_phoneme):
        """문자열을 띄어쓰기 기준으로 나눈 뒤 n-gram 리스트로 변환합니다."""
        tokens = str(text_or_phoneme).split()
        if len(tokens) < self.n:
            return tokens 
        
        ngrams = []
        for i in range(len(tokens) - self.n + 1):
            ngram = "_".join(tokens[i:i + self.n])
            ngrams.append(ngram)
        return ngrams

    def build_index(self, corpus_data, cache_path=None, **kwargs):
        logger.info(f"Building BM25 {self.n}-Gram Index (Size={len(corpus_data)})...")
        self.passages = corpus_data
        
        tokenized_corpus = []
        for item in corpus_data:
            # 사전 변환된 phoneme이 있으면 사용하고, 없으면 on-the-fly 변환
            if 'phoneme' in item and item['phoneme']:
                target_str = item['phoneme']
            else:
                target_str = self.g2p(item.get('text', ''))
                
            tokenized_corpus.append(self._tokenize(target_str))
            
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 N-Gram Index Ready!")

    def search(self, query: str, top_k: int = 100):
        if not self.bm25:
            raise ValueError("Index not built yet!")
            
        # 원본 텍스트 쿼리를 G2P로 음소 변환
        query_phoneme = self.g2p(query)
        
        # 변환된 음소를 N-gram으로 토큰화
        tokenized_query = self._tokenize(query_phoneme)
        scores = self.bm25.get_scores(tokenized_query)
        
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n_indices:
            passage = self.passages[idx]
            result_dict = {
                "passage_id": passage['passage_id'],
                "score": float(scores[idx]),
                "text": passage.get('text', "")
            }
            
            # 후속 파이프라인을 위한 음소 데이터 보존
            if 'phoneme' in passage:
                result_dict['phoneme'] = passage['phoneme']
            else:
                # 원본 코퍼스에 없으면 on-the-fly 변환
                result_dict['phoneme'] = self.g2p(passage.get('text', ''))
                
            results.append(result_dict)
            
        return results