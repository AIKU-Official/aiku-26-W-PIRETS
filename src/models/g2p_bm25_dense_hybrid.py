import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class G2PBM25DenseHybridRetriever(nn.Module):
    """
    G2P 기반 N-gram BM25(Lexical)와 Dense Retriever(Semantic)를 
    Min-Max 정규화 후 가중합(Weighted Sum) 방식으로 결합한 하이브리드 검색 모델.
    """
    def __init__(self, bm25_model, dense_retriever, alpha=0.5):
        super().__init__()
        self.bm25_model = bm25_model
        self.dense_retriever = dense_retriever
        self.alpha = alpha
        
        # evaluation 스크립트 출력용 네이밍
        self.backbone_name = f"G2PNgramHybrid_Alpha_{alpha}"

    def build_index(self, corpus_data, cache_path=None):
        logger.info("[G2P-Hybrid] Building BM25 N-Gram Index...")
        self.bm25_model.build_index(corpus_data)
        
        logger.info("[G2P-Hybrid] Building Dense Retriever Index...")
        self.dense_retriever.build_index(corpus_data, cache_path=cache_path)
        
        logger.info("[G2P-Hybrid] Both indices built successfully.")

    @staticmethod
    def _min_max_normalize(scores_dict):
        """점수 딕셔너리를 0~1 사이로 안전하게 Min-Max 정규화합니다."""
        if not scores_dict:
            return {}
            
        scores = list(scores_dict.values())
        min_score = min(scores)
        max_score = max(scores)
        
        # ZeroDivisionError 방어
        if max_score == min_score:
            return {k: 0.5 for k in scores_dict.keys()}
            
        return {k: float((v - min_score) / (max_score - min_score)) for k, v in scores_dict.items()}

    @torch.no_grad()
    def search(self, query_text, top_k=100):
        # 1. 탐색 후보 풀 확보
        search_k = top_k * 2 
        bm25_results = self.bm25_model.search(query_text, top_k=search_k)
        dense_results = self.dense_retriever.search(query_text, top_k=search_k)
        
        bm25_scores = {item['passage_id']: item['score'] for item in bm25_results}
        dense_scores = {item['passage_id']: item['score'] for item in dense_results}
        
        # 2. 메타데이터(text, phoneme) 보존
        id_to_text = {}
        id_to_phoneme = {} 
        
        for item in bm25_results + dense_results:
            pid = item['passage_id']
            id_to_text[pid] = item['text']
            if 'phoneme' in item:
                id_to_phoneme[pid] = item['phoneme']

        # 3. 이질적인 점수 스케일 통일 (0 ~ 1)
        norm_bm25 = self._min_max_normalize(bm25_scores)
        norm_dense = self._min_max_normalize(dense_scores)
        
        # 4. 합집합 생성 및 가중합 연산
        all_candidate_ids = set(norm_bm25.keys()).union(set(norm_dense.keys()))
        hybrid_results = []
        
        for pid in all_candidate_ids:
            # 한쪽 모델에서만 검색된 문서는 다른 모델 점수를 0점으로 처리
            s_dense = norm_dense.get(pid, 0.0)
            s_bm25 = norm_bm25.get(pid, 0.0)
            
            # 최종 스코어 = (알파 * Dense) + ((1-알파) * BM25)
            final_score = (self.alpha * s_dense) + ((1.0 - self.alpha) * s_bm25)
            
            result_dict = {
                'passage_id': pid,
                'score': final_score,
                'text': id_to_text[pid]
            }
            
            # Reranker용 음소 데이터 보존
            if pid in id_to_phoneme:
                result_dict['phoneme'] = id_to_phoneme[pid]
                
            hybrid_results.append(result_dict)
            
        # 5. 정렬 및 Top-K 절삭
        hybrid_results.sort(key=lambda x: x['score'], reverse=True)
        return hybrid_results[:top_k]