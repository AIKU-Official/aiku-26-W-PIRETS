import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class BM25DenseHybridRetriever(nn.Module):
    """
    BM25(Lexical)와 Dense Retriever(Semantic)를 결합한 하이브리드 검색 모델.
    """
    def __init__(self, bm25_model, dense_retriever, alpha=0.5):
        super().__init__()
        self.bm25_model = bm25_model
        self.dense_retriever = dense_retriever
        
        # Dense의 가중치(alpha). BM25의 가중치는 (1 - alpha)가 됩니다.
        self.alpha = alpha
        
        # evaluation 스크립트에서 이름 출력을 위해 속성 부여
        self.backbone_name = f"Hybrid_Alpha_{alpha}"

    def build_index(self, corpus_data, cache_path=None):
        """
        두 검색기의 인덱스를 각각 구축합니다.
        """
        logger.info("[Hybrid] Building BM25 Index...")
        # BM25는 캐시 파라미터를 받지 않으므로 corpus만 전달
        self.bm25_model.build_index(corpus_data)
        
        logger.info("[Hybrid] Building Dense Retriever Index...")
        # Dense 모델은 cache_path를 받을 수 있습니다.
        self.dense_retriever.build_index(corpus_data, cache_path=cache_path)
        
        logger.info("[Hybrid] Both indices built successfully.")

    @staticmethod
    def _min_max_normalize(scores_dict):
        """
        주어진 점수 딕셔너리를 0~1 사이로 Min-Max 정규화합니다.
        """
        if not scores_dict:
            return {}
            
        scores = list(scores_dict.values())
        min_score = min(scores)
        max_score = max(scores)
        
        # 모든 점수가 같을 경우 분모가 0이 되는 것을 방지 (ZeroDivisionError)
        if max_score == min_score:
            return {k: 0.5 for k in scores_dict.keys()}
            
        return {k: (v - min_score) / (max_score - min_score) for k, v in scores_dict.items()}

    @torch.no_grad()
    def search(self, query_text, top_k=100):
        search_k = top_k * 2 
        bm25_results = self.bm25_model.search(query_text, top_k=search_k)
        dense_results = self.dense_retriever.search(query_text, top_k=search_k)
        
        bm25_scores = {item['passage_id']: item['score'] for item in bm25_results}
        dense_scores = {item['passage_id']: item['score'] for item in dense_results}
        
        # 2. passage_id -> text 및 phoneme 매핑 보존 (결과 리턴용)
        id_to_text = {}
        id_to_phoneme = {} # Dense에서 가져온 음소 데이터 보존용
        
        for item in bm25_results + dense_results:
            id_to_text[item['passage_id']] = item['text']
            if 'phoneme' in item:
                id_to_phoneme[item['passage_id']] = item['phoneme']

        # 3. 각 점수 공간(Score Space) 정규화
        norm_bm25 = self._min_max_normalize(bm25_scores)
        norm_dense = self._min_max_normalize(dense_scores)
        
        # 4. 합집합(Union) 생성 및 점수 결합 (Convex Combination)
        all_candidate_ids = set(norm_bm25.keys()).union(set(norm_dense.keys()))
        
        hybrid_results = []
        for pid in all_candidate_ids:
            s_dense = norm_dense.get(pid, 0.0)
            s_bm25 = norm_bm25.get(pid, 0.0)
            
            final_score = (self.alpha * s_dense) + ((1.0 - self.alpha) * s_bm25)
            
            result_dict = {
                'passage_id': pid,
                'score': final_score,
                'text': id_to_text[pid]
            }
            
            # Reranker에서 사용할 수 있도록 음소를 결과에 포함
            if pid in id_to_phoneme:
                result_dict['phoneme'] = id_to_phoneme[pid]
                
            hybrid_results.append(result_dict)
            
        # 5. 최종 점수 기준 내림차순 정렬 및 Top-K 절삭
        hybrid_results.sort(key=lambda x: x['score'], reverse=True)
        return hybrid_results[:top_k]