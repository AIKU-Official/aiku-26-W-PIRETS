from rank_bm25 import BM25Okapi
from .base import RetrievalModel
from ..utils import normalize_text
import numpy as np

class BM25Model(RetrievalModel):
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.passages = []

    def build_index(self, corpus):
        print(f"Building BM25 Index (k1={self.k1}, b={self.b})...")
        self.passages = corpus
        
        # 텍스트 토큰화 (띄어쓰기 기준)
        tokenized_corpus = [
            normalize_text(doc['text']).split() for doc in corpus
        ]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        print("Index Ready!")

    def search(self, query, top_k=10):
        if not self.bm25:
            raise ValueError("Index not built yet!")
            
        tokenized_query = normalize_text(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # 점수 높은 순으로 정렬하여 상위 top_k 추출
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n_indices:
            results.append({
                'passage_id': self.passages[idx]['passage_id'],
                'score': float(scores[idx]),
                'text': self.passages[idx]['text']
            })
        return results