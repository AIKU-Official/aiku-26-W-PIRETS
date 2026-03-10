"""
BM25Retriever - jaeseung distractor 스크립트 호환용.
커스텀 BM25 구현 (rank_bm25 의존 없이 직접 IDF/TF 계산).
"""
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

from src.text_utils import normalize_text


def _tokenize(text: str) -> List[str]:
    if text is None:
        return []
    return normalize_text(str(text)).split()


class BM25Retriever:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_lengths: Dict[str, int] = {}
        self.doc_tfs: Dict[str, Counter] = {}
        self.df: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}
        self.avg_doc_len = 0.0
        self.num_docs = 0

    def fit(self, documents: Iterable[Tuple[str, str]]) -> None:
        docs = list(documents)
        self.num_docs = len(docs)
        total_len = 0

        for doc_id, text in docs:
            tokens = _tokenize(text)
            tf = Counter(tokens)
            self.doc_tfs[doc_id] = tf
            self.doc_lengths[doc_id] = len(tokens)
            total_len += len(tokens)

            for term in tf.keys():
                self.df[term] += 1

        self.avg_doc_len = (total_len / self.num_docs) if self.num_docs > 0 else 0.0

        for term, df in self.df.items():
            self.idf[term] = math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))

    def _score(self, query_terms: List[str], doc_id: str) -> float:
        if self.num_docs == 0:
            return 0.0

        score = 0.0
        doc_len = self.doc_lengths.get(doc_id, 0)
        tf = self.doc_tfs.get(doc_id, Counter())

        for term in query_terms:
            if term not in tf:
                continue
            term_tf = tf[term]
            idf = self.idf.get(term, 0.0)
            denom = term_tf + self.k1 * (1 - self.b + self.b * (doc_len / (self.avg_doc_len + 1e-12)))
            score += idf * ((term_tf * (self.k1 + 1)) / (denom + 1e-12))

        return score

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_terms = _tokenize(query)
        ranked = [(doc_id, self._score(query_terms, doc_id)) for doc_id in self.doc_tfs.keys()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def batch_retrieve(self, queries: Dict[str, str], top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        results = {}
        for qid, qtext in queries.items():
            results[qid] = self.retrieve(qtext, top_k=top_k)
        return results
