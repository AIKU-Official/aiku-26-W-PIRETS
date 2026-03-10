from abc import ABC, abstractmethod

class RetrievalModel(ABC):
    @abstractmethod
    def build_index(self, corpus: list):
        """
        corpus: [{'passage_id': '...', 'text': '...'}, ...] 형태의 리스트
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int):
        """
        return: [{'passage_id': '...', 'score': 0.0}, ...] 형태의 리스트
        """
        pass