import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import logging

# RetrievalModel 임포트
from .base import RetrievalModel

log = logging.getLogger(__name__)

# RetrievalModel 상속
class DPRModel(RetrievalModel):
    def __init__(self, model_name_or_path, device="cpu"):
        self.device = torch.device(device)
        log.info(f"Loading DPR model on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        self.passage_embeddings = None
        self.passage_ids = []
        self.corpus_contents = {}

    def _get_embedding(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # CLS 토큰 벡터 추출
            embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        return embeddings

    def build_index(self, corpus_data):
        log.info("Encoding corpus into vectors...")
        all_texts = [item['text'] for item in corpus_data]
        self.passage_ids = [item['passage_id'] for item in corpus_data]
        self.corpus_contents = {item['passage_id']: item['text'] for item in corpus_data}
        
        embeddings = []
        batch_size = 16 
        for i in tqdm(range(0, len(all_texts), batch_size), desc="Indexing"):
            batch = all_texts[i:i+batch_size]
            embeddings.append(self._get_embedding(batch))
            
        self.passage_embeddings = np.vstack(embeddings)

    def search(self, query: str, top_k: int):
        query_embedding = self._get_embedding([query])
        
        # 코사인 유사도 계산
        norm_q = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-9)
        norm_p = self.passage_embeddings / (np.linalg.norm(self.passage_embeddings, axis=1, keepdims=True) + 1e-9)
        
        scores = np.dot(norm_q, norm_p.T)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            pid = self.passage_ids[idx]
            results.append({
                "passage_id": pid,
                "score": float(scores[idx]),
                "text": self.corpus_contents.get(pid, "")
            })
        return results