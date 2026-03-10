import os
import torch
import glob
import math
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder

logger = logging.getLogger(__name__)

class SemanticPipelineRetriever:
    """
    1차 검색기(E5 Bi-Encoder)로 후보군을 추출하고, 
    2차 검색기(BGE Cross-Encoder Reranker)로 순위를 재조정하는 파이프라인.
    """
    def __init__(self, e5_model_name="intfloat/multilingual-e5-large", 
                 bge_reranker_name="BAAI/bge-reranker-v2-m3", device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone_name = f"E5_Retriever_BGE_Reranker"
        
        # 1차 검색기: E5 (Bi-Encoder)
        logger.info(f"Loading Retriever (E5): {e5_model_name} on {self.device}")
        self.retriever = SentenceTransformer(e5_model_name, device=self.device)
        
        # 2차 검색기: BGE Reranker (Cross-Encoder)
        logger.info(f"Loading Reranker (BGE): {bge_reranker_name} on {self.device}")
        # BGE 리랭커는 CrossEncoder 클래스를 사용
        self.reranker = CrossEncoder(bge_reranker_name, device=self.device)
        
        self.corpus_ids = []
        self.corpus_texts = []
        self.corpus_embeddings = None

    def build_index(self, corpus, cache_path=None, chunk_size=500):
        """
        Chunk 단위로 인코딩을 수행하여 OOM을 방지하고,
        중단 시 이어달리기를 지원하는 Indexing 함수.
        """
        self.corpus = corpus
        
        # 임베딩 연산 전에 ID와 텍스트 풀 초기화
        self.corpus_ids = [doc['passage_id'] for doc in corpus]
        self.corpus_texts = [doc['text'] for doc in corpus]
        
        # 1. 완전한 마스터 캐시가 이미 존재하면 즉시 로드
        if cache_path and os.path.exists(cache_path):
            try:
                embeddings = torch.load(cache_path)
                if len(embeddings) == len(corpus):
                    self.corpus_embeddings = embeddings
                    logger.info(f"마스터 캐시 로드 완료: {len(embeddings)}건")
                    return
            except Exception as e:
                logger.warning(f"기존 캐시 파일 손상. 복구 및 덮어쓰기를 진행합니다. ({e})")
        
        logger.info(f"텍스트 {len(corpus)}개에 대한 분할 임베딩을 시작합니다. (Chunk 크기: {chunk_size})")
        texts = [doc['text'] for doc in corpus]
        all_embeddings = []
        
        # 2. 임시 청크(Chunk) 저장용 독립 폴더 생성
        cache_dir = os.path.dirname(cache_path) if cache_path else "outputs/cache"
        base_name = os.path.basename(cache_path).replace('.pt', '') if cache_path else "embed"
        chunk_dir = os.path.join(cache_dir, f"{base_name}_chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        
        num_chunks = math.ceil(len(texts) / chunk_size)
        
        for i in range(num_chunks):
            chunk_file = os.path.join(chunk_dir, f"chunk_{i}.pt")
            
            if os.path.exists(chunk_file):
                logger.info(f"[Resume] 청크 {i+1}/{num_chunks} 복구 완료 ({chunk_file})")
                chunk_emb = torch.load(chunk_file)
            else:
                logger.info(f"청크 {i+1}/{num_chunks} 인코딩 시작...")
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(texts))
                chunk_texts = texts[start_idx:end_idx]
                
                # 진행률 바 활성화
                chunk_emb = self.retriever.encode(chunk_texts, show_progress_bar=True, convert_to_tensor=True)
                
                torch.save(chunk_emb.cpu(), chunk_file)
                logger.info(f"청크 {i+1} 임시 저장 완료.")
            
            # 로드 또는 새로 인코딩한 청크를 리스트에 추가
            all_embeddings.append(chunk_emb)
            
        # 4. 모든 청크 벡터를 하나의 거대한 텐서로 병합
        self.corpus_embeddings = torch.cat(all_embeddings, dim=0)
        
        # 5. 최종 마스터 캐시 저장 및 임시 청크 파일 정리
        if cache_path:
            torch.save(self.corpus_embeddings, cache_path)
            logger.info(f"최종 마스터 캐시 저장 완료: {cache_path}")
            
            for f in glob.glob(os.path.join(chunk_dir, "*.pt")):
                os.remove(f)
            os.rmdir(chunk_dir)
            logger.info("임시 청크 디렉토리 정리 완료")
            
    @torch.no_grad()
    def search(self, query_text: str, top_k: int = 100):
        # Stage 1: E5 Retriever로 후보 풀 확보
        search_k = top_k * 2 
        e5_query = f"query: {query_text}"
        q_emb = self.retriever.encode(e5_query, convert_to_tensor=True, normalize_embeddings=True)
        
        # corpus_embeddings를 q_emb와 같은 device로 이동
        if self.corpus_embeddings.device != q_emb.device:
            self.corpus_embeddings = self.corpus_embeddings.to(q_emb.device)
        
        scores = torch.matmul(q_emb, self.corpus_embeddings.T).squeeze(0)
        actual_k = min(search_k, len(self.corpus_ids))
        top_scores, top_indices = torch.topk(scores, actual_k)
        
        # 1차 후보군 추출
        first_stage_candidates = []
        for idx in top_indices.tolist():
            first_stage_candidates.append({
                'passage_id': self.corpus_ids[idx],
                'text': self.corpus_texts[idx]
            })

        # Stage 2: BGE Reranker (prefix 없이 순수 텍스트 쌍 입력)
        cross_inp = [[query_text, item['text']] for item in first_stage_candidates]
        cross_scores = self.reranker.predict(cross_inp)
        
        for i, score in enumerate(cross_scores):
            first_stage_candidates[i]['score'] = float(score)
            
        # 최종 BGE 점수로 재정렬 및 Top-K 절삭
        first_stage_candidates.sort(key=lambda x: x['score'], reverse=True)
        return first_stage_candidates[:top_k]