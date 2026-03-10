import os
#  [1. 완벽한 단일 GPU 격리 및 멀티스레딩 충돌 원천 차단]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from omegaconf import OmegaConf
from src.models import get_model
import random
from src.g2p import G2PConverter
import gc

#  PyTorch 내부 C++ 스레드 1개로 강제 고정 (CPU Busy-wait 방지)
torch.set_num_threads(1)

# ==========================================
# 1. Config Loader
# ==========================================
def load_config(yaml_path="conf/mine_config.yaml"):
    print(f" Loading configuration from {yaml_path}...")
    return OmegaConf.load(yaml_path)

# ==========================================
# 2. Data Loading
# ==========================================
def load_corpus(tsv_path, jsonl_path):
    print(f" Loading Corpus from {tsv_path} & {jsonl_path}...")
    corpus = {}
    df = pd.read_csv(tsv_path, sep='\t')
    for _, row in df.iterrows():
        corpus[str(row['id'])] = {'raw': str(row['text']), 'g2p': ''}
        
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            doc_id = str(data['id'])
            if doc_id in corpus:
                corpus[doc_id]['g2p'] = str(data['phoneme'])
    return corpus

# ==========================================
# 3. Indexing & Embedding
# ==========================================
def build_bm25_index(corpus):
    print(" Building BM25 Index...")
    doc_ids = list(corpus.keys())
    tokenized_corpus = [corpus[doc_id]['raw'].split(" ") for doc_id in doc_ids]
    print(" BM25")
    return BM25Okapi(tokenized_corpus), doc_ids

def load_dense_model(config_dict, model_path, device):
    model_type = config_dict.get('name', 'Unknown Model')
    print(f" Loading Dense Model [{model_type}]...")
    
    model = get_model(config_dict)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f" Successfully loaded checkpoint from {model_path}")
    else:
        raise FileNotFoundError(f" Error: Checkpoint not found at '{model_path}'.")
        
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def get_dense_embeddings(model, texts, device, batch_size=256, desc="Encoding"):
    num_texts = len(texts)
    all_embs = None
    
    #  [Memory Mapping] 리스트 append로 인한 RAM 파편화/brk() 데드락 방지
    mmap_file = f"temp_{desc.replace(' ', '_')}.mmap"
    if os.path.exists(mmap_file):
        os.remove(mmap_file)

    for i in tqdm(range(0, num_texts, batch_size), desc=desc):
        batch_texts = texts[i:i+batch_size]
        
        #  [Tokenizer 정규식 길이 방어] 1000글자로 강제 슬라이싱
        safe_texts = [str(t)[:1000] for t in batch_texts]
        
        inputs = model.tokenizer(safe_texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        
        #  [AMP 적용] VRAM 및 TITAN Xp 전력 부하 감소
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            embs = model.encode(inputs['input_ids'], inputs['attention_mask'])
            
        embs_cpu = embs.detach().cpu().numpy().astype('float32')
        
        #  [동적 차원 감지] OmegaConf 에러 우회 및 Memmap 초기화
        if all_embs is None:
            hidden_size = embs_cpu.shape[1]
            print(f" Detected Hidden Size: {hidden_size} (Mapping to Disk...)")
            all_embs = np.memmap(mmap_file, dtype='float32', mode='w+', shape=(num_texts, hidden_size))
        
        # 생성된 memmap에 직접 기록 (RAM 점유 거의 없음)
        all_embs[i:i+embs_cpu.shape[0]] = embs_cpu
        
        # 메모리 찌꺼기 즉각 폐기
        del inputs, embs, embs_cpu
        
        # 주기적 디스크 동기화
        if i % (batch_size * 10) == 0:
            all_embs.flush()
            
    all_embs.flush()
    torch.cuda.empty_cache()
    
    return all_embs, mmap_file

def _create_query_span(text: str, rng: random.Random, min_words=4, max_words=12) -> str:
    words = text.split()
    total_words = len(words)
    if total_words <= min_words: 
        return text
        
    current_max = min(total_words, max_words)
    target_len = rng.randint(min_words, current_max)
    max_start_idx = total_words - target_len
    start_idx = rng.randint(0, max_start_idx)
    
    return " ".join(words[start_idx : start_idx + target_len])

# ==========================================
# 4. Hard Negative Mining Core
# ==========================================
def mine_hard_negatives(corpus, bm25, bm25_ids, dense_model, dense_embs, dense_ids, device, num_neg=4, top_k=10):
    print(" Generating Fixed-Seed ICT Queries...")
    g2p = G2PConverter()
    doc_ids = list(corpus.keys())
    
    queries_raw = []
    queries_g2p = []
    
    for pos_id in tqdm(doc_ids, desc="Slicing"):
        raw_text = corpus[pos_id]['raw']
        seed_val = hash(raw_text)
        rng = random.Random(seed_val)
        
        q_raw = _create_query_span(raw_text, rng)
        q_g2p = g2p(q_raw)
        
        queries_raw.append(q_raw)
        queries_g2p.append(q_g2p)
        
    print(" Encoding ICT Queries...")
    # 쿼리 인코딩 결과도 튜플로 반환받음
    q_embs_mmap, q_mmap_file = get_dense_embeddings(dense_model, queries_g2p, device, batch_size=256, desc="Query_Encoding")
    
    print(" Moving embeddings to GPU for high-speed similarity search...")
    #  [GPU 고속 내적] CPU 연산 데드락 완전 제거
    # Memmap에 저장된 배열을 GPU 텐서로 한 번에 복사
    dense_embs_gpu = torch.tensor(dense_embs, dtype=torch.float32, device=device)
    q_embs_gpu = torch.tensor(q_embs_mmap, dtype=torch.float32, device=device)
    
    print(" Mining Hard Negative IDs (Caching)...")
    hard_negative_cache = {} 
    
    for i, pos_id in enumerate(tqdm(doc_ids, desc="Mining")):
        q_raw = queries_raw[i]
        
        # --- A. BM25 Mining ---
        tokenized_query = q_raw.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = bm25_scores.argsort()[::-1][:top_k+1]
        bm25_neg_ids = [bm25_ids[idx] for idx in top_bm25_indices if bm25_ids[idx] != pos_id][:top_k]
        
        # --- B. Dense Mining ---
        q_emb = q_embs_gpu[i] 
        dense_scores = torch.matmul(q_emb, dense_embs_gpu.T) 
        top_dense_indices = torch.topk(dense_scores, k=top_k+1).indices.tolist()
        dense_neg_ids = [dense_ids[idx] for idx in top_dense_indices if dense_ids[idx] != pos_id][:top_k]
        
        # --- C. Hybrid Mix ---
        final_neg_ids = []
        for d_id in dense_neg_ids:
            if len(final_neg_ids) < (num_neg // 2) and d_id not in final_neg_ids:
                final_neg_ids.append(d_id)
                
        for b_id in bm25_neg_ids:
            if len(final_neg_ids) < num_neg and b_id not in final_neg_ids:
                final_neg_ids.append(b_id)
                
        hard_negative_cache[pos_id] = final_neg_ids
        
    # 메모리 정리
    del dense_embs_gpu, q_embs_gpu, dense_scores
    torch.cuda.empty_cache()
    
    # 임시 생성된 쿼리 memmap 파일 삭제
    if os.path.exists(q_mmap_file):
        os.remove(q_mmap_file)
        
    return hard_negative_cache

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    cfg = load_config("conf/mine_config.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==========================================
    #  [Stage 1] Train Cache Mining
    # ==========================================
    print("\n" + "="*40 + "\n [Stage 1] Train Data Mining\n" + "="*40)
    train_corpus = load_corpus(cfg['data']['train_tsv'], cfg['data']['train_jsonl'])
    train_bm25, train_bm25_ids = build_bm25_index(train_corpus)
    
    dense_model = load_dense_model(cfg['model'], cfg['model']['model_checkpoint'], device)
    
    # 튜플로 반환되는 (memmap_객체, 임시파일경로)를 받음
    train_dense_embs, train_mmap_file = get_dense_embeddings(dense_model, [train_corpus[doc_id]['g2p'] for doc_id in train_corpus], device, desc="Train_Encoding")
    train_dense_ids = list(train_corpus.keys())
    
    train_cache = mine_hard_negatives(
        train_corpus, train_bm25, train_bm25_ids, 
        dense_model, train_dense_embs, train_dense_ids, device,
        num_neg=cfg['mining']['train_num_negatives'], top_k=cfg['mining']['top_k']
    )
    with open("data/train_dataset/hard_negative_cache_train.json", "w", encoding="utf-8") as f:
        json.dump(train_cache, f, ensure_ascii=False, indent=2)
        
    # 다 쓴 임시 Train 코퍼스 임베딩 파일 삭제
    if os.path.exists(train_mmap_file):
        os.remove(train_mmap_file)

    # ==========================================
    #  [Stage 2] Validation Cache Mining
    # ==========================================
    print("\n" + "="*40 + "\n [Stage 2] Validation Data Mining\n" + "="*40)
    val_corpus = load_corpus(cfg['data']['val_tsv'], cfg['data']['val_jsonl'])
    val_bm25, val_bm25_ids = build_bm25_index(val_corpus)
    
    val_dense_embs, val_mmap_file = get_dense_embeddings(dense_model, [val_corpus[doc_id]['g2p'] for doc_id in val_corpus], device, desc="Val_Encoding")
    val_dense_ids = list(val_corpus.keys())
    
    val_cache = mine_hard_negatives(
        val_corpus, val_bm25, val_bm25_ids, 
        dense_model, val_dense_embs, val_dense_ids, device,
        num_neg=cfg['mining']['val_num_negatives'], top_k=cfg['mining']['top_k']
    )
    with open("data/train_dataset/hard_negative_cache_val.json", "w", encoding="utf-8") as f:
        json.dump(val_cache, f, ensure_ascii=False, indent=2)
        
    # 다 쓴 임시 Val 코퍼스 임베딩 파일 삭제
    if os.path.exists(val_mmap_file):
        os.remove(val_mmap_file)