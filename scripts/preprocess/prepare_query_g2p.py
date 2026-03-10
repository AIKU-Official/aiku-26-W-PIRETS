import os
import json
import random
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import multiprocessing as mp

from src.g2p import G2PConverter

# Global variable for multiprocessing (avoids pickling issues)
global_g2p = None

def worker_init():
    """각 CPU 코어(Worker)가 각자의 G2P 모델을 메모리에 로드하도록 초기화합니다."""
    global global_g2p
    global_g2p = G2PConverter()
    global_g2p("init") 

def _create_query_span(text: str, rng: random.Random, min_words=4, max_words=12) -> str:
    words = text.split()
    total_words = len(words)
    if total_words <= min_words: return text
    current_max = min(total_words, max_words)
    target_len = rng.randint(min_words, current_max)
    max_start_idx = total_words - target_len
    start_idx = rng.randint(0, max_start_idx)
    return " ".join(words[start_idx : start_idx + target_len])

def process_line(line):
    """각 CPU 코어가 병렬로 실행할 단일 라인 처리 함수"""
    parts = line.strip().split('\t')
    if len(parts) < 2: return None
    
    doc_id = parts[0].strip()
    if doc_id.lower() == 'id': return None
    
    raw_text = parts[1].strip()
    
    # [Deterministic] 고정 난수 생성
    seed_val = hash(raw_text)
    fixed_rng = random.Random(seed_val)
    
    # Query Span 생성 및 글로벌 G2P 객체로 연산
    query_span = _create_query_span(raw_text, fixed_rng)
    query_phoneme = global_g2p(query_span) 
    
    json_record = {"id": doc_id, "query_g2p": query_phoneme}
    return json.dumps(json_record, ensure_ascii=False)

def build_offline_query_g2p_parallel(tsv_path, output_jsonl_path, num_cores):
    if not os.path.exists(tsv_path):
        print(f"Error: TSV File not found - {tsv_path}")
        return

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    
    # 파일을 메모리에 올려 병렬 처리를 준비합니다.
    with open(tsv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"Start Multiprocessing with {num_cores} CPU cores...")
    
    # Start parallel processing pool
    with mp.Pool(processes=num_cores, initializer=worker_init) as pool:
        # imap을 사용하여 tqdm 프로그레스 바와 함께 병렬 연산 결과를 수집합니다.
        results = list(tqdm(pool.imap(process_line, lines, chunksize=100), total=len(lines), desc=f"Processing {os.path.basename(tsv_path)}"))
        
    # 결과를 파일로 저장
    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        for res in results:
            if res is not None:
                f_out.write(res + '\n')

@hydra.main(version_base=None, config_path="conf", config_name="train_reranker_config")
def main(config: DictConfig):
    # 동적 CPU 코어 감지
    num_cores = max(1, mp.cpu_count() - 2)
    
    train_tsv = config.data.train_tsv
    train_out = config.data.train_query_jsonl
    val_tsv = config.data.val_tsv
    val_out = config.data.val_query_jsonl

    print(f"\n[1/2] Building Train Query G2P -> {train_out}")
    build_offline_query_g2p_parallel(train_tsv, train_out, num_cores)
    
    print(f"\n[2/2] Building Val Query G2P -> {val_out}")
    build_offline_query_g2p_parallel(val_tsv, val_out, num_cores)
    
    print("\nAll Query G2P Offline Caching Completed Successfully!")

if __name__ == "__main__":
    main()