import os
import sys
import json
import math
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

# =============================================================================
# 1. 경로 설정
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.g2p import G2PConverter

# =============================================================================
# 2. 전역 변수 및 초기화
# =============================================================================
# 각 프로세스(Worker)가 가질 전역 변수
worker_converter = None

def init_worker():
    """
    프로세스가 시작될 때 딱 한 번만 실행됨.
    무거운 G2P 모델을 여기서 미리 로드해둠.
    """
    global worker_converter
    worker_converter = G2PConverter()

# =============================================================================
# 3. Worker 함수
# =============================================================================
def process_batch(batch_data):
    """
    이제 모델을 새로 만들지 않고, 이미 만들어진 worker_converter를 씀
    """
    global worker_converter
    results = []
    
    for sample_id, text in batch_data:
        try:
            # 전역 변수 사용
            phoneme = worker_converter(str(text))
            results.append(json.dumps({'id': sample_id, 'phoneme': phoneme}, ensure_ascii=False))
        except Exception:
            continue
            
    return results

# =============================================================================
# 4. 메인 로직
# =============================================================================
def preprocess_optimized(input_path, output_path, num_workers=12, batch_size=128):
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    # CPU 코어 수 감지
    print(f"Smart Mode: Using {num_workers} CPU cores with Pre-initialization")
    
    df = pd.read_csv(input_path, sep='\t', on_bad_lines='skip', dtype=str).dropna(subset=['id', 'text'])
    total_rows = len(df)
    print(f"Total Rows: {total_rows:,}")

    # 데이터 쪼개기
    data_list = list(zip(df['id'], df['text']))
    chunks = [data_list[i : i + batch_size] for i in range(0, total_rows, batch_size)]
    
    print(f"Split into {len(chunks):,} batches.")
    print(f"Starting Processing...")

    results = []
    
    # 각 프로세스가 init_worker로 G2P 모델을 미리 로드한 후, process_batch에서 그 모델을 사용하여 변환 작업 수행
    with Pool(num_workers, initializer=init_worker) as pool:
        
        with tqdm(total=total_rows, desc="Converting", unit="sent") as pbar:
            for batch_result in pool.imap_unordered(process_batch, chunks):
                results.extend(batch_result)
                pbar.update(len(batch_result))

    print(f"Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + "\n")

    print(f"\nAll Done! Processed {len(results):,} items.")

if __name__ == "__main__":
    INPUT_FILE = os.path.join(project_root, "data", "train_dataset", "train.tsv")
    OUTPUT_FILE = os.path.join(project_root, "data", "train_dataset", "train_g2p.jsonl")
    
    # 안전빵으로 12개만 먼저 돌려봅시다. (잘 되면 24개로 올려도 됨)
    preprocess_optimized(INPUT_FILE, OUTPUT_FILE, num_workers=12, batch_size=128)