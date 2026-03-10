import os
import csv
import multiprocessing as mp
from tqdm import tqdm
from src.g2p import G2PConverter

# 워커(Worker) 프로세스들이 공유할 전역 변수
global_g2p = None

def init_worker():
    """각 CPU 코어가 깨어날 때마다 자신만의 G2P 인스턴스를 하나씩 쥐어줍니다."""
    global global_g2p
    global_g2p = G2PConverter()

def process_row(row):
    """개별 코어가 실행할 순수 연산 함수"""
    text = row.get('text', '').strip()
    # 전역 G2P 객체를 사용하여 변환 후 딕셔너리에 추가
    row['phoneme'] = global_g2p(text)
    return row

def run_parallel_g2p(input_path, output_path, num_cores=None):
    print(f"Loading original data from: {input_path}")
    
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = list(reader.fieldnames)
        if 'phoneme' not in fieldnames:
            fieldnames.append('phoneme')
        for row in reader:
            data.append(row)
            
    if num_cores is None:
        num_cores = max(1, mp.cpu_count() - 1) # OS를 위해 코어 1개는 남겨둠
        
    print(f"Starting Parallel G2P Conversion using {num_cores} Cores...")
    
    # 프로세스 풀(Pool) 생성 및 매핑
    processed_data = []
    with mp.Pool(processes=num_cores, initializer=init_worker) as pool:
        # pool.imap을 사용하여 진행률(tqdm)을 실시간으로 추적합니다.
        for result in tqdm(pool.imap(process_row, data), total=len(data), desc="Processing"):
            processed_data.append(result)
            
    print(f"Saving processed data to: {output_path}")
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(processed_data)
        
    print("Parallel Pre-processing Complete!")

if __name__ == "__main__":
    base_dir = "data/eval_dataset" # 경로를 알맞게 수정하세요
    
    # 1. Base 코퍼스 변환
    base_input = os.path.join(base_dir, "corpus.tsv")
    base_output = os.path.join(base_dir, "corpus_g2p.tsv")
    print("\n--- [Task 1] Base Corpus ---")
    run_parallel_g2p(base_input, base_output)
    
    # 2. Distractor 코퍼스 변환
    dist_input = os.path.join(base_dir, "distractor_corpus.tsv")
    dist_output = os.path.join(base_dir, "distractor_corpus_g2p.tsv")
    print("\n--- [Task 2] Distractor Corpus ---")
    run_parallel_g2p(dist_input, dist_output)