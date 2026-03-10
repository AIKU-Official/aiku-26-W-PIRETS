import os
import sys
import json
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import nltk

# =============================================================================
# 1. 설정 및 초기화
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.g2p import G2PConverter

# 전역 변수
worker_converter = None

def init_worker():
    """
    워커 프로세스 초기화
    - 여기서는 다운로드 하지 않음 (충돌 방지)
    - 오직 모델 로딩만 수행
    """
    global worker_converter
    try:
        worker_converter = G2PConverter()
    except Exception as e:
        print(f" Worker init failed: {e}")

def process_batch(batch_data):
    """배치 처리 함수"""
    global worker_converter
    results = []
    
    if worker_converter is None: return []

    for sample_id, text in batch_data:
        phoneme = "" 
        try:
            if text and str(text).strip():
                phoneme = worker_converter(str(text))
            
            # 결과 담기 (성공하든 실패하든)
            results.append(json.dumps({'id': sample_id, 'phoneme': phoneme}, ensure_ascii=False))
            
        except Exception:
            # 에러 발생 시 빈 값으로 담음
            results.append(json.dumps({'id': sample_id, 'phoneme': ""}, ensure_ascii=False))
            
    return results

# =============================================================================
# 2. 메인 로직
# =============================================================================
def preprocess_val(input_path, output_path, num_workers=8, batch_size=256):
    print(f" Starting Preprocessing for VAL dataset...")
    
    #  메인 프로세스가 시작 전에 NLTK를 미리 받아놓음 (충돌 방지)
    print(" Checking NLTK resources...")
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        print("   -> Downloading missing NLTK data (Once for all)...")
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    print(" NLTK Ready.")
    
    # 1. 기존 파일 청소 및 로드 (Resume 기능)
    processed_ids = set()
    valid_lines = []
    
    if os.path.exists(output_path):
        print(f" Scanning existing file ({output_path})...")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Scanning", unit="lines"):
                try:
                    line = line.strip()
                    if not line: continue
                    data = json.loads(line)
                    
                    # 빈 값이 아닌 유효한 데이터만 유지
                    if data.get('phoneme', '').strip() != "":
                        processed_ids.add(str(data['id']))
                        valid_lines.append(line)
                except:
                    pass
        
        # 파일 다시 쓰기 (Clean version)
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in valid_lines:
                f.write(line + "\n")
    
    print(f" Already processed: {len(processed_ids):,} items")

    # 2. 원본 데이터 로드
    print(f" Loading input: {input_path}")
    if not os.path.exists(input_path):
        print(f" Error: Input file not found at {input_path}")
        return

    df = pd.read_csv(input_path, sep='\t', on_bad_lines='skip', dtype=str).dropna(subset=['id', 'text'])
    total_rows = len(df)
    
    # 3. 해야 할 작업 필터링
    df_remaining = df[~df['id'].isin(processed_ids)]
    remaining_count = len(df_remaining)
    
    if remaining_count == 0:
        print(" Val dataset is 100% ready! No action needed.")
        return

    print(f" Processing remaining {remaining_count:,} items...")

    # 4. 실행
    data_list = list(zip(df_remaining['id'], df_remaining['text']))
    chunks = [data_list[i : i + batch_size] for i in range(0, remaining_count, batch_size)]
    
    # Append 모드로 실행
    with open(output_path, 'a', encoding='utf-8') as f_out:
        # maxtasksperchild=10: 메모리 누수 방지
        with Pool(num_workers, initializer=init_worker, maxtasksperchild=10) as pool:
            with tqdm(total=remaining_count, desc="Converting Val", unit="sent") as pbar:
                for batch_result in pool.imap_unordered(process_batch, chunks):
                    for line in batch_result:
                        try:
                            item = json.loads(line)
                            if item['phoneme'].strip() != "":
                                f_out.write(line + "\n")
                        except:
                            pass
                    f_out.flush()
                    pbar.update(len(batch_result))

    print(f"\n Validation Preprocessing Done!")
    print(f" Total Valid Items: {len(processed_ids) + remaining_count:,} / {total_rows:,}")

if __name__ == "__main__":
    INPUT_FILE = os.path.join(project_root, "data", "train_dataset", "val.tsv")
    OUTPUT_FILE = os.path.join(project_root, "data", "train_dataset", "val_g2p.jsonl")
    
    # 8코어로 안전하게 실행
    preprocess_val(INPUT_FILE, OUTPUT_FILE, num_workers=8, batch_size=256)