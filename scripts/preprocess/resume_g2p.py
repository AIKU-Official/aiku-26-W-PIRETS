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
    """워커 프로세스 초기화: 모델 로딩 + NLTK 자동 설치"""
    global worker_converter
    
    # 1. NLTK 리소스 없으면 자동 설치 (에러 방지)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)

    # 2. 모델 로딩
    try:
        worker_converter = G2PConverter()
    except Exception as e:
        print(f"Worker init failed: {e}")

def process_batch(batch_data):
    """
    배치 처리 함수
    - 성공한 것만 반환함.
    - 실패하면 빈 값을 반환하지만, 저장 단계에서 필터링될 예정.
    """
    global worker_converter
    results = []
    
    if worker_converter is None: return []

    for sample_id, text in batch_data:
        phoneme = "" 
        try:
            if text and str(text).strip():
                phoneme = worker_converter(str(text))
            
            # 성공 여부와 상관없이 일단 결과 생성
            # (빈 값인지 아닌지는 나중에 판단)
            results.append(json.dumps({'id': sample_id, 'phoneme': phoneme}, ensure_ascii=False))
            
        except Exception:
            # 에러나면 빈 값으로 일단 리턴
            results.append(json.dumps({'id': sample_id, 'phoneme': ""}, ensure_ascii=False))
            
    return results

# =============================================================================
# 2. 메인 로직 (Clean -> Detect -> Fill)
# =============================================================================
def resume_g2p_smart(input_path, output_path, num_workers=8, batch_size=256):
    print(f"Starting Smart Resume Process...")
    
    # ---------------------------------------------------------
    # Phase 1: 파일 청소 (빈 값 제거 및 유효 데이터 확보)
    # ---------------------------------------------------------
    processed_ids = set()
    valid_lines = []
    
    if os.path.exists(output_path):
        print(f"Phase 1: Cleaning existing file ({output_path})...")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Scanning & Cleaning", unit="lines"):
                try:
                    line = line.strip()
                    if not line: continue
                    
                    data = json.loads(line)
                    curr_id = str(data['id'])
                    curr_phoneme = str(data.get('phoneme', '')).strip()
                    
                    # 빈 값이 아닌 진짜 데이터만 메모리에 남김
                    if curr_phoneme != "":
                        # 중복 제거: 나중에 나온 게 덮어쓰도록 (최신 데이터 유지)
                        if curr_id in processed_ids:
                            # 이미 있는 ID면 리스트에서 기존 거 제거하고 새로 추가 (좀 비효율적이지만 확실하게)
                            # 하지만 속도를 위해 그냥 Set으로 관리하고 나중에 덮어쓰는 방식 사용
                            pass 
                        
                        processed_ids.add(curr_id)
                        valid_lines.append(line) # 유효한 라인만 저장
                except:
                    pass
        
        # 청소된 내용으로 파일 덮어쓰기 (Rewrite)
        print(f"Rewriting cleaned file... (Saved {len(valid_lines):,} valid items)")
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in valid_lines:
                f.write(line + "\n")
    else:
        print("No existing file found. Starting from scratch.")

    print(f"Cleaned! Currently valid items: {len(processed_ids):,}")

    # ---------------------------------------------------------
    # Phase 2: 해야 할 작업 색출 (Targeting)
    # ---------------------------------------------------------
    print(f"Phase 2: Loading input data to find gaps...")
    df = pd.read_csv(input_path, sep='\t', on_bad_lines='skip', dtype=str).dropna(subset=['id', 'text'])
    total_rows = len(df)
    
    # [전체] - [이미 처리된 유효 ID] = [해야 할 것]
    df_remaining = df[~df['id'].isin(processed_ids)]
    remaining_count = len(df_remaining)
    
    if remaining_count == 0:
        print("All 100% Complete! Nothing to do.")
        return

    print(f"Targets Acquired: {remaining_count:,} items to process.")
    print(f"   (These are gaps + previously empty items)")

    # ---------------------------------------------------------
    # Phase 3: 빈 구멍 채우기 (Filling)
    # ---------------------------------------------------------
    data_list = list(zip(df_remaining['id'], df_remaining['text']))
    chunks = [data_list[i : i + batch_size] for i in range(0, remaining_count, batch_size)]
    
    print(f"Phase 3: Starting Workers (Cores: {num_workers})...")
    
    # Append 모드로 열어서 뒤에 붙임
    with open(output_path, 'a', encoding='utf-8') as f_out:
        with Pool(num_workers, initializer=init_worker, maxtasksperchild=10) as pool:
            with tqdm(total=remaining_count, desc="Filling Gaps", unit="sent") as pbar:
                
                for batch_result in pool.imap_unordered(process_batch, chunks):
                    for line in batch_result:
                        # JSON 파싱해서 빈 값이면 저장 안 함 (파일 더럽히기 방지)
                        try:
                            item = json.loads(line)
                            if item['phoneme'].strip() != "":
                                f_out.write(line + "\n")
                        except:
                            pass
                    
                    f_out.flush() # 실시간 저장
                    pbar.update(len(batch_result))

    print(f"\nAll Done! Final Check: {len(processed_ids) + remaining_count:,} / {total_rows:,}")

if __name__ == "__main__":
    INPUT_FILE = os.path.join(project_root, "data", "train_dataset", "train.tsv")
    OUTPUT_FILE = os.path.join(project_root, "data", "train_dataset", "train_g2p.jsonl")
    
    # 8코어로 실행
    resume_g2p_smart(INPUT_FILE, OUTPUT_FILE, num_workers=8, batch_size=256)