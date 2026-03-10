import os
import sys
import glob
import re
import gc
import random
from tqdm import tqdm

# =============================================================================
# 1. 경로 설정 & 모듈 Import
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

try:
    from src.utils import normalize_text
    print(f"Imported 'normalize_text' from src.utils")
except ImportError:
    print("Error: 'src.utils' not found.")
    sys.exit(1)

try:
    from gen_corpus import create_sliding_windows
    print(f"Imported 'create_sliding_windows' from gen_corpus.py")
except ImportError:
    print(" Error: 'gen_corpus.py'를 찾을 수 없습니다.")
    sys.exit(1)

# =============================================================================
# 2. 핵심 설정 (Configuration)
# =============================================================================
WINDOW_SIZE = 64
STRIDE = 64            
TARGET_TRAIN_COUNT = 120000 
TARGET_VAL_COUNT = 22500    
TOTAL_TARGET = TARGET_TRAIN_COUNT + TARGET_VAL_COUNT

# [최적화] 필요한 분량보다 넉넉하게 약 20배 정도의 문장만 읽고 멈춤
# 14만 개 Passage * 64단어 ≈ 900만 단어 필요함
# 문장당 평균 10단어 가정 시 90만 문장 필요. 넉넉히 300만 문장만 읽으면 충분함
MAX_LOADING_SENTENCES = 3000000 

# =============================================================================
# 3. 전처리 로직
# =============================================================================
def has_korean(text):
    return bool(re.search(r'[가-힣]', text))

def process_opensub_english(input_dir, train_output_path, val_output_path):
    # [Step 1] 파일 찾기
    all_files = glob.glob(os.path.join(input_dir, "**/*"), recursive=True)
    target_files = [f for f in all_files if f.endswith('.txt') or f.endswith('.en')]
    
    #  파일 순서를 섞어서 랜덤하게 추출 (앞부분 파일만 읽는 편향 방지)
    random.shuffle(target_files)
    
    print(f"\n Found {len(target_files)} English files.")
    print(f" Target Passages: {TOTAL_TARGET:,}")
    print(f" optimization: Will stop loading after {MAX_LOADING_SENTENCES:,} lines.")
    
    all_sentences = []
    current_lines_count = 0
    
    # [Step 2] 로드 및 정제 (Smart Loading)
    for file_path in tqdm(target_files, desc="Loading Data (Smart Mode)"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = normalize_text(line)
                
                if not line or has_korean(line):
                    continue
                    
                all_sentences.append(line)
                current_lines_count += 1
            
            #  일정량 이상 읽었으면 파일 로딩 중단!
            if current_lines_count >= MAX_LOADING_SENTENCES:
                print(f" Loaded enough data ({current_lines_count:,} lines). Stopping load.")
                break
                
        except Exception as e:
            continue
    
    # [Step 3] 병합
    print(" Merging text streams...")
    giant_corpus = " ".join(all_sentences)
    
    del all_sentences
    gc.collect()
    
    total_words = len(giant_corpus.split())
    print(f" Total Tokens: {total_words:,}")
    
    if total_words < (TOTAL_TARGET * WINDOW_SIZE):
        print(" Warning: Loaded data might be smaller than target. (But let's try)")

    # [Step 4] 이어달리기 저장 (Train -> Val)
    print(f" Slicing and splitting into Train/Val...")
    
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_output_path), exist_ok=True)

    with open(train_output_path, 'w', encoding='utf-8') as f_train, \
         open(val_output_path, 'w', encoding='utf-8') as f_val:
        
        f_train.write("id\ttext\n")
        f_val.write("id\ttext\n")
        
        train_count = 0
        val_count = 0
        
        for chunk_text, _, _ in create_sliding_windows(giant_corpus, WINDOW_SIZE, STRIDE):
            
            if len(chunk_text.split()) < WINDOW_SIZE // 2:
                continue
            
            # 1: Train 채우기
            if train_count < TARGET_TRAIN_COUNT:
                pid = f"open_en_train_{train_count}"
                f_train.write(f"{pid}\t{chunk_text}\n")
                train_count += 1
                
            # 2: Val 채우기
            elif val_count < TARGET_VAL_COUNT:
                pid = f"open_en_val_{val_count}"
                f_val.write(f"{pid}\t{chunk_text}\n")
                val_count += 1
                
            # 3: 완료 시 종료
            else:
                print(f" Reached both targets! Stopping.")
                break
    
    del giant_corpus
    gc.collect()

    print(f"\n 완료!")
    print(f"   - Train Saved: {train_count:,} passages")
    print(f"   - Val   Saved: {val_count:,} passages")

# =============================================================================
# 4. 메인 실행
# =============================================================================
if __name__ == "__main__":
    INPUT_DIR = os.path.join(root_dir, "data", "train_dataset", "opensubtitles")
    
    TRAIN_OUTPUT = os.path.join(root_dir, "data", "train_dataset", "opensubtitles", "train", "opensub_train.tsv")
    VAL_OUTPUT = os.path.join(root_dir, "data", "train_dataset", "opensubtitles", "val", "opensub_val.tsv")
    
    process_opensub_english(INPUT_DIR, TRAIN_OUTPUT, VAL_OUTPUT)