import os
import sys
import glob
import re
import gc
from tqdm import tqdm

# =============================================================================
# 1. 경로 설정 & 모듈 Import
# =============================================================================
# 현재 파일(scripts/preprocess_aihub.py)의 부모(scripts)의 부모(root)를 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 1) 공통 전처리 함수 (src/utils.py)
try:
    from src.utils import normalize_text
    print(f"Imported 'normalize_text' from src.utils")
except ImportError:
    print("Error: 'src.utils' not found.")
    sys.exit(1)

# 2) 슬라이딩 윈도우 함수 (gen_corpus.py)
try:
    # gen_corpus.py가 루트에 있다고 가정
    from gen_corpus import create_sliding_windows
    print(f"Imported 'create_sliding_windows' from gen_corpus.py")
except ImportError:
    print("Error: 'gen_corpus.py' not found.")
    sys.exit(1)

# =============================================================================
# 2. 핵심 설정 (Configuration)
# =============================================================================
WINDOW_SIZE = 64
STRIDE = 64  # 겹침 없이

# =============================================================================
# 3. 전처리 로직
# =============================================================================
def remove_speaker_tag(text):
    """AI Hub 데이터 전용: 문장 앞 화자 태그(1 :, 2 :) 제거"""
    return re.sub(r'^\d+\s*:\s*', '', text)

def process_dataset(input_dir, output_file, id_prefix="pid"):
    files = glob.glob(os.path.join(input_dir, "**/*.txt"), recursive=True)
    
    if not files:
        print(f"Warning: No txt files found in '{input_dir}'.")
        return

    print(f"\nProcessing {len(files)} files from: {input_dir}")
    print(f"Mode: {id_prefix.upper()} | Config: Window={WINDOW_SIZE}, Stride={STRIDE}")
    
    all_sentences = []
    
    # [Step 1] 모든 파일 로드 및 정제
    for file_path in tqdm(files, desc=f"Loading ({id_prefix})"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = remove_speaker_tag(line)
                line = normalize_text(line)
                
                if line:
                    all_sentences.append(line)
                    
        except Exception as e:
            print(f"Skip {file_path}: {e}")
            continue
            
    # [Step 2] 거대 코퍼스 생성 (이어붙이기)
    print("Merging text streams...")
    giant_corpus = " ".join(all_sentences)
    
    # 메모리 최적화
    del all_sentences
    gc.collect() 
    
    total_words = len(giant_corpus.split())
    print(f"Total Tokens (Words): {total_words:,}")
    
    # [Step 3] 슬라이딩 윈도우 및 저장
    print(f"Slicing and Saving to {output_file}...")
    
    count_passages = 0
    
    # 폴더가 없으면 생성
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out:
        # 헤더 작성
        out.write("id\ttext\n") 
        
        for chunk_text, _, _ in create_sliding_windows(giant_corpus, WINDOW_SIZE, STRIDE):
            
            # 너무 짧은 짜투리(절반 이하)는 버림
            if len(chunk_text.split()) < WINDOW_SIZE // 2:
                continue

            # ID 부여 (예: train_pid_0, val_pid_0)
            passage_id = f"{id_prefix}_{count_passages}"
            out.write(f"{passage_id}\t{chunk_text}\n")
            
            count_passages += 1
            
    # 메모리 최적화
    del giant_corpus
    gc.collect()

    print(f"Done! ({id_prefix}) -> {count_passages:,} passages created.")

# =============================================================================
# 4. 메인 실행 (Train -> Val 순서)
# =============================================================================
if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # 1. Train dataset generation
    # ---------------------------------------------------------
    print("[1/2] Starting Train Dataset Generation...")
    TRAIN_INPUT = os.path.join(root_dir, "data", "train_dataset", "aihub", "train_raw")
    TRAIN_OUTPUT = os.path.join(root_dir, "data", "train_dataset", "aihub", "train", "aihub_corpus_train.tsv")
    
    process_dataset(TRAIN_INPUT, TRAIN_OUTPUT, id_prefix="train_pid")

    # ---------------------------------------------------------
    # 2. Val dataset generation
    # ---------------------------------------------------------
    print("\n[2/2] Starting Validation Dataset Generation...")
    VAL_INPUT = os.path.join(root_dir, "data", "train_dataset", "aihub", "val_raw")
    VAL_OUTPUT = os.path.join(root_dir, "data", "train_dataset", "aihub", "val", "aihub_corpus_val.tsv")
    
    process_dataset(VAL_INPUT, VAL_OUTPUT, id_prefix="val_pid")
    
    print("\nAll jobs finished successfully!")