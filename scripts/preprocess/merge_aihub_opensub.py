import os
import sys
import random
from tqdm import tqdm

# =============================================================================
# 1. 경로 설정
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

def merge_and_shuffle(file_list, output_file, mode="train"):
    # mode: 'train' 또는 'val' (소문자로 받아서 ID 접두사로 사용)
    print(f"\n[Merging {mode.upper()}] Processing {len(file_list)} files...")
    
    combined_lines = []
    
    # [Step 1] 파일 읽기
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        print(f"   Reading: {os.path.basename(file_path)}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # 헤더가 있는 경우 건너뛰기 (첫 줄이 id text 면 skip)
            if lines and lines[0].startswith("id\ttext"):
                data_lines = lines[1:]
            else:
                data_lines = lines
                
            combined_lines.extend(data_lines)
            
    print(f"   Total {mode.upper()} samples: {len(combined_lines):,}")
    
    # [Step 2] 섞기 (Shuffle)
    print(f"   Shuffling...")
    random.shuffle(combined_lines)
    
    # [Step 3] ID 재발급 및 저장 (Re-indexing)
    print(f"   Re-indexing and Saving...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out:
        # 최종 헤더 작성
        out.write("id\ttext\n")
        
        for idx, line in enumerate(tqdm(combined_lines, desc=f"Writing {mode.upper()}")):
            parts = line.strip().split('\t')
            
            # 텍스트 내용만 추출 (혹시 탭이 여러 개일 경우 대비, 뒤쪽 텍스트 전체 가져오기)
            if len(parts) >= 2:
                # 기존 ID는 버리고 텍스트만 취함
                text_content = parts[1] 
            else:
                # 형식이 이상하면 그냥 통째로 씀 (예외처리)
                text_content = line.strip()

            # Assign new sequential ID (e.g., train_000000)
            new_id = f"{mode}_{idx:06d}"
            
            out.write(f"{new_id}\t{text_content}\n")
            
    print(f"Saved to: {output_file}")

# =============================================================================
# 2. 메인 실행
# =============================================================================
if __name__ == "__main__":
    # ---------------------------------------------------------
    # Input file paths
    # ---------------------------------------------------------
    TRAIN_FILES = [
        os.path.join(root_dir, "data", "train_dataset", "aihub", "train", "aihub_corpus_train.tsv"),
        os.path.join(root_dir, "data", "train_dataset", "opensubtitles", "train", "opensub_train.tsv")
    ]
    
    VAL_FILES = [
        os.path.join(root_dir, "data", "train_dataset", "aihub", "val", "aihub_corpus_val.tsv"),
        os.path.join(root_dir, "data", "train_dataset", "opensubtitles", "val", "opensub_val.tsv")
    ]
    
    # ---------------------------------------------------------
    # Output paths
    # ---------------------------------------------------------
    FINAL_TRAIN_OUTPUT = os.path.join(root_dir, "data", "train_dataset", "train.tsv")
    FINAL_VAL_OUTPUT = os.path.join(root_dir, "data", "train_dataset", "val.tsv")
    
    # ---------------------------------------------------------
    # Execute
    # ---------------------------------------------------------
    # ID 접두사를 'train'으로 설정
    merge_and_shuffle(TRAIN_FILES, FINAL_TRAIN_OUTPUT, mode="train")
    
    # ID 접두사를 'val'로 설정
    merge_and_shuffle(VAL_FILES, FINAL_VAL_OUTPUT, mode="val")
    
    print("\nAll Done! Datasets are Clean & Ready.")