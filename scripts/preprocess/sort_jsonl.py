import os
import sys
import json
from tqdm import tqdm

# =============================================================================
# 설정
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

def sort_jsonl_file(file_path):
    print(f"Reading: {os.path.basename(file_path)}...")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = []
    # 1. 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                if line.strip():
                    data.append(json.loads(line))
            except:
                pass
    
    print(f"   -> Loaded {len(data):,} items. Sorting...")

    # 2. 정렬 (ID 기준)
    # 문자열 정렬이지만 'train_000001' 포맷이라 숫자 순서대로 정렬됨
    data.sort(key=lambda x: x['id'])
    
    # 3. 다시 쓰기
    print(f"Saving sorted file...")
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Done! First ID: {data[0]['id']} / Last ID: {data[-1]['id']}\n")

if __name__ == "__main__":
    # Train과 Val 파일 경로
    train_file = os.path.join(project_root, "data", "train_dataset", "train_g2p.jsonl")
    val_file = os.path.join(project_root, "data", "train_dataset", "val_g2p.jsonl")
    
    print("Starting Sort Process...\n")
    
    # 두 파일 모두 정렬 수행
    sort_jsonl_file(train_file)
    sort_jsonl_file(val_file)
    
    print("All files are sorted by ID!")