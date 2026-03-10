import os
import csv
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

# 설정 상수: 합칠 팀원들의 ID 리스트
USER_IDS = [0, 1, 2, 3]
TARGET_FILES = ['songs', 'queries', 'corpus', 'qrels']

# 색상 코드 (로그용)
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f" Created directory: {directory}")

def merge_files(file_type, base_dir):
    """
    file_type(예: songs)에 해당하는 각 팀원의 파일(songs0.tsv ~ songs3.tsv)을 읽어
    통합 파일(songs.tsv)로 병합합니다.
    """
    output_path = os.path.join(base_dir, f'{file_type}.tsv')
    print(f"\nProcessing [{file_type}] ...")
    
    total_lines = 0
    seen_ids = set()
    header_saved = False
    
    # 중복 체크할 ID 컬럼 인덱스 (0번째 컬럼: song_id or query_id or passage_id)
    id_col_idx = 0 
    
    with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        
        for uid in USER_IDS:
            # 읽을 파일 경로: data/eval_dataset/songs0.tsv
            input_filename = os.path.join(base_dir, f'{file_type}{uid}.tsv')
            
            if not os.path.exists(input_filename):
                print(f"  - {YELLOW}User {uid} file missing ({os.path.basename(input_filename)}) -> Skipped{RESET}")
                continue
            
            with open(input_filename, 'r', encoding='utf-8') as infile:
                reader = csv.reader(infile, delimiter='\t')
                
                # 헤더 처리
                header = next(reader, None)
                if not header: continue
                
                # 첫 번째 파일의 헤더만 통합 파일에 기록
                if not header_saved:
                    writer.writerow(header)
                    header_saved = True
                
                # 내용 병합
                user_line_cnt = 0
                for row in reader:
                    if not row: continue
                    
                    # 중복 ID 체크 (songs, queries 파일의 경우)
                    if file_type in ['songs', 'queries']:
                        obj_id = row[id_col_idx]
                        if obj_id in seen_ids:
                            # 이미 있는 ID는 경고만 하고 일단 저장 (혹은 skip 가능)
                            # seen_ids.add(obj_id) # 여기서는 중복 카운팅만 하고 넘어감
                            pass
                        seen_ids.add(obj_id)
                    
                    writer.writerow(row)
                    user_line_cnt += 1
                    total_lines += 1
                    
                print(f"  - User {uid}: {user_line_cnt} lines merged.")

    print(f"{GREEN} Saved to {output_path} (Total {total_lines} lines){RESET}")

@hydra.main(version_base=None, config_path="conf", config_name="eval_config")
def main(cfg: DictConfig):
    # 1. Config에서 base_dir 가져오기 (evaluate.yaml 참조)
    if 'base_dir' not in cfg.data:
        print(f"{RED}[Error] Config에 'base_dir' 설정이 없습니다.{RESET}")
        return

    # Hydra가 절대 경로로 변환해줌 (중요)
    base_dir = to_absolute_path(cfg.data.base_dir)
    
    print(f" Starting Dataset Merge")
    print(f" Target Directory: {base_dir}")
    
    ensure_dir(base_dir)
    
    # 2. 파일별 병합 수행
    for fname in TARGET_FILES:
        merge_files(fname, base_dir)
        
    print("\n" + "="*50)
    print(f"{GREEN} All datasets merged successfully!{RESET}")
    print(f"   (Output: songs.tsv, queries.tsv, corpus.tsv, qrels.tsv)")
    print("="*50)

if __name__ == "__main__":
    main()