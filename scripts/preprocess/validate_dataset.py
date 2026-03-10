# 실행 예시:
# 1. 통합 파일 검사: python validate_dataset.py
# 2. 개별 파일 검사: python validate_dataset.py user_id=0

import os
import csv
import re
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from collections import defaultdict

# 색상 코드
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def validate_id_format(id_str, type='song'):
    if type == 'song':
        return bool(re.match(r'^S\d{4}$', id_str))
    elif type == 'query':
        return bool(re.match(r'^S\d{4}_Q\d{4}$', id_str))
    return False

def check_file_exists(filepath):
    if not os.path.exists(filepath):
        print(f"{RED}[FAIL] 파일이 없습니다: {filepath}{RESET}")
        return False
    return True

@hydra.main(version_base=None, config_path="conf", config_name="eval_config")
def main(cfg: DictConfig):
    # 1. 검사 모드 결정 (user_id 유무에 따라 Suffix 설정)
    # config.yaml에 user_id: null이 있어도 안전하게 처리
    if cfg.user_id is None or cfg.user_id == '???':
        suffix = ""
        mode_name = "Merged Dataset Check (songs.tsv)"
    else:
        suffix = str(cfg.user_id)
        mode_name = f"User {suffix} Check (songs{suffix}.tsv)"

    print(f"Starting Validation: {YELLOW}[{mode_name}]{RESET}")

    # 2. 경로 설정 (Evaluate.yaml 참조)
    base_dir = to_absolute_path(cfg.data.base_dir)
    lyrics_dir = to_absolute_path(cfg.data.lyrics_dir)
    
    # 파일명 결정 (suffix 적용)
    songs_file = os.path.join(base_dir, f'songs{suffix}.tsv')
    queries_file = os.path.join(base_dir, f'queries{suffix}.tsv')
    
    if not check_file_exists(songs_file) or not check_file_exists(queries_file):
        return

    error_cnt = 0
    warning_cnt = 0
    
    # 3. Songs 로드 및 검증
    print(f"\nChecking {songs_file}...")
    valid_song_ids = set()
    
    with open(songs_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader, 1):
            sid = row.get('song_id', '').strip()
            if not sid: continue
            
            if not validate_id_format(sid, 'song'):
                print(f"{RED}  - Line {i}: 잘못된 Song ID ({sid}){RESET}")
                error_cnt += 1
                continue
            
            # 가사 파일 존재 여부 확인
            lyric_path = os.path.join(lyrics_dir, f"{sid}.txt")
            if not os.path.exists(lyric_path):
                print(f"{RED}  - Line {i}: 가사 파일 누락 ({lyric_path}){RESET}")
                error_cnt += 1
            
            valid_song_ids.add(sid)
            
    print(f"{GREEN}Songs 검증 완료 (총 {len(valid_song_ids)}곡){RESET}")

    # 4. Queries 검증
    print(f"\nChecking {queries_file}...")
    
    song_difficulty_map = defaultdict(list)
    seen_query_ids = set()
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for i, row in enumerate(reader, 1):
            qid = row.get('query_id', '').strip()
            sid = row.get('song_id', '').strip()
            gold = row.get('gold_span_text', '').strip()
            diff = row.get('difficulty', '').strip()
            
            if not qid: continue

            # 포맷 및 무결성 검사
            if not validate_id_format(qid, 'query'):
                print(f"{RED}  - Line {i}: 잘못된 Query ID ({qid}){RESET}")
                error_cnt += 1
            if qid in seen_query_ids:
                print(f"{RED}  - Line {i}: 중복된 Query ID ({qid}){RESET}")
                error_cnt += 1
            seen_query_ids.add(qid)

            if sid not in valid_song_ids:
                print(f"{RED}  - Line {i}: 알 수 없는 Song ID ({sid}){RESET}")
                error_cnt += 1
            
            if not gold:
                print(f"{RED}  - Line {i}: 정답 텍스트 누락{RESET}")
                error_cnt += 1
            
            # 난이도 값 기록
            if diff in ['0', '1', '2']:
                song_difficulty_map[sid].append(diff)
            else:
                print(f"{YELLOW}  - Line {i}: 이상한 난이도 값 ({diff}){RESET}")
                warning_cnt += 1

    # 5. 통계 및 밸런스 검증
    print("\nChecking Balance (Rule: 1 query per difficulty [0, 1, 2])...")
    
    required_set = {'0', '1', '2'}
    
    for sid in sorted(list(valid_song_ids)):
        diffs = song_difficulty_map[sid]
        diff_set = set(diffs)
        
        # 5-1. 개수 확인
        if len(diffs) != 3:
            print(f"{YELLOW}  - [Count Warning] {sid}: 쿼리 개수가 3개가 아님 (현재 {len(diffs)}개){RESET}")
            warning_cnt += 1
            continue 

        # 5-2. 구성 확인
        if diff_set != required_set:
            missing = required_set - diff_set
            print(f"{YELLOW}  - [Balance Warning] {sid}: 난이도 구성 불균형! {diffs} -> {missing} 누락됨{RESET}")
            warning_cnt += 1

    # 6. 최종 리포트
    print("\n" + "="*40)
    print(f"Validation Summary")
    print("="*40)
    
    if error_cnt == 0 and warning_cnt == 0:
        print(f"{GREEN}[SUCCESS] 완벽합니다! 개수와 난이도 밸런스까지 모두 맞습니다.{RESET}")
    else:
        if error_cnt > 0: print(f"{RED}[FAIL] {error_cnt}개의 오류 수정 필요{RESET}")
        if warning_cnt > 0: print(f"{YELLOW}[WARNING] {warning_cnt}개의 경고 확인 필요{RESET}")
    print("="*40)

if __name__ == "__main__":
    main()