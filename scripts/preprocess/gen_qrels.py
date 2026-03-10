# 실행 예시:
# 1. 통합 파일 처리: python gen_qrels.py
# 2. 팀원 파일 처리: python gen_qrels.py user_id=0

import os
import csv
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from src.utils import normalize_text

def find_sub_list(sl, l):
    """
    전체 토큰 리스트(l)에서 부분 토큰 리스트(sl)가 등장하는 모든 시작 인덱스를 찾음
    """
    results = []
    s_len = len(sl)
    l_len = len(l)
    
    if s_len == 0 or s_len > l_len:
        return results

    for i in range(l_len - s_len + 1):
        if l[i : i + s_len] == sl:
            results.append(i)
    return results

@hydra.main(version_base=None, config_path="conf", config_name="eval_config")
def main(cfg: DictConfig):
    # 1. user_id 유무에 따라 읽을 파일의 접미사(suffix) 결정
    if cfg.user_id is None or cfg.user_id == '???':
        suffix = ""
        print("Processing Target: Merged Dataset (queries.tsv, corpus.tsv)")
    else:
        suffix = str(cfg.user_id)
        print(f"Processing Target: User {suffix} (queries{suffix}.tsv, corpus{suffix}.tsv)")
    
    # 2. 경로 설정 (Evaluate.yaml 참조)
    base_dir = to_absolute_path(cfg.data.base_dir)
    lyrics_dir = to_absolute_path(cfg.data.lyrics_dir)
    
    # suffix를 이용해 파일명 자동 조립
    queries_file = os.path.join(base_dir, f'queries{suffix}.tsv')
    corpus_file = os.path.join(base_dir, f'corpus{suffix}.tsv')
    output_file = os.path.join(base_dir, f'qrels{suffix}.tsv')
    
    # 파일 존재 여부 확인
    if not os.path.exists(queries_file) or not os.path.exists(corpus_file):
        print(f"[Error] 필수 파일 누락: {queries_file} 또는 {corpus_file}")
        return

    print(f"Base Directory: {base_dir}")

    # 3. Corpus Metadata 로드 (Global Index 매핑용)
    passage_info = {}
    print("Loading Corpus Metadata...")
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            sid = row['song_id']
            if sid not in passage_info:
                passage_info[sid] = []
            
            passage_info[sid].append({
                'pid': row['passage_id'],
                'start': int(row['start']),
                'end': int(row['end'])
            })

    # 4. Queries 처리 및 Qrels 생성
    qrels_data = []
    error_log = []

    print("Processing Queries...")
    with open(queries_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            qid = row['query_id']
            sid = row['song_id']
            gold_text = row['gold_span_text']
            
            # 가사 파일 로드
            lyric_path = os.path.join(lyrics_dir, f"{sid}.txt")
            if not os.path.exists(lyric_path):
                # 에러 로그에만 남기고 계속 진행
                continue
            
            with open(lyric_path, 'r', encoding='utf-8') as lf:
                full_lyrics_raw = lf.read()
            
            # 전처리
            norm_full_lyrics = normalize_text(full_lyrics_raw)
            norm_gold = normalize_text(gold_text)
            
            full_tokens = norm_full_lyrics.split()
            gold_tokens = norm_gold.split()
            
            if not gold_tokens:
                error_log.append(f"[{qid}] 전처리 후 Gold Text가 비어있음.")
                continue

            # 매칭
            match_indices = find_sub_list(gold_tokens, full_tokens)
            
            if not match_indices:
                error_log.append(f"[{qid}] 매칭 실패 (Norm: '{norm_gold}')")
                continue
            
            # Passage ID 매핑
            matched_count_for_query = 0
            
            if sid in passage_info:
                for global_start_idx in match_indices:
                    global_end_idx = global_start_idx + len(gold_tokens)
                    
                    for p_meta in passage_info[sid]:
                        p_start = p_meta['start']
                        p_end = p_meta['end']
                        pid = p_meta['pid']
                        
                        # Full Containment 조건
                        if global_start_idx >= p_start and global_end_idx <= p_end:
                            qrels_data.append([
                                qid, 
                                pid, 
                                global_start_idx, 
                                global_end_idx, 
                                norm_gold
                            ])
                            matched_count_for_query += 1
            
            if matched_count_for_query == 0:
                error_log.append(f"[{qid}] 포함하는 Passage 없음.")

    # 5. 결과 저장
    with open(output_file, 'w', encoding='utf-8', newline='') as out_f:
        writer = csv.writer(out_f, delimiter='\t')
        writer.writerow(['query_id', 'passage_id', 'start', 'end', 'gold_span_norm'])
        writer.writerows(qrels_data)
        
    print(f"\n[Success] {output_file} 생성 완료. (총 {len(qrels_data)} pairs)")
    
    if error_log:
        print("\n" + "="*50)
        print(f"[Error Report] {len(error_log)}건의 이슈 (확인 필요)")
        for err in error_log:
            print(f"   - {err}")
        print("="*50)

if __name__ == "__main__":
    main()