import os
import csv
import re
import pandas as pd
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

# Import shared text normalization
from src.utils import normalize_text

def get_robust_token_mapping(full_lyrics_raw, gold_text):
    """
    원본 텍스트를 normalize_text로 정제한 뒤 토큰화하여,
    기존 corpus.tsv와 100% 동일한 글로벌 토큰 인덱스를 부여합니다.
    """
    # 1. 가사 전문 전처리 및 토큰화
    norm_full_lyrics = normalize_text(full_lyrics_raw)
    tokens = norm_full_lyrics.split()
    
    norm_full = ""
    token_map = []
    
    for token_idx, token in enumerate(tokens):
        token_clean = token.lower()
        norm_full += token_clean
        # 정제된 토큰의 길이만큼 현재 인덱스 부여
        token_map.extend([token_idx] * len(token_clean))
        
    # 2. 정답 텍스트도 동일한 전처리 수행 후 공백 제거
    norm_gold = normalize_text(gold_text)
    norm_gold_clean = re.sub(r'\s+', '', norm_gold).lower()
        
    return norm_full, token_map, norm_gold_clean

@hydra.main(version_base=None, config_path="conf", config_name="eval_config")
def main(cfg: DictConfig):
    if cfg.user_id is None or cfg.user_id == '???':
        suffix = ""
    else:
        suffix = str(cfg.user_id)
        
    base_dir = to_absolute_path(cfg.data.base_dir)
    lyrics_dir = to_absolute_path(cfg.data.lyrics_dir)
    
    queries_file = os.path.join(base_dir, f'queries{suffix}.tsv')
    corpus_file = os.path.join(base_dir, f'corpus{suffix}.tsv')
    orig_qrels_file = os.path.join(base_dir, f'qrels{suffix}.tsv') 
    output_file = os.path.join(base_dir, f'qrels_fixed{suffix}.tsv')
    
    if not os.path.exists(orig_qrels_file):
        print(f"[Error] 원본 qrels 파일이 없습니다: {orig_qrels_file}")
        return

    orig_df = pd.read_csv(orig_qrels_file, sep='\t')
    
    passage_info = {}
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

    qrels_data = []
    error_log = []

    print("Processing Queries with Synchronized NLP Engine...")
    with open(queries_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            qid = row['query_id']
            sid = row['song_id']
            gold_text = row['gold_span_text']
            
            lyric_path = os.path.join(lyrics_dir, f"{sid}.txt")
            if not os.path.exists(lyric_path):
                continue
                
            with open(lyric_path, 'r', encoding='utf-8') as lf:
                full_lyrics_raw = lf.read()
                
            if not full_lyrics_raw.strip():
                continue
            
            # Apply robust token mapping
            norm_full, token_map, norm_gold_clean = get_robust_token_mapping(full_lyrics_raw, gold_text)
            
            if not norm_gold_clean:
                continue

            matches = [m for m in re.finditer(re.escape(norm_gold_clean), norm_full)]
            
            if not matches:
                continue
            
            if sid in passage_info:
                for match in matches:
                    start_char_idx = match.start()
                    end_char_idx = match.end() - 1
                    
                    global_start_idx = token_map[start_char_idx]
                    global_end_idx = token_map[end_char_idx] + 1 
                    
                    for p_meta in passage_info[sid]:
                        p_start = p_meta['start']
                        p_end = p_meta['end']
                        pid = p_meta['pid']
                        
                        if global_start_idx >= p_start and global_end_idx <= p_end:
                            qrels_data.append([
                                qid, pid, global_start_idx, global_end_idx, gold_text
                            ])

    new_df = pd.DataFrame(qrels_data, columns=['query_id', 'passage_id', 'start', 'end', 'gold_span_norm'])
    orig_pairs = set(zip(orig_df['query_id'], orig_df['passage_id']))
    
    filtered_new_rows = []
    for _, row in new_df.iterrows():
        if (row['query_id'], row['passage_id']) not in orig_pairs:
            filtered_new_rows.append(row)
            orig_pairs.add((row['query_id'], row['passage_id']))
            
    filtered_new_df = pd.DataFrame(filtered_new_rows, columns=['query_id', 'passage_id', 'start', 'end', 'gold_span_norm'])
    
    final_df = pd.concat([orig_df, filtered_new_df], ignore_index=True)
    final_df = final_df.sort_values(by=['query_id', 'passage_id']).reset_index(drop=True)

    added_count = len(filtered_new_df)
    
    print("\n" + "="*60)
    print(f"[QA Report] Integrity check and additions")
    print("="*60)
    print(f" Preserved existing answers: {len(orig_df)}")
    print(f" Newly discovered answers added: {added_count}")
    
    if added_count > 0:
        print("-" * 60)
        for _, row in filtered_new_df.sort_values(by=['query_id', 'passage_id']).iterrows():
            print(f"   [NEW] Query: {row['query_id']} => Passage: {row['passage_id']} (start: {row['start']}, end: {row['end']})")
        print("-" * 60)
        
    print(f" Final answer set size: {len(final_df)}")
    print("="*60 + "\n")

    final_df.to_csv(output_file, sep='\t', index=False)
    print(f"[Success] Sorted {output_file} created.")

if __name__ == "__main__":
    main()