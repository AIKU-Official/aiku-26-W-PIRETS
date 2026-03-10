# 실행 예시:
# 1. 통합 파일 처리: python gen_corpus.py
# 2. 팀원 파일 처리: python gen_corpus.py user_id=0

import os
import csv
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from src.utils import normalize_text

WINDOW_SIZE = 64
STRIDE = 32

def create_sliding_windows(text, window_size, stride):
    tokens = text.split()
    if len(tokens) <= window_size:
        yield ' '.join(tokens), 0, len(tokens)
        return
    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i : i + window_size]
        start = i
        end = i + len(chunk_tokens)
        yield ' '.join(chunk_tokens), start, end
        if i + window_size >= len(tokens):
            break

@hydra.main(version_base=None, config_path="conf", config_name="eval_config")
def main(cfg: DictConfig):
    # [변경] user_id 유무에 따라 읽을 파일의 접미사(suffix) 결정
    if cfg.user_id is None or cfg.user_id == '???':
        suffix = ""
        print("Processing Target: Merged Dataset (songs.tsv)")
    else:
        suffix = str(cfg.user_id)
        print(f"Processing Target: User {suffix} (songs{suffix}.tsv)")
    
    # 2. 경로 설정
    base_dir = to_absolute_path(cfg.data.base_dir)
    lyrics_dir = to_absolute_path(cfg.data.lyrics_dir)
    
    # [변경] f-string에 suffix 적용
    # suffix가 ""이면 songs.tsv, "0"이면 songs0.tsv가 됨
    songs_file = os.path.join(base_dir, f'songs{suffix}.tsv')
    output_file = os.path.join(base_dir, f'corpus{suffix}.tsv')

    if not os.path.exists(songs_file):
        print(f"[Error] {songs_file} 파일이 없습니다.")
        return

    print(f"Base Dir: {base_dir}")
    
    corpus_data = []

    try:
        with open(songs_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)
            if header and not header[0].startswith('S'): f.seek(0)

            for row in reader:
                if not row: continue
                song_id = row[0].strip()
                lyric_path = os.path.join(lyrics_dir, f"{song_id}.txt")
                
                if not os.path.exists(lyric_path):
                    continue
                
                with open(lyric_path, 'r', encoding='utf-8') as lf:
                    raw_text = lf.read()
                
                clean_text = normalize_text(raw_text)
                passages = list(create_sliding_windows(clean_text, WINDOW_SIZE, STRIDE))
                
                for idx, (passage_text, start, end) in enumerate(passages):
                    passage_id = f"{song_id}_P{idx+1:04d}"
                    corpus_data.append([passage_id, song_id, passage_text, start, end])

        with open(output_file, 'w', encoding='utf-8', newline='') as out_f:
            writer = csv.writer(out_f, delimiter='\t')
            writer.writerow(['passage_id', 'song_id', 'text', 'start', 'end'])
            writer.writerows(corpus_data)
            
        print(f"[Success] {output_file} Saved.")

    except Exception as e:
        print(f"[Error] {e}")

if __name__ == "__main__":
    main()