import json
from tqdm import tqdm

def merge_to_single_file(tsv_path, doc_jsonl_path, query_jsonl_path, output_path):
    print(f"[1/4] Reading TSV structure... ({tsv_path})")
    corpus = {}
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[0].lower() != 'id':
                corpus[parts[0]] = {'id': parts[0], 'raw': parts[1], 'g2p': '', 'query_g2p': ''}
                
    print(f"[2/4] Merging document G2P... ({doc_jsonl_path})")
    with open(doc_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            doc_id = str(data.get('id', ''))
            if doc_id in corpus:
                corpus[doc_id]['g2p'] = data.get('phoneme', '')
                
    print(f"[3/4] Merging query G2P... ({query_jsonl_path})")
    with open(query_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            doc_id = str(data.get('id', ''))
            if doc_id in corpus:
                corpus[doc_id]['query_g2p'] = data.get('query_g2p', '')
                
    print(f"[4/4] Saving to single file... ({output_path})")
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc_id, data in tqdm(corpus.items(), desc="Writing JSONL"):
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
            
    print("Merge complete! Use this single file for training.")

# 실행 예시 (train과 val을 각각 병합)
merge_to_single_file(
    'data/train_dataset/train.tsv', 
    'data/train_dataset/train_g2p.jsonl', 
    'data/train_dataset/train_query_g2p.jsonl',
    'data/train_dataset/unified_train.jsonl'
)

merge_to_single_file(
    'data/train_dataset/val.tsv', 
    'data/train_dataset/val_g2p.jsonl', 
    'data/train_dataset/val_query_g2p.jsonl',
    'data/train_dataset/unified_val.jsonl'
)