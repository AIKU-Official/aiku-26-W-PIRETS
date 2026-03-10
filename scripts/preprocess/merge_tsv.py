import pandas as pd

# 1. TSV 파일 로드 (sep='\t' 필수)
df1 = pd.read_csv('data/eval_dataset/qrels.tsv', sep='\t')
df2 = pd.read_csv('data/eval_dataset/memory_qrels.tsv', sep='\t')

# 2. DataFrame 병합 (세로로 이어 붙이기)
merged_df = pd.concat([df1, df2], ignore_index=True)

# 3. 새로운 TSV로 저장 (인덱스 제외)
merged_df.to_csv('data/eval_dataset/merged_qrels.tsv', sep='\t', index=False)
print(f"Merge complete: {len(merged_df)} total rows saved.")