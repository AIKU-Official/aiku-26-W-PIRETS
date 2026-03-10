import os
import time
import json
import logging
import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from tqdm import tqdm

# 프로젝트 모듈 임포트
from src.models.bm25_ngram import BM25NGramModel
from src.metrics import calculate_metrics_generic

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="eval_config")
def main(cfg: DictConfig):
    # 1. 경로 설정 및 데이터 로딩
    # Hydra 실행 시 작업 디렉토리가 변경되므로 절대 경로로 변환
    base_dir = to_absolute_path(cfg.data.base_dir)
    output_root = os.getcwd() 
    
    log.info(f"Experiment start. Model: {cfg.model.name}")
    log.info(f"Parameters: n={cfg.model.params.n}, k1={cfg.model.params.k1}, b={cfg.model.params.b}")

    # 데이터 로딩 (IPA 버전)
    try:
        queries = pd.read_csv(os.path.join(base_dir, "queries_ipa.tsv"), sep='\t').to_dict('records')
        qrels = pd.read_csv(os.path.join(base_dir, "qrels.tsv"), sep='\t')
        base_corpus = pd.read_csv(os.path.join(base_dir, "corpus_ipa.tsv"), sep='\t').to_dict('records')
        dist_corpus_all = pd.read_csv(os.path.join(base_dir, "distractor_corpus_ipa.tsv"), sep='\t').to_dict('records')
    except Exception as e:
        log.error(f"Data loading failed: {e}")
        return

    # 실험할 디스트랙터 크기 및 지표 K 설정
    dist_sizes = [0, 1000, 10000, 20000, 30000, 43234]
    metric_ks = [1, 5, 10, 50, 100]

    for size in dist_sizes:
        # 각 사이즈별 저장 폴더 생성
        size_dir = os.path.join(output_root, f"size_{size}")
        os.makedirs(size_dir, exist_ok=True)
        
        log.info(f"\n[Evaluating] Distractors: {size}")
        
        # 지표 저장용 딕셔너리 초기화
        passage_metrics = {f'R@{k}': [] for k in metric_ks}
        passage_metrics['MRR'] = []
        song_metrics = {f'R@{k}': [] for k in metric_ks}
        song_metrics['MRR'] = []
        detailed_rows = []

        # 코퍼스 구성 및 매핑
        current_corpus = base_corpus + dist_corpus_all[:size]
        pid2sid = {item['passage_id']: item['song_id'] for item in current_corpus}
        qrels_map = qrels.groupby('query_id')['passage_id'].apply(set).to_dict()

        # 2. 모델 인덱싱 (YAML 설정값 주입)
        model = BM25NGramModel(
            n=cfg.model.params.n, 
            k1=cfg.model.params.k1, 
            b=cfg.model.params.b
        )
        model.build_index(current_corpus)

        # 3. 검색 및 평가 루프
        for q in tqdm(queries, desc=f"Evaluating Size {size}", ascii=True):
            qid = q['query_id']
            if qid not in qrels_map: continue
            
            gold_pids = qrels_map[qid]
            gold_sid = q['song_id']
            
            # 검색 수행
            results = model.search(q['text'], top_k=100)
            pred_pids = [r['passage_id'] for r in results]
            pred_sids = [pid2sid.get(pid, "unknown") for pid in pred_pids]
            
            # 지표 계산 (Passage & Song Level)
            p_scores = calculate_metrics_generic(pred_pids, gold_pids, metric_ks)
            for k, v in p_scores.items(): passage_metrics[k].append(v)
            
            s_scores = calculate_metrics_generic(pred_sids, {gold_sid}, metric_ks)
            for k, v in s_scores.items(): song_metrics[k].append(v)

            # 상세 결과 수집 (TSV 저장용)
            top1 = results[0] if results else None
            best_rank = -1
            for idx, pid in enumerate(pred_pids):
                if pid in gold_pids:
                    best_rank = idx + 1
                    break
            
            detailed_rows.append({
                "query_id": qid,
                "query_text": q['text'],
                "predicted_passage_id": top1['passage_id'] if top1 else "None",
                "difficulty": q.get('difficulty', 'N/A'),
                "best_gold_rank": best_rank,
                "is_passage_correct": top1['passage_id'] in gold_pids if top1 else False,
                "is_song_correct": pid2sid.get(top1['passage_id']) == gold_sid if top1 else False,
                "predicted_passage_text": top1['text'].replace('\t', ' ') if top1 else ""
            })

        # 4. 결과 요약 및 파일 저장
        final_p = {k: float(np.mean(v)) for k, v in passage_metrics.items()}
        final_s = {k: float(np.mean(v)) for k, v in song_metrics.items()}

        # JSON 저장 (Metadata에 YAML 설정값 반영)
        json_output = {
            "meta": {
                "model": cfg.model.name,
                "params": OmegaConf.to_container(cfg.model.params),
                "distractor_size": size,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "metrics": {"passage_level": final_p, "song_level": final_s}
        }
        with open(os.path.join(size_dir, "metrics.json"), "w", encoding='utf-8') as f:
            json.dump(json_output, f, indent=4, ensure_ascii=False)

        # TSV 저장
        pd.DataFrame(detailed_rows).to_csv(os.path.join(size_dir, "prediction_results.tsv"), sep='\t', index=False)

        # 터미널 출력 및 LOG 파일 저장
        log_content = [
            "="*60,
            f"[Final Result] ({cfg.model.name}) - Distractors: {size}",
            "="*60,
            f"{'Metric':<10} | {'Passage Level':<15} | {'Song Level':<15}",
            "-" * 45
        ]
        for k in ['R@1', 'R@5', 'R@10', 'R@50', 'R@100', 'MRR']:
            log_content.append(f"{k:<10} | {final_p[k]:.4f}          | {final_s[k]:.4f}")
        log_content.append("="*60)

        log_str = "\n".join(log_content)
        print(log_str)
        with open(os.path.join(size_dir, "evaluate.log"), "w") as f:
            f.write(log_str)

    log.info(f"\nAll experiments completed! Results saved to: {output_root}")

if __name__ == "__main__":
    main()