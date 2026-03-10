import os
import json
import logging
import numpy as np
from collections import defaultdict
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from src.dataloader import RetrievalDataLoader
from src.metrics import calculate_metrics_generic

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("ensemble_benchmark_result.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@hydra.main(version_base=None, config_path="conf", config_name="eval_config")
def main(cfg: DictConfig):
    log.info("=" * 80)
    log.info("[Hybrid Phase 3] Multi-Algorithm Ensemble Benchmark")
    log.info("=" * 80)

    phonetic_json_path = cfg.get("phonetic_path", "outputs/phonetic_run/rankings_phonetic_scale_0.json")
    semantic_json_path = cfg.get("semantic_path", "outputs/semantic_run/rankings_semantic_pipeline_scale_0.json")
    rrf_k = cfg.get("rrf_k", 60)

    base_dir = to_absolute_path(cfg.data.base_dir)
    loader = RetrievalDataLoader(base_dir, cfg.user_id)
    base_corpus = loader.load_corpus()
    queries_data = loader.load_queries()
    qrels_data = loader.load_qrels()
    pid2sid, qrels_map = loader.get_mappings(base_corpus, qrels_data)

    log.info(f"Phonetic Rankings: {phonetic_json_path}")
    phonetic_ranks = load_json(phonetic_json_path)
    log.info(f"Semantic Rankings: {semantic_json_path}")
    semantic_ranks = load_json(semantic_json_path)

    metric_k_list = [1, 5, 10, 50, 100]
    
    # 4가지 앙상블 방법론에 대한 독립적인 지표 저장소 초기화
    methods = ['Standard_RRF', 'Dynamic_RRF', 'MinMax_Weighted_Sum', 'Z_Score_Sum']
    passage_metrics = {m: {f'R@{k}': [] for k in metric_k_list} for m in methods}
    for m in methods: passage_metrics[m]['MRR'] = []
    
    song_metrics = {m: {f'R@{k}': [] for k in metric_k_list} for m in methods}
    for m in methods: song_metrics[m]['MRR'] = []

    log.info(f"Applying {len(methods)} Fusion Algorithms simultaneously...")

    for query_item in queries_data:
        qid = query_item['query_id']
        if qid not in qrels_map: continue
        if qid not in phonetic_ranks or qid not in semantic_ranks: continue

        gold_pids_set = qrels_map[qid]
        gold_sid = query_item['song_id']
        gold_sids_set = {gold_sid}

        list_a = phonetic_ranks[qid]
        list_b = semantic_ranks[qid]
        if not list_a or not list_b: continue

        # --- 메타데이터 추출 ---
        pids_a = [item['passage_id'] for item in list_a]
        pids_b = [item['passage_id'] for item in list_b]
        all_pids = set(pids_a + pids_b)

        scores_a = np.array([item['score'] for item in list_a])
        scores_b = np.array([item['score'] for item in list_b])

        min_a, max_a = np.min(scores_a), np.max(scores_a)
        min_b, max_b = np.min(scores_b), np.max(scores_b)
        mean_a, std_a = np.mean(scores_a), np.std(scores_a)
        mean_b, std_b = np.mean(scores_b), np.std(scores_b)

        # 안전장치: std가 0일 경우 Z-score 분모 보호
        if std_a == 0: std_a = 1e-9
        if std_b == 0: std_b = 1e-9
        if max_a == min_a: max_a = min_a + 1e-9
        if max_b == min_b: max_b = min_b + 1e-9

        # --- Dynamic 가중치 계산 ---
        margin_a = (scores_a[0] - min_a) / (max_a - min_a) - (scores_a[1] - min_a) / (max_a - min_a) if len(scores_a) > 1 else 0.0
        margin_b = (scores_b[0] - min_b) / (max_b - min_b) - (scores_b[1] - min_b) / (max_b - min_b) if len(scores_b) > 1 else 0.0
        tot_margin = margin_a + margin_b
        w_a, w_b = (margin_a / tot_margin, margin_b / tot_margin) if tot_margin > 0 else (0.5, 0.5)

        # --- 점수 계산 딕셔너리 ---
        scores_dict = {m: defaultdict(float) for m in methods}

        for pid in all_pids:
            rank_a = pids_a.index(pid) if pid in pids_a else 1000
            rank_b = pids_b.index(pid) if pid in pids_b else 1000

            raw_a = list_a[rank_a]['score'] if rank_a != 1000 else min_a
            raw_b = list_b[rank_b]['score'] if rank_b != 1000 else min_b

            # 1. Standard RRF
            if rank_a != 1000: scores_dict['Standard_RRF'][pid] += 1.0 / (rrf_k + rank_a + 1)
            if rank_b != 1000: scores_dict['Standard_RRF'][pid] += 1.0 / (rrf_k + rank_b + 1)

            # 2. Dynamic RRF
            if rank_a != 1000: scores_dict['Dynamic_RRF'][pid] += w_a * (1.0 / (rrf_k + rank_a + 1))
            if rank_b != 1000: scores_dict['Dynamic_RRF'][pid] += w_b * (1.0 / (rrf_k + rank_b + 1))

            # 3. MinMax Weighted Sum (0~1 정규화 합산, 5:5 비율)
            norm_a = (raw_a - min_a) / (max_a - min_a)
            norm_b = (raw_b - min_b) / (max_b - min_b)
            scores_dict['MinMax_Weighted_Sum'][pid] = norm_a + norm_b

            # 4. Z-Score Sum (표준 정규 분포 합산)
            z_a = (raw_a - mean_a) / std_a
            z_b = (raw_b - mean_b) / std_b
            scores_dict['Z_Score_Sum'][pid] = z_a + z_b

        # --- 각 방법론별 Top-100 추출 및 지표 계산 ---
        for m in methods:
            fused_pids = sorted(scores_dict[m].keys(), key=lambda x: scores_dict[m][x], reverse=True)[:100]
            
            p_scores = calculate_metrics_generic(fused_pids, gold_pids_set, metric_k_list)
            for k, v in p_scores.items(): passage_metrics[m][k].append(float(v))
                
            fused_sids = [pid2sid.get(p, "unknown") for p in fused_pids]
            s_scores = calculate_metrics_generic(fused_sids, gold_sids_set, metric_k_list)
            for k, v in s_scores.items(): song_metrics[m][k].append(float(v))

    # --- 최종 벤치마크 결과 일괄 출력 ---
    log.info("\n\n" + "="*80)
    log.info(f"[Ultimate Benchmark] Multi-Algorithm Ensemble Results")
    log.info("="*80)
    
    for m in methods:
        final_p = {k: float(np.mean(v)) for k, v in passage_metrics[m].items() if len(v) > 0}
        final_s = {k: float(np.mean(v)) for k, v in song_metrics[m].items() if len(v) > 0}
        
        log.info(f"\nMethod: {m.replace('_', ' ')}")
        log.info("-" * 60)
        log.info(f"{'Metric':<10} | {'Passage Level':<15} | {'Song Level':<15}")
        log.info("-" * 60)
        for k in ['R@1', 'R@5', 'R@10', 'R@50', 'R@100', 'MRR']:
            p_val, s_val = final_p.get(k, 0.0), final_s.get(k, 0.0)
            log.info(f"{k:<10} | {p_val:<15.4f} | {s_val:.4f}")
        log.info("-" * 60)

if __name__ == "__main__":
    main()