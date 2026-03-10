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
        logging.FileHandler("ensemble_minmax_search.log", mode='a', encoding='utf-8'),
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
    log.info("[Hybrid Phase 3] MinMax Weight Grid Search (Phonetic vs Semantic)")
    log.info("=" * 80)

    phonetic_json_path = cfg.get("phonetic_path", "outputs/phonetic_run/rankings_phonetic_scale_0.json")
    semantic_json_path = cfg.get("semantic_path", "outputs/semantic_run/rankings_semantic_pipeline_scale_0.json")

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
    
    # 가중치 조합 생성 (Phonetic:Semantic 비율)
    weight_combinations = [
        (0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6),
        (0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1), (1.0, 0.0)
    ]
    
    # 라벨링 포맷: "P_0.3_S_0.7"
    methods = [f"P_{p:.1f}_S_{s:.1f}" for p, s in weight_combinations]
    
    passage_metrics = {m: {f'R@{k}': [] for k in metric_k_list} for m in methods}
    for m in methods: passage_metrics[m]['MRR'] = []
    
    song_metrics = {m: {f'R@{k}': [] for k in metric_k_list} for m in methods}
    for m in methods: song_metrics[m]['MRR'] = []

    log.info(f"Sweeping {len(weight_combinations)} weight combinations for MinMax Fusion...")

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

        # --- 메타데이터 추출 및 Min/Max 확인 ---
        pids_a = [item['passage_id'] for item in list_a]
        pids_b = [item['passage_id'] for item in list_b]
        all_pids = set(pids_a + pids_b)

        scores_a = np.array([item['score'] for item in list_a])
        scores_b = np.array([item['score'] for item in list_b])

        min_a, max_a = np.min(scores_a), np.max(scores_a)
        min_b, max_b = np.min(scores_b), np.max(scores_b)

        # 안전장치 (Zero Division 방지)
        if max_a == min_a: max_a = min_a + 1e-9
        if max_b == min_b: max_b = min_b + 1e-9

        # --- 가중치 조합별 점수 계산 딕셔너리 ---
        scores_dict = {m: defaultdict(float) for m in methods}

        for pid in all_pids:
            # 리스트에 존재하지 않는 문서는 최하위 점수(min)로 처리하여 Norm 값이 0.0이 되도록 보간
            raw_a = list_a[pids_a.index(pid)]['score'] if pid in pids_a else min_a
            raw_b = list_b[pids_b.index(pid)]['score'] if pid in pids_b else min_b

            norm_a = (raw_a - min_a) / (max_a - min_a)
            norm_b = (raw_b - min_b) / (max_b - min_b)

            # 11가지 가중치 동시 적용
            for (w_p, w_s), m in zip(weight_combinations, methods):
                scores_dict[m][pid] = (w_p * norm_a) + (w_s * norm_b)

        # --- 가중치 조합별 Top-100 추출 및 지표 계산 ---
        for m in methods:
            fused_pids = sorted(scores_dict[m].keys(), key=lambda x: scores_dict[m][x], reverse=True)[:100]
            
            p_scores = calculate_metrics_generic(fused_pids, gold_pids_set, metric_k_list)
            for k, v in p_scores.items(): passage_metrics[m][k].append(float(v))
                
            fused_sids = [pid2sid.get(p, "unknown") for p in fused_pids]
            s_scores = calculate_metrics_generic(fused_sids, gold_sids_set, metric_k_list)
            for k, v in s_scores.items(): song_metrics[m][k].append(float(v))

    # --- 최종 Grid Search 결과 일괄 출력 ---
    log.info("\n\n" + "="*100)
    log.info(f"[MinMax Weight Sweep] Phonetic vs Semantic Full Results")
    log.info("="*100)
    
    # 헤더 출력 (열 너비 고정)
    log.info(f"{'Weight (Phonetic:Semantic)':<28} | {'MRR':<8} | {'R@1':<8} | {'R@5':<8} | {'R@10':<8} | {'R@50':<8} | {'R@100':<8}")
    log.info("-" * 100)
    
    for m in methods:
        # 각 지표별 평균 계산 (값이 없는 경우 0.0으로 처리)
        p_mrr = np.mean(passage_metrics[m]['MRR']) if passage_metrics[m]['MRR'] else 0.0
        p_r1 = np.mean(passage_metrics[m]['R@1']) if passage_metrics[m]['R@1'] else 0.0
        p_r5 = np.mean(passage_metrics[m]['R@5']) if passage_metrics[m]['R@5'] else 0.0
        p_r10 = np.mean(passage_metrics[m]['R@10']) if passage_metrics[m]['R@10'] else 0.0
        p_r50 = np.mean(passage_metrics[m]['R@50']) if passage_metrics[m]['R@50'] else 0.0
        p_r100 = np.mean(passage_metrics[m]['R@100']) if passage_metrics[m]['R@100'] else 0.0
        
        # 포맷팅하여 출력
        log.info(f"{m:<28} | {p_mrr:<8.4f} | {p_r1:<8.4f} | {p_r5:<8.4f} | {p_r10:<8.4f} | {p_r50:<8.4f} | {p_r100:<8.4f}")
    
    log.info("="*100)
    log.info("Complete metrics are fully logged in 'ensemble_minmax_search.log'.")

if __name__ == "__main__":
    main()