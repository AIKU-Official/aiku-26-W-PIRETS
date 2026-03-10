# 1. Python 표준 라이브러리 (Standard Library)
import os
import time
import json
import csv
import random
import logging
import inspect

# 2. 서드파티 데이터 연산 및 딥러닝 라이브러리 (Third-party)
import numpy as np
import torch
from tqdm import tqdm

# 3. Hydra 및 설정 관리 라이브러리 (Configuration)
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig

# 4. 커스텀 로컬 모듈 (Local Modules)
from src.models import get_model
from src.dataloader import RetrievalDataLoader
from src.metrics import calculate_metrics_generic

# 로거 초기화
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="eval_config")
def main(cfg: DictConfig):
    output_dir = os.getcwd()
    log.info(f"[Experiment Start] Distractor Scaling Evaluation")

    # 1. 전체 데이터 로드
    base_dir = to_absolute_path(cfg.data.base_dir)
    loader = RetrievalDataLoader(base_dir, cfg.user_id)
    full_corpus = loader.load_corpus()
    queries_data = loader.load_queries()
    qrels_data = loader.load_qrels()
    pid2sid, qrels_map = loader.get_mappings(full_corpus, qrels_data)

    # 2. 정답(Gold) Passage ID 추출 (절대 누락되면 안 되는 문서들)
    gold_pids = set()
    for qid in qrels_map:
        gold_pids.update(qrels_map[qid])
    
    # 전체 코퍼스를 Gold와 Non-Gold(잠재적 Distractor)로 분리
    gold_corpus = [doc for doc in full_corpus if doc['passage_id'] in gold_pids]
    non_gold_corpus = [doc for doc in full_corpus if doc['passage_id'] not in gold_pids]
    
    log.info(f"Gold Passages: {len(gold_corpus)} | Available Distractors: {len(non_gold_corpus)}")

    # 3. 모델 초기화
    model = get_model(cfg.model)
    if hasattr(model, 'load_checkpoints'):
        model.load_checkpoints(
            retriever_path=cfg.get("retriever_checkpoint"), 
            reranker_path=cfg.get("reranker_checkpoint")
        )

    # 4. Scaling 실험 설정
    # 노이즈 크기를 1,000개부터 전체까지 점진적으로 증가
    distractor_sizes = [0, 1000, 5000, 10000, len(non_gold_corpus)]
    experiment_results = []

    for size in distractor_sizes:
        if size > len(non_gold_corpus):
            size = len(non_gold_corpus)

        log.info("=" * 50)
        log.info(f"[Scale: {size} Distractors] Evaluation Started")
        
        # 현재 Scale에 맞는 코퍼스 구성 (Gold + Sampled Distractors)
        random.seed(42)
        sampled_distractors = random.sample(non_gold_corpus, size)
        
        # 현재 Scale의 코퍼스와 Gold corpus 매핑 정보 준비
        current_corpus = gold_corpus + sampled_distractors
        log.info(f"Current Corpus Size: {len(current_corpus)}")

        # 리플렉션 방어 코드
        base_cache_path = cfg.get("embed_cache_path", "outputs/cache/embed.pt")
        cache_dir, cache_name = os.path.split(base_cache_path)
        current_cache_path = os.path.join(cache_dir, f"size_{size}_{cache_name}")

        # Indexing (리플렉션 방어 코드 적용)
        build_index_sig = inspect.signature(model.build_index)
        start_time = time.time()
        if 'cache_path' in build_index_sig.parameters:
            model.build_index(current_corpus, cache_path=current_cache_path)
        else:
            model.build_index(current_corpus)
        log.info(f"Indexing Time: {time.time() - start_time:.2f}s")

        # -------------------------------------------------------------
        # [이전 코드 생략] model.build_index(...) 직후부터 시작됩니다.
        # -------------------------------------------------------------
        
        # 3. Reset metrics for each scale
        metric_k_list = [1, 5, 10, 50, 100]
        passage_metrics = {f'R@{k}': [] for k in metric_k_list}
        passage_metrics['MRR'] = []
        song_metrics = {f'R@{k}': [] for k in metric_k_list}
        song_metrics['MRR'] = []
        
        config_top_k = cfg.get("top_k", 100)
        final_top_k = max(config_top_k, max(metric_k_list))
        
        detailed_rows = []
        log.info(f"[Search] Running Evaluation for Scale: {size}...")

        # 4. 평가 루프
        for query_item in tqdm(queries_data, ascii=True, desc=f"Eval Scale {size}"):
            qid = query_item['query_id']
            if qid not in qrels_map: continue
                
            gold_pids_set = qrels_map[qid]
            gold_sid = query_item['song_id']
            gold_sids_set = {gold_sid}
            
            query_text = query_item.get('text')
            difficulty = query_item.get('difficulty', 'N/A')

            # [검색] (현재 Scale의 Index를 바탕으로 검색)
            search_results = model.search(query_text, top_k=final_top_k)
            
            # [지표 계산]
            pred_pids = [r['passage_id'] for r in search_results]
            p_scores = calculate_metrics_generic(pred_pids, gold_pids_set, metric_k_list)
            for k, v in p_scores.items(): passage_metrics[k].append(v)
                
            pred_sids = [pid2sid.get(pid, "unknown") for pid in pred_pids]
            s_scores = calculate_metrics_generic(pred_sids, gold_sids_set, metric_k_list)
            for k, v in s_scores.items(): song_metrics[k].append(v)

            # [상세 결과 기록]
            if cfg.get("save_detail", True):
                best_gold_rank = -1
                for idx, pid in enumerate(pred_pids):
                    if pid in gold_pids_set:
                        best_gold_rank = idx + 1
                        break
                
                top1 = search_results[0] if search_results else None
                pred_pid = top1['passage_id'] if top1 else "None"
                pred_text = top1['text'].replace('\n', ' ').replace('\t', ' ') if top1 else "None"
                
                detailed_rows.append({
                    "query_id": qid,
                    "query_text": query_text,
                    "predicted_passage_id": pred_pid,
                    "difficulty": difficulty,
                    "best_gold_rank": best_gold_rank,
                    "is_passage_correct": (pred_pid in gold_pids_set),
                    "is_song_correct": (pid2sid.get(pred_pid) == gold_sid),
                    "predicted_passage_text": pred_text
                })

        # 5. 현재 Scale에 대한 결과 취합 및 저장
        final_p_metrics = {k: float(np.mean(v)) for k, v in passage_metrics.items() if len(v) > 0}
        final_s_metrics = {k: float(np.mean(v)) for k, v in song_metrics.items() if len(v) > 0}
        
        log.info("-" * 45)
        log.info(f"Results for Distractor Scale: {size} (Corpus: {len(current_corpus)})")
        log.info("-" * 45)
        for k in ['R@1', 'R@5', 'R@10', 'R@50', 'R@100', 'MRR']:
            p_val = final_p_metrics.get(k, 0.0)
            s_val = final_s_metrics.get(k, 0.0)
            log.info(f"{k:<10} | {p_val:.4f}          | {s_val:.4f}")
        
        # 전체 Trend 리스트에 현재 Scale 결과 누적
        experiment_results.append({
            "distractor_size": size,
            "total_corpus_size": len(current_corpus),
            "metrics": {
                "passage_level": final_p_metrics,
                "song_level": final_s_metrics
            }
        })

        # 현재 Scale에 대한 상세 TSV 개별 저장
        if detailed_rows:
            tsv_path = os.path.join(output_dir, f"prediction_results_scale_{size}.tsv")
            fieldnames = ["query_id", "query_text", "predicted_passage_id", "difficulty", "best_gold_rank", "is_passage_correct", "is_song_correct", "predicted_passage_text"]
            with open(tsv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
                writer.writeheader()
                writer.writerows(detailed_rows)
            log.info(f"Saved Details for Scale {size} -> {tsv_path}")

    # =========================================================
    # 6. 실험 종료: 모든 Scale의 결과를 모아 하나의 JSON(Trend)으로 저장
    # =========================================================
    log.info("="*60)
    log.info(f"[Experiment End] Distractor Scaling Complete")
    log.info("="*60)

    # 파라미터 추출
    if isinstance(cfg.model, (DictConfig, ListConfig)):
        model_conf_dict = OmegaConf.to_container(cfg.model, resolve=True)
    else:
        model_conf_dict = cfg.model
    model_params = model_conf_dict.get("params", {})
    user_label = str(cfg.user_id) if cfg.user_id is not None else "Merged"

    # 추이(Trend) 기록용 JSON 생성
    trend_json_path = os.path.join(output_dir, "distractor_scaling_trend.json")
    final_output_json = {
        "meta": {
            "model": cfg.model.name,
            "params": model_params,
            "data_user": user_label,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_type": "Distractor Scaling"
        },
        "trend_results": experiment_results
    }
    
    with open(trend_json_path, "w", encoding='utf-8') as f:
        json.dump(final_output_json, f, indent=4, ensure_ascii=False)
        
    log.info(f"[Saved Trend] Final experiment trend saved to: {trend_json_path}")

if __name__ == "__main__":
    main()