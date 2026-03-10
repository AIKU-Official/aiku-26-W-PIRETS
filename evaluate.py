import os
import time
import json
import csv
import logging
import inspect
import numpy as np
import torch
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig

from src.models import get_model
from src.dataloader import RetrievalDataLoader
from src.metrics import calculate_metrics_generic

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="eval_config")
def main(cfg: DictConfig):
    # Output 경로 설정
    try:
        output_dir = HydraConfig.get().runtime.output_dir
    except:
        output_dir = os.getcwd()

    log.info("=" * 60)
    log.info("[Experiment Start] Distractor Scaling Evaluation")
    log.info("=" * 60)

    # 1. Base 데이터 로드
    base_dir = to_absolute_path(cfg.data.base_dir)
    loader = RetrievalDataLoader(base_dir, cfg.user_id)

    base_corpus = loader.load_corpus()
    queries_data = loader.load_queries()
    qrels_data = loader.load_qrels()
    pid2sid, qrels_map = loader.get_mappings(base_corpus, qrels_data)

    # 2. Distractor 코퍼스 로드
    distractor_corpus = []
    distractor_path = os.path.join(base_dir, "distractor_corpus_g2p.tsv")

    log.info(f"Loading Distractors from: {distractor_path}")
    if os.path.exists(distractor_path):
        with open(distractor_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                row['text'] = row.get('text', '').strip()
                distractor_corpus.append(row)
        log.info(f"Loaded {len(distractor_corpus)} Distractors successfully.")
    else:
        log.warning(f"Distractor file not found at {distractor_path}! Proceeding with base corpus only.")

    log.info(f"Base Passages: {len(base_corpus)} | Total Distractors Available: {len(distractor_corpus)}")

    # 3. 모델 초기화 및 체크포인트 로딩
    model = get_model(cfg.model)

    retriever_ckpt = cfg.get("retriever_checkpoint")
    reranker_ckpt = cfg.get("reranker_checkpoint")
    single_ckpt = cfg.get("checkpoint_path")

    # Pipeline Mode: retriever + reranker 체크포인트
    if retriever_ckpt and reranker_ckpt:
        log.info("[Pipeline Mode] Loading retriever and reranker checkpoints.")
        if hasattr(model, 'load_checkpoints'):
            model.load_checkpoints(retriever_path=retriever_ckpt, reranker_path=reranker_ckpt)
        else:
            log.warning("Model does not support pipeline checkpoint loading!")

    # Single Mode: 단일 체크포인트
    elif single_ckpt and os.path.exists(single_ckpt):
        log.info("[Single Mode] Loading checkpoint_path.")

        if 'reranker' in cfg.model.name.lower():
            log.error("[Architecture Mismatch] Reranker pipeline model with single retriever weights!")
            log.error("Reranker will use random weights. Change model config to a standalone retriever.")

        if hasattr(model, 'load_state_dict'):
            state_dict = torch.load(single_ckpt, map_location='cpu')
            new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=False)

    else:
        log.error("No valid checkpoint path found in config.")

    # Device 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, 'to'):
        model.to(device)
    if hasattr(model, 'eval'):
        model.eval()

    # 4. Scaling 실험 단위 설정
    raw_sizes = cfg.get("distractor_sizes", [0, 1000, 10000, 20000, 30000, 43234])
    if isinstance(raw_sizes, ListConfig):
        distractor_sizes = OmegaConf.to_container(raw_sizes, resolve=True)
    else:
        distractor_sizes = list(raw_sizes)

    experiment_results = []
    trend_json_path = os.path.join(output_dir, "distractor_scaling_trend.json")
    if os.path.exists(trend_json_path):
        with open(trend_json_path, 'r', encoding='utf-8') as f:
            trend_data = json.load(f)
            experiment_results = trend_data.get("trend_results", [])

    model_name_log = cfg.model.name

    # Scale-level 체크포인트 로드
    checkpoint_file = os.path.join(output_dir, "temp_checkpoint.json")
    completed_sizes = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            completed_sizes = json.load(f)
        log.info(f"Resuming from checkpoint. Already completed scales: {completed_sizes}")

    for size in distractor_sizes:
        # 이미 완료된 Scale이면 건너뛰기
        if size in completed_sizes:
            log.info(f"Scale {size} is already completed. Skipping evaluation...")

            for res in experiment_results:
                if res["distractor_size"] == size:
                    p_mets = res["metrics"]["passage_level"]
                    s_mets = res["metrics"]["song_level"]

                    log.info("="*60)
                    log.info(f"[Loaded Past Result] ({model_name_log}) - Distractors: {size}")
                    log.info("="*60)
                    log.info(f"{'Metric':<10} | {'Passage Level':<15} | {'Song Level':<15}")
                    log.info("-" * 60)
                    for k in ['R@1', 'R@5', 'R@10', 'R@50', 'R@100', 'MRR']:
                        log.info(f"{k:<10} | {p_mets.get(k, 0.0):<15.4f} | {s_mets.get(k, 0.0):.4f}")
                    log.info("="*60)
            continue

        if size > len(distractor_corpus):
            size = len(distractor_corpus)

        print(f"\n[Experiment] distractor: {size}")

        # 순차 슬라이싱
        current_distractors = distractor_corpus[:size]

        # Base 코퍼스에 Distractor 병합
        current_corpus = base_corpus + current_distractors

        # 단일 마스터 캐시 사용
        base_cache_path = cfg.get("embed_cache_path", f"outputs/cache/{cfg.model.name}_master_embed.pt")
        os.makedirs(os.path.dirname(base_cache_path), exist_ok=True)

        # Indexing
        build_index_sig = inspect.signature(model.build_index)
        if 'cache_path' in build_index_sig.parameters:
            model.build_index(current_corpus, cache_path=base_cache_path)
        else:
            model.build_index(current_corpus)

        # 5. 쿼리 단위 마이크로 체크포인트
        metric_k_list = [1, 5, 10, 50, 100]
        query_ckpt_file = os.path.join(output_dir, f"query_checkpoint_scale_{size}.json")
        processed_qids = set()

        if os.path.exists(query_ckpt_file):
            with open(query_ckpt_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            passage_metrics = state['passage_metrics']
            song_metrics = state['song_metrics']
            detailed_rows = state['detailed_rows']
            processed_qids = set(state['processed_qids'])
            current_rankings = state.get('current_rankings', {})
            log.info(f"[Micro Resume] Scale {size}: Resuming from {len(processed_qids)} completed queries.")
        else:
            passage_metrics = {f'R@{k}': [] for k in metric_k_list}
            passage_metrics['MRR'] = []
            song_metrics = {f'R@{k}': [] for k in metric_k_list}
            song_metrics['MRR'] = []
            detailed_rows = []
            current_rankings = {}

        final_top_k = max(cfg.get("top_k", 100), max(metric_k_list))

        # 6. 평가 루프
        for query_item in tqdm(queries_data, desc=f"Evaluating Size {size}"):
            qid = query_item['query_id']

            if qid in processed_qids:
                continue

            if qid not in qrels_map: continue

            gold_pids_set = qrels_map[qid]
            gold_sid = query_item['song_id']
            gold_sids_set = {gold_sid}

            query_text = query_item.get('text')
            difficulty = query_item.get('difficulty', 'N/A')

            search_results = model.search(query_text, top_k=final_top_k)

            # Rankings 저장 (RRF 앙상블용)
            current_rankings[qid] = [{'passage_id': r['passage_id'], 'score': float(r['score'])} for r in search_results[:100]]

            pred_pids = [r['passage_id'] for r in search_results]
            p_scores = calculate_metrics_generic(pred_pids, gold_pids_set, metric_k_list)
            for k, v in p_scores.items(): passage_metrics[k].append(float(v))

            pred_sids = [pid2sid.get(pid, "unknown") for pid in pred_pids]
            s_scores = calculate_metrics_generic(pred_sids, gold_sids_set, metric_k_list)
            for k, v in s_scores.items(): song_metrics[k].append(float(v))

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

            # 실시간 쿼리 체크포인트 백업
            processed_qids.add(qid)
            current_state = {
                "passage_metrics": passage_metrics,
                "song_metrics": song_metrics,
                "detailed_rows": detailed_rows,
                "processed_qids": list(processed_qids),
                "current_rankings": current_rankings
            }
            with open(query_ckpt_file, 'w', encoding='utf-8') as f:
                json.dump(current_state, f)

        # 7. 현재 Scale 결과 집계
        final_p_metrics = {k: float(np.mean(v)) for k, v in passage_metrics.items() if len(v) > 0}
        final_s_metrics = {k: float(np.mean(v)) for k, v in song_metrics.items() if len(v) > 0}

        result_lines = []
        result_lines.append("="*60)
        result_lines.append(f"[Final Result] ({model_name_log}) - Distractors: {size}")
        result_lines.append("="*60)
        result_lines.append(f"{'Metric':<10} | {'Passage Level':<15} | {'Song Level':<15}")
        result_lines.append("-" * 60)

        for k in ['R@1', 'R@5', 'R@10', 'R@50', 'R@100', 'MRR']:
            p_val = final_p_metrics.get(k, 0.0)
            s_val = final_s_metrics.get(k, 0.0)
            result_lines.append(f"{k:<10} | {p_val:<15.4f} | {s_val:.4f}")

        result_lines.append("="*60)

        final_log_message = "\n" + "\n".join(result_lines) + "\n"
        log.info(final_log_message)

        # Trend 기록
        experiment_results.append({
            "distractor_size": size,
            "total_corpus_size": len(current_corpus),
            "metrics": {
                "passage_level": final_p_metrics,
                "song_level": final_s_metrics
            }
        })

        # TSV 개별 저장
        if detailed_rows:
            tsv_path = os.path.join(output_dir, f"prediction_results_scale_{size}.tsv")
            fieldnames = ["query_id", "query_text", "predicted_passage_id", "difficulty", "best_gold_rank", "is_passage_correct", "is_song_correct", "predicted_passage_text"]
            with open(tsv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
                writer.writeheader()
                writer.writerows(detailed_rows)

        # Rankings JSON 저장
        ranking_json_path = os.path.join(output_dir, f"rankings_{cfg.model.name.replace('/', '_')}_scale_{size}.json")
        with open(ranking_json_path, "w", encoding="utf-8") as f:
            json.dump(current_rankings, f, ensure_ascii=False)

        # Scale 체크포인트 업데이트
        completed_sizes.append(size)

        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(completed_sizes, f)
        log.info(f"Checkpoint updated. Completed scales: {completed_sizes}")

        # 마이크로 체크포인트 정리
        if os.path.exists(query_ckpt_file):
            os.remove(query_ckpt_file)

    # 8. 최종 Trend JSON 저장
    log.info("[Experiment End] Distractor Scaling Complete")

    if isinstance(cfg.model, (DictConfig, ListConfig)):
        model_conf_dict = OmegaConf.to_container(cfg.model, resolve=True)
    else:
        model_conf_dict = cfg.model

    trend_json_path = os.path.join(output_dir, "distractor_scaling_trend.json")
    final_output_json = {
        "meta": {
            "model": cfg.model.name,
            "params": model_conf_dict.get("params", {}),
            "data_user": str(cfg.user_id) if cfg.user_id is not None else "Merged",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "trend_results": experiment_results
    }

    with open(trend_json_path, "w", encoding='utf-8') as f:
        json.dump(final_output_json, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
