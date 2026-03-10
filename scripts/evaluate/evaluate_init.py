import os
import time
import json
import logging
import hydra
import numpy as np
import csv
import torch
from omegaconf import DictConfig, OmegaConf, ListConfig
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm
import inspect

# [Refactoring] 분리된 모듈 사용
from src.models import get_model
from src.dataloader import RetrievalDataLoader
from src.metrics import calculate_metrics_generic

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="eval_config")
def main(cfg: DictConfig):
    # 0. Output 경로 확인 (Hydra가 설정한 Flat한 경로)
    try:
        output_dir = HydraConfig.get().runtime.output_dir
    except:
        output_dir = os.getcwd()
        
    log.info(f"[Start Evaluation] Model: {cfg.model.name}")
    log.info(f"[Output Dir] {output_dir}")

    # 1. 데이터 로드 (Dataloader 사용)
    base_dir = to_absolute_path(cfg.data.base_dir)
    loader = RetrievalDataLoader(base_dir, cfg.user_id)
    
    corpus_data = loader.load_corpus()
    queries_data = loader.load_queries()
    qrels_data = loader.load_qrels()
    
    if not corpus_data or not queries_data:
        log.error("필수 데이터가 누락되어 종료합니다.")
        return

    # 매핑 정보 가져오기
    pid2sid, qrels_map = loader.get_mappings(corpus_data, qrels_data)
    
    log.info(f" - Corpus: {len(corpus_data)}")
    log.info(f" - Queries: {len(qrels_map)}")

    # 2. 모델 빌드 & 체크포인트 로드
    model = get_model(cfg.model)
    
    # 체크포인트 로드 로직
    checkpoint_path = cfg.get("checkpoint_path", None)
    retriever_ckpt = cfg.get("retriever_checkpoint", None)
    reranker_ckpt = cfg.get("reranker_checkpoint", None)
    
    # Pipeline 모델인 경우, retriever와 reranker 체크포인트를 별도로 로드
    if hasattr(model, 'load_checkpoints'):
        log.info("[Pipeline Mode] 2-Stage Pipeline model detected. Loading multiple checkpoints...")
        model.load_checkpoints(retriever_path=retriever_ckpt, reranker_path=reranker_ckpt)
        
    # 단일 모델 로직 (기존 BM25, 단일 Retriever 등)
    elif checkpoint_path and os.path.exists(checkpoint_path):
        if hasattr(model, 'load_state_dict'):
            log.info(f"Loading single checkpoint from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            log.info("Checkpoint loaded successfully!")
        else:
            log.info("Assuming checkpoints are handled inside get_model().")
    elif checkpoint_path:
        log.warning("Checkpoint path provided but file not found!")

    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(model, torch.nn.Module):
        model.to(device)
        model.eval()
        log.info(f"Model moved to {device}")
    elif hasattr(model, 'to'): 
        # EndToEndSearcher 내부에 커스텀 to() 메서드를 만들어 리랭커만 GPU로 보내도록 호출
        model.to(device)
        log.info(f"Pipeline internal models moved to {device}")
    else:
        log.info(f"Model [{cfg.model.name}] is entirely CPU-based (e.g., pure BM25).")
        
    # yaml 파일에서 캐시 저장 경로 가져오기 (없으면 None)
    embed_cache_path = cfg.get("embed_cache_path", None)

    log.info("[Index] Building Index...")
    start_time = time.time()
    
    # 인덱스 빌드 시, 모델의 build_index 메서드가 'cache_path' 인자를 지원하는지 리플렉션으로 확인
    build_index_sig = inspect.signature(model.build_index)
    
    if 'cache_path' in build_index_sig.parameters:
        # Dense Retriever 계열: 캐시 경로를 전달하여 스마트하게 인덱싱
        model.build_index(corpus_data, cache_path=embed_cache_path)
    else:
        # BM25 계열: 기존 방식대로 코퍼스 데이터만 전달
        if embed_cache_path:
            log.info(f"Model [{cfg.model.name}] does not support 'cache_path'. Ignoring cache settings.")
        model.build_index(corpus_data)
        
    log.info(f"[Index] Build Time: {time.time() - start_time:.2f}s")

    # ---------------------------------------------------------
    # 3. 평가 준비 및 임시 저장소(Checkpoint) 로드
    # ---------------------------------------------------------
    metric_k_list = [1, 5, 10, 50, 100]
    config_top_k = cfg.get("top_k", 100)
    final_top_k = max(config_top_k, max(metric_k_list))
    
    # 중간 저장소 경로 설정 (Hydra의 output_dir 내부에 저장)
    output_dir = HydraConfig.get().runtime.output_dir
    temp_checkpoint_path = os.path.join(output_dir, "temp_checkpoint.json")
    
    if os.path.exists(temp_checkpoint_path):
        log.info(f"Found interrupted checkpoint! Resuming evaluation from: {temp_checkpoint_path}")
        with open(temp_checkpoint_path, 'r', encoding='utf-8') as f:
            ckpt = json.load(f)
        processed_qids = set(ckpt['processed_qids'])
        passage_metrics = ckpt['passage_metrics']
        song_metrics = ckpt['song_metrics']
        detailed_rows = ckpt['detailed_rows']
        log.info(f"Loaded {len(processed_qids)} previously processed queries.")
    else:
        processed_qids = set()
        passage_metrics = {f'R@{k}': [] for k in metric_k_list}
        passage_metrics['MRR'] = []
        song_metrics = {f'R@{k}': [] for k in metric_k_list}
        song_metrics['MRR'] = []
        detailed_rows = []

    log.info(f"[Search] Running Evaluation...")

    # ---------------------------------------------------------
    # 4. 평가 루프 (with 중간 저장 로직)
    # ---------------------------------------------------------
    for query_item in tqdm(queries_data, ascii=True):
        qid = query_item['query_id']
        
        # 이미 처리된 쿼리는 스킵 (중간 저장에서 로드된 경우)
        if qid not in qrels_map or qid in processed_qids: 
            continue
            
        gold_pids_set = qrels_map[qid]
        gold_sid = query_item['song_id']
        gold_sids_set = {gold_sid}
        
        query_text = query_item.get('text')
        difficulty = query_item.get('difficulty', 'N/A')

        # [검색 및 지표 계산]
        search_results = model.search(query_text, top_k=final_top_k)
        
        pred_pids = [r['passage_id'] for r in search_results]
        p_scores = calculate_metrics_generic(pred_pids, gold_pids_set, metric_k_list)
        # JSON serialization safety
        for k, v in p_scores.items(): passage_metrics[k].append(float(v))
            
        pred_sids = [pid2sid.get(pid, "unknown") for pid in pred_pids]
        s_scores = calculate_metrics_generic(pred_sids, gold_sids_set, metric_k_list)
        # JSON serialization safety
        for k, v in s_scores.items(): song_metrics[k].append(float(v))

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

        # 체크포인트 업데이트
        processed_qids.add(qid)
        with open(temp_checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump({
                "processed_qids": list(processed_qids),
                "passage_metrics": passage_metrics,
                "song_metrics": song_metrics,
                "detailed_rows": detailed_rows
            }, f, ensure_ascii=False)

    # 5. 결과 저장 (JSON & TSV)
    final_p_metrics = {k: float(np.mean(v)) for k, v in passage_metrics.items()}
    final_s_metrics = {k: float(np.mean(v)) for k, v in song_metrics.items()}
    
    # user_id 기록
    user_label = str(cfg.user_id) if cfg.user_id is not None else "Merged"

    log.info("="*60)
    log.info(f"[Final Result] ({cfg.model.name}) - User: {user_label}")
    log.info("="*60)
    log.info(f"{'Metric':<10} | {'Passage Level':<15} | {'Song Level':<15}")
    log.info("-" * 45)
    
    for k in ['R@1', 'R@5', 'R@10', 'R@50', 'R@100', 'MRR']:
        p_val = final_p_metrics.get(k, 0.0)
        s_val = final_s_metrics.get(k, 0.0)
        log.info(f"{k:<10} | {p_val:.4f}          | {s_val:.4f}")
        
    log.info("="*60)

    # =========================================================
    # Extract params (supports both OmegaConf and dict)
    # =========================================================
    if isinstance(cfg.model, (DictConfig, ListConfig)):
        # 1. OmegaConf 객체면 -> dict로 변환
        model_conf_dict = OmegaConf.to_container(cfg.model, resolve=True)
    else:
        # 2. 이미 dict라면 -> 그대로 사용
        model_conf_dict = cfg.model

    # params가 없으면 빈 딕셔너리 {} 반환
    model_params = model_conf_dict.get("params", {})
    # =========================================================

    # JSON 저장 (user_label 사용)
    output_json = {
        "meta": {
            "model": cfg.model.name,
            "params": model_params,
            "data_user": user_label,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "metrics": {
            "passage_level": final_p_metrics,
            "song_level": final_s_metrics
        }
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding='utf-8') as f:
        json.dump(output_json, f, indent=4, ensure_ascii=False)

    if detailed_rows:
        tsv_path = os.path.join(output_dir, "prediction_results.tsv")
        fieldnames = ["query_id", "query_text", "predicted_passage_id", "difficulty", "best_gold_rank", "is_passage_correct", "is_song_correct", "predicted_passage_text"]
        with open(tsv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(detailed_rows)
        log.info(f"[Saved Details] {tsv_path}")

if __name__ == "__main__":
    main()