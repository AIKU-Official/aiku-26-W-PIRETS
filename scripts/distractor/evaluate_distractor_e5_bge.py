import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from importlib.machinery import ModuleSpec

from src.dataloader import load_tsv
from src.metrics import calculate_metrics_generic
from src.text_utils import normalize_text

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _install_torchvision_stub() -> None:
    if "torchvision" in sys.modules:
        return

    tv = types.ModuleType("torchvision")
    tv.__spec__ = ModuleSpec("torchvision", loader=None)
    trans = types.ModuleType("torchvision.transforms")
    trans.__spec__ = ModuleSpec("torchvision.transforms", loader=None)
    trans_v2 = types.ModuleType("torchvision.transforms.v2")
    trans_v2.__spec__ = ModuleSpec("torchvision.transforms.v2", loader=None)
    trans_v2_func = types.ModuleType("torchvision.transforms.v2.functional")
    trans_v2_func.__spec__ = ModuleSpec("torchvision.transforms.v2.functional", loader=None)

    class _InterpolationMode:
        NEAREST = 0
        BILINEAR = 2
        BICUBIC = 3
        LANCZOS = 1
        HAMMING = 4
        BOX = 5

    trans.InterpolationMode = _InterpolationMode
    trans_v2.functional = trans_v2_func
    tv.transforms = trans

    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = trans
    sys.modules["torchvision.transforms.v2"] = trans_v2
    sys.modules["torchvision.transforms.v2.functional"] = trans_v2_func


_install_torchvision_stub()
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing required package 'transformers'.\n"
        f"Current python: {sys.executable}\n"
        "Install with:\n"
        f"  {sys.executable} -m pip install \"transformers>=4.30.0\""
    ) from e


logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
_CHUNK_RE = re.compile(r"^chunk_(\d{8})_(\d{8})\.pt$")


def _parse_distractor_counts(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("No distractor counts provided.")
    if any(v < 0 for v in vals):
        raise ValueError("Distractor counts must be non-negative.")
    seen = set()
    ordered_unique: List[int] = []
    for v in vals:
        if v not in seen:
            ordered_unique.append(v)
            seen.add(v)
    return ordered_unique


def _resolve_distractor_corpus_path(data_dir: Path) -> Path:
    preferred = [
        data_dir / "distractor_corpus.tsv",
        data_dir / "distractor_corpus_cleaned.tsv",
    ]
    for path in preferred:
        if path.exists():
            return path

    candidates = sorted(data_dir.glob("distractor*corpus*.tsv"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find distractor corpus TSV in {data_dir}. "
        "Expected one of: distractor_corpus.tsv, distractor_corpus_cleaned.tsv, distractor*corpus*.tsv"
    )


def _write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _write_heartbeat(
    heartbeat_path: Optional[Path],
    stage: str,
    next_index: int,
    total_texts: int,
) -> None:
    if heartbeat_path is None:
        return
    payload: Dict[str, object] = {
        "stage": stage,
        "next_index": next_index,
        "total_texts": total_texts,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    _write_json_atomic(heartbeat_path, payload)


def _collect_contiguous_chunks(encode_dir: Path, total_texts: int) -> List[Tuple[int, int, Path]]:
    parsed: List[Tuple[int, int, Path]] = []
    for path in encode_dir.glob("chunk_*.pt"):
        m = _CHUNK_RE.match(path.name)
        if not m:
            continue
        start = int(m.group(1))
        end = int(m.group(2))
        if start < 0 or end <= start:
            continue
        if start >= total_texts or end > total_texts:
            continue
        parsed.append((start, end, path))
    parsed.sort(key=lambda x: (x[0], x[1]))

    contiguous: List[Tuple[int, int, Path]] = []
    expected = 0
    for start, end, path in parsed:
        if start != expected:
            break
        contiguous.append((start, end, path))
        expected = end
        if expected == total_texts:
            break
    return contiguous


def _load_chunk_tensor(chunks: Sequence[Tuple[int, int, Path]], total_texts: int) -> torch.Tensor:
    if total_texts == 0:
        return torch.empty((0, 0), dtype=torch.float32)
    if not chunks or chunks[-1][1] != total_texts:
        raise RuntimeError(
            f"Incomplete checkpoint. expected={total_texts}, got={(chunks[-1][1] if chunks else 0)}"
        )
    parts = [torch.load(path, map_location="cpu") for _, _, path in chunks]
    return torch.cat(parts, dim=0)


def _load_vectors_from_checkpoint(
    checkpoint_root: Path,
    checkpoint_key: str,
    total_texts: int,
    expected_model_name: str,
    expected_max_length: int,
) -> torch.Tensor:
    encode_dir = checkpoint_root / checkpoint_key
    meta_path = encode_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing checkpoint meta: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    saved_total = int(meta.get("total_texts", -1))
    if saved_total != total_texts:
        raise RuntimeError(
            f"Checkpoint total mismatch for {checkpoint_key}: expected={total_texts}, saved={saved_total}"
        )

    saved_model = str(meta.get("model_name", ""))
    if saved_model != expected_model_name:
        logger.warning(
            "Checkpoint model_name mismatch for %s: expected=%s saved=%s",
            checkpoint_key,
            expected_model_name,
            saved_model,
        )

    saved_max_len = int(meta.get("max_length", -1))
    if saved_max_len != expected_max_length:
        logger.warning(
            "Checkpoint max_length mismatch for %s: expected=%d saved=%d",
            checkpoint_key,
            expected_max_length,
            saved_max_len,
        )

    chunks = _collect_contiguous_chunks(encode_dir, total_texts)
    vecs = _load_chunk_tensor(chunks, total_texts)
    logger.info("Loaded %s vectors from checkpoint: %s", checkpoint_key, encode_dir)
    return vecs


def _build_passage_qrels(rows: List[Dict[str, str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for r in rows:
        out.setdefault(str(r["query_id"]), []).append(str(r["passage_id"]))
    return out


def _write_ids(path: Path, header: str, ids: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{header}\n")
        for x in ids:
            f.write(f"{x}\n")


def _write_scored_passages(path: Path, scored: Dict[str, List[Tuple[str, float]]], top_k: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("query_id\trank\tpassage_id\tscore\n")
        for qid, pairs in scored.items():
            for idx, (pid, score) in enumerate(pairs[:top_k], start=1):
                f.write(f"{qid}\t{idx}\t{pid}\t{score:.8f}\n")


def _write_song_predictions(path: Path, ranked: Dict[str, List[str]], top_k: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("query_id\trank\tsong_id\n")
        for qid, song_ids in ranked.items():
            for idx, sid in enumerate(song_ids[:top_k], start=1):
                f.write(f"{qid}\t{idx}\t{sid}\n")


def _aggregate_metrics(
    predictions: Dict[str, List[str]],
    gold_sets: Dict[str, set],
    ks: Sequence[int],
) -> Dict[str, float]:
    values: Dict[str, List[float]] = {f"R@{k}": [] for k in ks}
    values["MRR"] = []

    for qid, gold_set in gold_sets.items():
        if not gold_set:
            continue
        scores = calculate_metrics_generic(predictions.get(qid, []), gold_set, list(ks))
        for key, val in scores.items():
            values[key].append(float(val))

    return {k: ((sum(v) / len(v)) if v else 0.0) for k, v in values.items()}


def _build_song_rankings(
    passage_rankings: Dict[str, List[Tuple[str, float]]],
    passage_to_song: Dict[str, str],
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for qid, pairs in passage_rankings.items():
        best: Dict[str, float] = {}
        for pid, score in pairs:
            sid = passage_to_song.get(pid)
            if sid is None:
                continue
            prev = best.get(sid)
            if prev is None or score > prev:
                best[sid] = score
        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
        out[qid] = [sid for sid, _ in ranked]
    return out


def _build_reranked_passage_rankings(
    query_ids: Sequence[str],
    query_texts: Sequence[str],
    doc_ids: Sequence[str],
    doc_texts: Sequence[str],
    dense_scores: torch.Tensor,
    reranker_tokenizer: AutoTokenizer,
    reranker_model: AutoModelForSequenceClassification,
    device: torch.device,
    rerank_top_n: int,
    rerank_batch_size: int,
    show_progress: bool,
) -> Dict[str, List[Tuple[str, float]]]:
    doc_text_by_id = {pid: txt for pid, txt in zip(doc_ids, doc_texts)}
    out: Dict[str, List[Tuple[str, float]]] = {}

    iterator = range(len(query_ids))
    if show_progress:
        iterator = tqdm(iterator, total=len(query_ids), desc="BGE reranking", unit="query")

    rerank_top_n = max(1, rerank_top_n)

    with torch.no_grad():
        for qi in iterator:
            qid = query_ids[qi]
            qtxt = query_texts[qi]
            row = dense_scores[qi]
            order = torch.argsort(row, descending=True)
            order_list = order.tolist()

            top_n_actual = min(rerank_top_n, len(order_list))
            top_indices = order_list[:top_n_actual]
            top_pids = [doc_ids[i] for i in top_indices]

            rerank_scores: List[float] = []
            for i in range(0, len(top_pids), rerank_batch_size):
                batch_pids = top_pids[i : i + rerank_batch_size]
                batch_queries = [qtxt] * len(batch_pids)
                batch_docs = [doc_text_by_id[pid] for pid in batch_pids]
                encoded = reranker_tokenizer(
                    batch_queries,
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                logits = reranker_model(**encoded).logits
                if logits.ndim == 2 and logits.shape[1] == 1:
                    logits = logits[:, 0]
                scores = logits.float().cpu().tolist()
                rerank_scores.extend(float(s) for s in scores)

            reranked_top = sorted(
                zip(top_pids, rerank_scores),
                key=lambda x: x[1],
                reverse=True,
            )

            dense_rest = [(doc_ids[i], float(row[i])) for i in order_list[top_n_actual:]]
            out[qid] = reranked_top + dense_rest

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/eval_dataset")
    parser.add_argument("--model-name", default="intfloat/multilingual-e5-base")
    parser.add_argument("--reranker-name", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--rerank-top-n", type=int, default=20)
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k-save", type=int, default=100)
    parser.add_argument("--distractor-counts", default="0,1000,10000,20000,30000,43234")
    parser.add_argument("--k-values", default="1,5,10,50,100")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--cuda-device-index", type=int, default=0)
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--encoding-checkpoint-dir", default="outputs/e5_encoding_checkpoints")
    parser.add_argument("--resume-encoding", action="store_true")  # monitor.py compatibility
    parser.add_argument("--heartbeat-file", default="")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    checkpoint_root = Path(args.encoding_checkpoint_dir)
    heartbeat_path = Path(args.heartbeat_file) if args.heartbeat_file else None

    corpus_rows = load_tsv(str(data_dir / "corpus.tsv"))
    distractor_path = _resolve_distractor_corpus_path(data_dir)
    distractor_rows = load_tsv(str(distractor_path))
    query_rows = load_tsv(str(data_dir / "queries.tsv"))
    qrels_rows = load_tsv(str(data_dir / "qrels.tsv"))

    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    distractor_counts_requested = _parse_distractor_counts(args.distractor_counts)

    rng = random.Random(args.seed)
    distractor_rows = list(distractor_rows)
    rng.shuffle(distractor_rows)

    query_ids = [str(r["query_id"]) for r in query_rows]
    query_texts = [normalize_text(str(r["text"])) for r in query_rows]
    gold_song_ids = {str(r["query_id"]): str(r["song_id"]) for r in query_rows}
    qrels = _build_passage_qrels(qrels_rows)

    base_doc_ids = [str(r["passage_id"]) for r in corpus_rows]
    base_doc_texts = [normalize_text(str(r["text"])) for r in corpus_rows]
    base_pid_to_sid = {str(r["passage_id"]): str(r["song_id"]) for r in corpus_rows}

    dist_doc_ids = [str(r["passage_id"]) for r in distractor_rows]
    dist_doc_texts = [normalize_text(str(r["text"])) for r in distractor_rows]
    dist_pid_to_sid = {str(r["passage_id"]): str(r["song_id"]) for r in distractor_rows}

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        if args.cuda_device_index < 0 or args.cuda_device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"--cuda-device-index {args.cuda_device_index} is out of range. "
                f"Available CUDA devices: 0..{torch.cuda.device_count() - 1}"
            )
        device = torch.device(f"cuda:{args.cuda_device_index}")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            if args.cuda_device_index < 0 or args.cuda_device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"--cuda-device-index {args.cuda_device_index} is out of range. "
                    f"Available CUDA devices: 0..{torch.cuda.device_count() - 1}"
                )
            device = torch.device(f"cuda:{args.cuda_device_index}")
        else:
            device = torch.device("cpu")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    _write_heartbeat(heartbeat_path, "encoding:load_cached_vectors", 0, 3)
    query_vecs = _load_vectors_from_checkpoint(
        checkpoint_root,
        "queries",
        total_texts=len(query_ids),
        expected_model_name=args.model_name,
        expected_max_length=args.max_length,
    )
    _write_heartbeat(heartbeat_path, "encoding:load_cached_vectors", 1, 3)

    base_vecs = _load_vectors_from_checkpoint(
        checkpoint_root,
        "base_corpus",
        total_texts=len(base_doc_ids),
        expected_model_name=args.model_name,
        expected_max_length=args.max_length,
    )
    _write_heartbeat(heartbeat_path, "encoding:load_cached_vectors", 2, 3)

    dist_vecs = _load_vectors_from_checkpoint(
        checkpoint_root,
        "distractor_corpus",
        total_texts=len(dist_doc_ids),
        expected_model_name=args.model_name,
        expected_max_length=args.max_length,
    )
    _write_heartbeat(heartbeat_path, "encoding_done", 0, 0)

    logger.info("Loading BGE reranker model: %s (device=%s)", args.reranker_name, device)
    reranker_tokenizer = AutoTokenizer.from_pretrained(args.reranker_name, local_files_only=True)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(
        args.reranker_name,
        local_files_only=True,
    ).to(device)
    reranker_model.eval()

    now = datetime.now().strftime("%y%m%d_%H%M%S")
    out_root = Path("outputs") / f"e5_bge_eval_{now}"
    out_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "model_name": args.model_name,
        "reranker_name": args.reranker_name,
        "rerank_top_n": args.rerank_top_n,
        "seed": args.seed,
        "k_values": k_values,
        "distractor_corpus_path": str(distractor_path),
        "distractor_counts_requested": distractor_counts_requested,
        "base_corpus_size": len(base_doc_ids),
        "distractor_pool_size": len(dist_doc_ids),
        "results": [],
    }

    query_vecs_f = query_vecs.float().to(device)

    for requested_count in distractor_counts_requested:
        add_count = min(requested_count, len(dist_doc_ids))
        if add_count < requested_count:
            logger.warning(
                "Requested distractor count %d exceeds pool size %d; using %d",
                requested_count,
                len(dist_doc_ids),
                add_count,
            )

        doc_ids = base_doc_ids + dist_doc_ids[:add_count]
        doc_texts = base_doc_texts + dist_doc_texts[:add_count]
        doc_vecs = torch.cat([base_vecs, dist_vecs[:add_count]], dim=0).float().to(device)

        pid_to_sid = dict(base_pid_to_sid)
        pid_to_sid.update({k: dist_pid_to_sid[k] for k in dist_doc_ids[:add_count]})

        dense_scores = (query_vecs_f @ doc_vecs.T).cpu()
        del doc_vecs

        passage_rankings = _build_reranked_passage_rankings(
            query_ids=query_ids,
            query_texts=query_texts,
            doc_ids=doc_ids,
            doc_texts=doc_texts,
            dense_scores=dense_scores,
            reranker_tokenizer=reranker_tokenizer,
            reranker_model=reranker_model,
            device=device,
            rerank_top_n=args.rerank_top_n,
            rerank_batch_size=args.rerank_batch_size,
            show_progress=not args.no_tqdm,
        )

        passage_pred = {qid: [pid for pid, _ in pairs] for qid, pairs in passage_rankings.items()}
        song_pred = _build_song_rankings(passage_rankings, pid_to_sid)

        eval_qids = [qid for qid in query_ids if qid in qrels]
        passage_gold_sets = {qid: set(qrels[qid]) for qid in eval_qids}
        song_gold_sets = {qid: {gold_song_ids[qid]} for qid in eval_qids if qid in gold_song_ids}

        passage_metrics = _aggregate_metrics(passage_pred, passage_gold_sets, k_values)
        song_metrics = _aggregate_metrics(song_pred, song_gold_sets, k_values)

        run_dir = out_root / f"run_{add_count:05d}d"
        run_dir.mkdir(parents=True, exist_ok=True)

        _write_ids(run_dir / "query_ids.tsv", "query_id", query_ids)
        _write_ids(run_dir / "doc_ids.tsv", "passage_id", doc_ids)
        _write_scored_passages(run_dir / "passage_predictions.tsv", passage_rankings, args.top_k_save)
        _write_song_predictions(run_dir / "song_predictions.tsv", song_pred, args.top_k_save)

        run_metrics = {
            "distractor_count_requested": requested_count,
            "num_distractors_added": add_count,
            "corpus_size": len(doc_ids),
            "passage_level": passage_metrics,
            "song_level": song_metrics,
        }
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(run_metrics, f, ensure_ascii=False, indent=2)

        summary["results"].append(run_metrics)
        logger.info(
            "distractors=%d corpus=%d passage_R@10=%.4f song_R@10=%.4f",
            add_count,
            len(doc_ids),
            passage_metrics["R@10"],
            song_metrics["R@10"],
        )

    with (out_root / "metrics_by_distractor_count.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    header = [
        "distractor_count_requested",
        "num_distractors_added",
        "corpus_size",
        "passage_R@1",
        "passage_R@5",
        "passage_R@10",
        "passage_R@50",
        "passage_R@100",
        "passage_MRR",
        "song_R@1",
        "song_R@5",
        "song_R@10",
        "song_R@50",
        "song_R@100",
        "song_MRR",
    ]
    with (out_root / "metrics_by_distractor_count.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for row in summary["results"]:
            writer.writerow(
                {
                    "distractor_count_requested": row["distractor_count_requested"],
                    "num_distractors_added": row["num_distractors_added"],
                    "corpus_size": row["corpus_size"],
                    "passage_R@1": row["passage_level"]["R@1"],
                    "passage_R@5": row["passage_level"]["R@5"],
                    "passage_R@10": row["passage_level"]["R@10"],
                    "passage_R@50": row["passage_level"]["R@50"],
                    "passage_R@100": row["passage_level"]["R@100"],
                    "passage_MRR": row["passage_level"]["MRR"],
                    "song_R@1": row["song_level"]["R@1"],
                    "song_R@5": row["song_level"]["R@5"],
                    "song_R@10": row["song_level"]["R@10"],
                    "song_R@50": row["song_level"]["R@50"],
                    "song_R@100": row["song_level"]["R@100"],
                    "song_MRR": row["song_level"]["MRR"],
                }
            )

    logger.info("Saved E5+BGE distractor outputs to %s", out_root)


if __name__ == "__main__":
    main()
