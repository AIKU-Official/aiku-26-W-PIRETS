import argparse
import csv
import json
import logging
import os
import random
import re
import shutil
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
    from transformers import AutoModel, AutoTokenizer  # noqa: E402
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
        return torch.empty((0, 0))
    if not chunks or chunks[-1][1] != total_texts:
        raise RuntimeError(
            f"Incomplete encoding checkpoint. expected={total_texts}, got={(chunks[-1][1] if chunks else 0)}"
        )
    parts = [torch.load(path, map_location="cpu") for _, _, path in chunks]
    return torch.cat(parts, dim=0)


def _reset_checkpoint_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _encode_texts(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
    desc: str,
    show_progress: bool,
    checkpoint_root: Optional[Path],
    checkpoint_key: str,
    resume: bool,
    model_name: str,
    heartbeat_path: Optional[Path],
) -> torch.Tensor:
    model.eval()
    text_count = len(texts)
    if text_count == 0:
        hidden_size = int(getattr(model.config, "hidden_size", 0))
        return torch.empty((0, hidden_size))

    if checkpoint_root is None:
        all_vecs: List[torch.Tensor] = []
        total_batches = (text_count + batch_size - 1) // batch_size
        batch_iter = range(0, text_count, batch_size)
        if show_progress:
            batch_iter = tqdm(batch_iter, total=total_batches, desc=desc, unit="batch")
        _write_heartbeat(heartbeat_path, f"encoding:{checkpoint_key}", 0, text_count)
        with torch.no_grad():
            for i in batch_iter:
                batch = texts[i : i + batch_size]
                encoded = tokenizer(
                    list(batch),
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                out = model(**encoded)
                cls = out.last_hidden_state[:, 0, :]
                vecs = torch.nn.functional.normalize(cls, p=2, dim=1).cpu()
                all_vecs.append(vecs)
                _write_heartbeat(heartbeat_path, f"encoding:{checkpoint_key}", i + len(batch), text_count)
        return torch.cat(all_vecs, dim=0)

    encode_dir = checkpoint_root / checkpoint_key
    meta_path = encode_dir / "meta.json"

    if not resume:
        _reset_checkpoint_dir(encode_dir)
    else:
        encode_dir.mkdir(parents=True, exist_ok=True)
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            meta_ok = (
                int(meta.get("total_texts", -1)) == text_count
                and str(meta.get("model_name", "")) == model_name
                and int(meta.get("max_length", -1)) == max_length
            )
            if not meta_ok:
                logger.warning("Checkpoint incompatible for %s. Rebuilding from scratch.", checkpoint_key)
                _reset_checkpoint_dir(encode_dir)

    contiguous = _collect_contiguous_chunks(encode_dir, text_count)
    start_index = contiguous[-1][1] if contiguous else 0
    if start_index > 0:
        logger.info("%s resume from %d/%d texts", desc, start_index, text_count)

    total_batches = (text_count + batch_size - 1) // batch_size
    initial_batches = start_index // batch_size
    batch_iter = range(start_index, text_count, batch_size)
    if show_progress:
        batch_iter = tqdm(
            batch_iter,
            total=total_batches,
            initial=initial_batches,
            desc=desc,
            unit="batch",
        )

    _write_heartbeat(heartbeat_path, f"encoding:{checkpoint_key}", start_index, text_count)

    with torch.no_grad():
        for i in batch_iter:
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            out = model(**encoded)
            cls = out.last_hidden_state[:, 0, :]
            vecs = torch.nn.functional.normalize(cls, p=2, dim=1).cpu()
            end_index = i + len(batch)

            chunk_path = encode_dir / f"chunk_{i:08d}_{end_index:08d}.pt"
            torch.save(vecs, chunk_path)
            _write_json_atomic(
                meta_path,
                {
                    "checkpoint_key": checkpoint_key,
                    "next_index": end_index,
                    "total_texts": text_count,
                    "model_name": model_name,
                    "batch_size": batch_size,
                    "max_length": max_length,
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                },
            )
            _write_heartbeat(heartbeat_path, f"encoding:{checkpoint_key}", end_index, text_count)

    final_chunks = _collect_contiguous_chunks(encode_dir, text_count)
    return _load_chunk_tensor(final_chunks, text_count)


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


def _scores_to_rankings(
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    scores: torch.Tensor,
) -> Dict[str, List[Tuple[str, float]]]:
    out: Dict[str, List[Tuple[str, float]]] = {}
    for qi, qid in enumerate(query_ids):
        order = torch.argsort(scores[qi], descending=True)
        out[qid] = [(doc_ids[int(di)], float(scores[qi, int(di)])) for di in order]
    return out


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/eval_dataset")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k-save", type=int, default=100)
    parser.add_argument("--distractor-counts", default="0,1000,10000,20000,30000,43234")
    parser.add_argument("--k-values", default="1,5,10,50,100")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--cuda-device-index", type=int, default=0)
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--encoding-checkpoint-dir", default="outputs/dpr_encoding_checkpoints")
    parser.add_argument("--resume-encoding", action="store_true")
    parser.add_argument("--heartbeat-file", default="")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
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

    logger.info("Loading DPR encoder model: %s (device=%s)", args.model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    encoder = AutoModel.from_pretrained(args.model_name, local_files_only=True).to(device)
    if device.type == "cuda":
        encoder = encoder.half()

    checkpoint_root = Path(args.encoding_checkpoint_dir) if args.encoding_checkpoint_dir else None
    heartbeat_path = Path(args.heartbeat_file) if args.heartbeat_file else None

    logger.info("Encoding queries/base/distractors to vectors")
    query_vecs = _encode_texts(
        encoder,
        tokenizer,
        query_texts,
        device,
        args.batch_size,
        args.max_length,
        desc="Encoding queries",
        show_progress=not args.no_tqdm,
        checkpoint_root=checkpoint_root,
        checkpoint_key="queries",
        resume=args.resume_encoding,
        model_name=args.model_name,
        heartbeat_path=heartbeat_path,
    )
    base_vecs = _encode_texts(
        encoder,
        tokenizer,
        base_doc_texts,
        device,
        args.batch_size,
        args.max_length,
        desc="Encoding base corpus",
        show_progress=not args.no_tqdm,
        checkpoint_root=checkpoint_root,
        checkpoint_key="base_corpus",
        resume=args.resume_encoding,
        model_name=args.model_name,
        heartbeat_path=heartbeat_path,
    )
    dist_vecs = _encode_texts(
        encoder,
        tokenizer,
        dist_doc_texts,
        device,
        args.batch_size,
        args.max_length,
        desc="Encoding distractor corpus",
        show_progress=not args.no_tqdm,
        checkpoint_root=checkpoint_root,
        checkpoint_key="distractor_corpus",
        resume=args.resume_encoding,
        model_name=args.model_name,
        heartbeat_path=heartbeat_path,
    )
    _write_heartbeat(heartbeat_path, "encoding_done", 0, 0)

    now = datetime.now().strftime("%y%m%d_%H%M%S")
    out_root = Path("outputs") / f"dpr_eval_{now}"
    out_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "model_name": args.model_name,
        "seed": args.seed,
        "k_values": k_values,
        "distractor_corpus_path": str(distractor_path),
        "distractor_counts_requested": distractor_counts_requested,
        "base_corpus_size": len(base_doc_ids),
        "distractor_pool_size": len(dist_doc_ids),
        "results": [],
    }

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
        doc_vecs = torch.cat([base_vecs, dist_vecs[:add_count]], dim=0)

        pid_to_sid = dict(base_pid_to_sid)
        pid_to_sid.update({k: dist_pid_to_sid[k] for k in dist_doc_ids[:add_count]})

        scores = query_vecs @ doc_vecs.T
        passage_rankings = _scores_to_rankings(query_ids, doc_ids, scores)
        passage_pred = {qid: [pid for pid, _ in pairs] for qid, pairs in passage_rankings.items()}
        song_pred = _build_song_rankings(passage_rankings, pid_to_sid)

        eval_qids = [qid for qid in query_ids if qid in qrels]
        passage_gold_sets = {qid: set(qrels[qid]) for qid in eval_qids}
        song_gold_sets = {qid: {gold_song_ids[qid]} for qid in eval_qids if qid in gold_song_ids}

        passage_metrics = _aggregate_metrics(passage_pred, passage_gold_sets, k_values)
        song_metrics = _aggregate_metrics(song_pred, song_gold_sets, k_values)

        run_dir = out_root / f"run_{add_count:05d}d"
        run_dir.mkdir(parents=True, exist_ok=True)

        torch.save(query_vecs, run_dir / "query_vectors.pt")
        torch.save(doc_vecs, run_dir / "doc_vectors.pt")
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

    logger.info("Saved DPR distractor outputs to %s", out_root)


if __name__ == "__main__":
    main()
