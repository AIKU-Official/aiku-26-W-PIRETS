import argparse
import csv
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from src.dataloader import load_tsv
from src.g2p import G2PConverter
from src.metrics import calculate_metrics_generic
from src.text_utils import normalize_text


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _parse_percentages(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return sorted(vals)


def _encode_texts(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    model.eval()
    all_vecs: List[torch.Tensor] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
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
            vecs = torch.nn.functional.normalize(cls, p=2, dim=1)
            all_vecs.append(vecs.cpu())

    return torch.cat(all_vecs, dim=0)


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


def _convert_with_cache(converter: G2PConverter, texts: Sequence[str], log_prefix: str) -> List[str]:
    cache: Dict[str, str] = {}
    converted: List[str] = []
    n = len(texts)

    for i, text in enumerate(texts, start=1):
        tokens = str(text).split()
        out_tokens: List[str] = []
        for token in tokens:
            val = cache.get(token)
            if val is None:
                val = str(converter(token))
                cache[token] = val
            out_tokens.append(val)
        converted.append(" ".join(out_tokens))

        if i % 100 == 0 or i == n:
            logger.info("G2P converted %s: %d/%d", log_prefix, i, n)

    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/eval_dataset")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k-save", type=int, default=100)
    parser.add_argument("--percentages", default="0,10,20,30,40,50,60,70,80,90,100")
    parser.add_argument("--k-values", default="1,5,10,50,100")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    corpus_rows = load_tsv(str(data_dir / "corpus.tsv"))
    distractor_rows = load_tsv(str(data_dir / "distractor_corpus_cleaned.tsv"))
    query_rows = load_tsv(str(data_dir / "queries.tsv"))
    qrels_rows = load_tsv(str(data_dir / "qrels.tsv"))

    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    percentages = _parse_percentages(args.percentages)

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

    logger.info("Building G2P converter (src.g2p.G2PConverter)")
    g2p_converter = G2PConverter()

    logger.info("Converting query/base/distractor texts to G2P (cached)")
    query_texts_g2p = _convert_with_cache(g2p_converter, query_texts, "queries")
    base_doc_texts_g2p = _convert_with_cache(g2p_converter, base_doc_texts, "base corpus")
    dist_doc_texts_g2p = _convert_with_cache(g2p_converter, dist_doc_texts, "distractors")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    logger.info("Loading DPR encoder model: %s (device=%s)", args.model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    encoder = AutoModel.from_pretrained(args.model_name, local_files_only=True).to(device)
    if device.type == "cuda":
        encoder = encoder.half()

    logger.info("Encoding G2P query/base/distractor texts to vectors")
    query_vecs = _encode_texts(encoder, tokenizer, query_texts_g2p, device, args.batch_size, args.max_length)
    base_vecs = _encode_texts(encoder, tokenizer, base_doc_texts_g2p, device, args.batch_size, args.max_length)
    dist_vecs = _encode_texts(encoder, tokenizer, dist_doc_texts_g2p, device, args.batch_size, args.max_length)

    now = datetime.now().strftime("%y%m%d_%H%M%S")
    out_root = Path("outputs") / f"g2p_dpr_eval_{now}"
    out_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "model_name": args.model_name,
        "seed": args.seed,
        "k_values": k_values,
        "base_corpus_size": len(base_doc_ids),
        "distractor_pool_size": len(dist_doc_ids),
        "g2p_source": "src.g2p.G2PConverter",
        "results": [],
    }

    for pct in percentages:
        add_count = int(len(dist_doc_ids) * (pct / 100.0))
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

        run_dir = out_root / f"run_{pct:03d}pct"
        run_dir.mkdir(parents=True, exist_ok=True)

        torch.save(query_vecs, run_dir / "query_vectors.pt")
        torch.save(doc_vecs, run_dir / "doc_vectors.pt")
        _write_ids(run_dir / "query_ids.tsv", "query_id", query_ids)
        _write_ids(run_dir / "doc_ids.tsv", "passage_id", doc_ids)
        _write_scored_passages(run_dir / "passage_predictions.tsv", passage_rankings, args.top_k_save)
        _write_song_predictions(run_dir / "song_predictions.tsv", song_pred, args.top_k_save)

        run_metrics = {
            "percentage": pct,
            "num_distractors_added": add_count,
            "corpus_size": len(doc_ids),
            "passage_level": passage_metrics,
            "song_level": song_metrics,
        }
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(run_metrics, f, ensure_ascii=False, indent=2)

        summary["results"].append(run_metrics)
        logger.info(
            "pct=%d corpus=%d passage_R@10=%.4f song_R@10=%.4f",
            pct,
            len(doc_ids),
            passage_metrics["R@10"],
            song_metrics["R@10"],
        )

    with (out_root / "metrics_by_ratio.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    header = [
        "percentage",
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
    with (out_root / "metrics_by_ratio.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for row in summary["results"]:
            writer.writerow(
                {
                    "percentage": row["percentage"],
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

    logger.info("Saved G2P+DPR distractor outputs to %s", out_root)


if __name__ == "__main__":
    main()
