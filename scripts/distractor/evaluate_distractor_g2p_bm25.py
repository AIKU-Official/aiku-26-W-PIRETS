import csv
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.dataloader import load_tsv
from src.g2p import G2PConverter
from src.metrics import calculate_metrics_generic
from src.models.bm25 import BM25Retriever
from src.text_utils import normalize_text


logger = logging.getLogger(__name__)


def _build_passage_qrels(
    qrels_rows: List[Dict[str, str]],
    query_col: str,
    passage_col: str,
) -> Dict[str, List[str]]:
    qrels: Dict[str, List[str]] = {}
    for row in qrels_rows:
        qid = str(row[query_col])
        pid = str(row[passage_col])
        qrels.setdefault(qid, []).append(pid)
    return qrels


def _build_song_rankings(
    scored_passages: Dict[str, List[Tuple[str, float]]],
    passage_to_song: Dict[str, str],
) -> Dict[str, List[str]]:
    song_rankings: Dict[str, List[str]] = {}
    for qid, pairs in scored_passages.items():
        song_best_score: Dict[str, float] = {}
        for passage_id, score in pairs:
            song_id = passage_to_song.get(passage_id)
            if song_id is None:
                continue
            prev = song_best_score.get(song_id)
            if prev is None or score > prev:
                song_best_score[song_id] = score
        ranked_songs = sorted(song_best_score.items(), key=lambda x: x[1], reverse=True)
        song_rankings[qid] = [song_id for song_id, _ in ranked_songs]
    return song_rankings


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


def _write_scored_passages(
    path: Path,
    scored: Dict[str, List[Tuple[str, float]]],
    top_k: int,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("query_id\trank\tpassage_id\tscore\n")
        for qid, pairs in scored.items():
            for idx, (passage_id, score) in enumerate(pairs[:top_k], start=1):
                f.write(f"{qid}\t{idx}\t{passage_id}\t{score:.8f}\n")


def _write_song_predictions(path: Path, song_rankings: Dict[str, List[str]], top_k: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("query_id\trank\tsong_id\n")
        for qid, song_ids in song_rankings.items():
            for idx, song_id in enumerate(song_ids[:top_k], start=1):
                f.write(f"{qid}\t{idx}\t{song_id}\n")


def _evaluate_once(
    cfg: DictConfig,
    corpus_rows: List[Dict[str, str]],
    query_rows: List[Dict[str, str]],
    qrels: Dict[str, List[str]],
    ks: Sequence[int],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, List[Tuple[str, float]]], Dict[str, List[str]]]:
    model = BM25Retriever(k1=float(cfg.model.k1), b=float(cfg.model.b))

    docs = [
        (str(r[cfg.data.doc_id_column]), str(r["g2p_text"]))
        for r in corpus_rows
    ]
    queries = {
        str(r[cfg.data.query_id_column]): str(r["g2p_text"])
        for r in query_rows
    }
    gold_song_ids = {str(r[cfg.data.query_id_column]): str(r["song_id"]) for r in query_rows}
    passage_to_song = {str(r[cfg.data.doc_id_column]): str(r["song_id"]) for r in corpus_rows}

    model.fit(docs)
    scored_passages = model.batch_retrieve(queries, top_k=len(corpus_rows))
    predicted_passages = {qid: [doc_id for doc_id, _ in pairs] for qid, pairs in scored_passages.items()}
    predicted_songs = _build_song_rankings(scored_passages, passage_to_song)

    eval_qids = [qid for qid in queries.keys() if qid in qrels]
    passage_gold_sets = {qid: set(qrels[qid]) for qid in eval_qids}
    song_gold_sets = {qid: {gold_song_ids[qid]} for qid in eval_qids if qid in gold_song_ids}

    passage_metrics = _aggregate_metrics(predicted_passages, passage_gold_sets, ks)
    song_metrics = _aggregate_metrics(predicted_songs, song_gold_sets, ks)

    return passage_metrics, song_metrics, scored_passages, predicted_songs


@hydra.main(config_path="conf", config_name="eval_distractor_g2p_bm25_config")
def main(cfg: DictConfig) -> None:
    logger.info("Eval distractor G2P+BM25 config:\n%s", OmegaConf.to_yaml(cfg))

    g2p_converter = G2PConverter()
    logger.info("Using G2P converter from src.g2p.G2PConverter")

    # Speed up expensive G2P by caching word-level conversions.
    g2p_word_cache: Dict[str, str] = {}

    def convert_text_cached(text: str) -> str:
        tokens = str(text).split()
        if not tokens:
            return ""
        out: List[str] = []
        for token in tokens:
            cached = g2p_word_cache.get(token)
            if cached is None:
                cached = str(g2p_converter(token))
                g2p_word_cache[token] = cached
            out.append(cached)
        return " ".join(out)

    corpus_rows = load_tsv(to_absolute_path(cfg.data.corpus_path))
    query_rows = load_tsv(to_absolute_path(cfg.data.queries_path))
    qrels_rows = load_tsv(to_absolute_path(cfg.data.qrels_path))
    distractor_rows = load_tsv(to_absolute_path(cfg.distractor.path))

    logger.info("Converting queries/corpus/distractors to G2P once (cached)")
    for i, row in enumerate(query_rows, start=1):
        row["g2p_text"] = convert_text_cached(normalize_text(str(row[cfg.data.query_text_column])))
        if i % 100 == 0 or i == len(query_rows):
            logger.info("G2P converted queries: %d/%d", i, len(query_rows))
    for i, row in enumerate(corpus_rows, start=1):
        row["g2p_text"] = convert_text_cached(normalize_text(str(row[cfg.data.doc_text_column])))
        if i % 100 == 0 or i == len(corpus_rows):
            logger.info("G2P converted corpus: %d/%d", i, len(corpus_rows))
    for i, row in enumerate(distractor_rows, start=1):
        row["g2p_text"] = convert_text_cached(normalize_text(str(row[cfg.data.doc_text_column])))
        if i % 100 == 0 or i == len(distractor_rows):
            logger.info("G2P converted distractors: %d/%d", i, len(distractor_rows))

    qrels = _build_passage_qrels(
        qrels_rows,
        cfg.data.qrels_query_id_column,
        cfg.data.qrels_doc_id_column,
    )

    rng = random.Random(int(cfg.seed))
    shuffled_distractors = list(distractor_rows)
    rng.shuffle(shuffled_distractors)

    ks: Sequence[int] = [int(k) for k in cfg.eval.k_values]
    save_top_k = int(cfg.distractor.save_top_k_predictions)

    summary: Dict[str, object] = {
        "base_corpus_size": len(corpus_rows),
        "distractor_pool_size": len(shuffled_distractors),
        "seed": int(cfg.seed),
        "g2p_variant": "src.g2p.G2PConverter",
        "results": [],
    }

    for pct in [int(p) for p in cfg.distractor.percentages]:
        add_count = int(len(shuffled_distractors) * (pct / 100.0))
        merged_corpus = corpus_rows + shuffled_distractors[:add_count]

        logger.info(
            "Running percentage=%d%% with base=%d + distractor=%d -> total=%d",
            pct,
            len(corpus_rows),
            add_count,
            len(merged_corpus),
        )

        passage_metrics, song_metrics, scored_passages, predicted_songs = _evaluate_once(
            cfg,
            merged_corpus,
            query_rows,
            qrels,
            ks,
        )

        run_dir = Path(f"run_{pct:03d}pct")
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "percentage": pct,
                    "num_distractors_added": add_count,
                    "corpus_size": len(merged_corpus),
                    "passage_level": passage_metrics,
                    "song_level": song_metrics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        _write_scored_passages(run_dir / "passage_predictions.tsv", scored_passages, save_top_k)
        _write_song_predictions(run_dir / "song_predictions.tsv", predicted_songs, save_top_k)

        summary["results"].append(
            {
                "percentage": pct,
                "num_distractors_added": add_count,
                "corpus_size": len(merged_corpus),
                "passage_level": passage_metrics,
                "song_level": song_metrics,
            }
        )

    with Path("metrics_by_ratio.json").open("w", encoding="utf-8") as f:
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
    with Path("metrics_by_ratio.tsv").open("w", encoding="utf-8", newline="") as f:
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

    logger.info("Saved G2P+BM25 distractor pipeline outputs to %s", Path.cwd())


if __name__ == "__main__":
    main()
