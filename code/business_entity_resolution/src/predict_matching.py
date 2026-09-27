"""
Applies a trained matching model (train_matching_model.py's output) to
test's candidate_pairs.tsv, producing matching_results.tsv in the exact
submission format: one row per test Source-1 entity (every one of them,
including those blocking found zero candidates for), matched_entity_ids
comma-joined with no duplicates, empty string when no match survives.

Run with plain `python`, NOT under `python -m cudf.pandas`.
"""
import argparse
import os
import pickle
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import pandas as pd

import feature_engineering as fe
import pair_table as pt
from consolidation import consolidate
from io_utils import read_tsv

PREDICT_CHUNK = 5_000_000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True, help="Phase A output (test_*_clean.parquet)")
    ap.add_argument("--candidates-dir", required=True, help="Phase B output (test's candidate_pairs.tsv)")
    ap.add_argument("--model", required=True, help="path to the pickle saved by train_matching_model.py")
    ap.add_argument("--output", required=True, help="path to write matching_results.tsv")
    ap.add_argument("--workers", type=int, default=0, help="feature processes (0 = all cores)")
    args = ap.parse_args()

    log(f"loading model from {args.model}...")
    with open(args.model, "rb") as f:
        saved = pickle.load(f)
    clf, threshold, feature_cols = saved["model"], saved["threshold"], saved["feature_cols"]
    use_emb = "embedding_cosine" in feature_cols
    expected = list(fe.FEATURE_COLS) + (["embedding_cosine"] if use_emb else [])
    if feature_cols != expected:
        raise ValueError(f"model features {feature_cols} != this code's features {expected}")
    log(f"features: {feature_cols}, threshold={threshold}")

    log("loading processed test tables...")
    s1, other = pt.load_tables(args.processed_dir, "test")
    s1_index, other_index = pd.Index(s1["entity_id"]), pd.Index(other["entity_id"])
    log(f"s1={len(s1):,} other(S2+S3)={len(other):,}")

    log("loading candidate_pairs.tsv...")
    candidate_pairs = read_tsv(os.path.join(args.candidates_dir, "candidate_pairs.tsv"))
    all_s1_ids = candidate_pairs["source1_entity_id"].to_numpy()
    if len(all_s1_ids) != len(s1) or set(all_s1_ids) != set(s1["entity_id"]):
        raise ValueError("candidate_pairs.tsv does not list every test S1 entity exactly once")
    a_pos, c_pos = pt.parse_id_lists(candidate_pairs, "source1_entity_id", "candidate_entity_ids",
                                     s1_index, other_index, "candidate")
    del candidate_pairs
    log(f"scoring {len(a_pos):,} candidate pairs across {len(all_s1_ids):,} S1 entities...")

    t0 = time.time()
    X = fe.compute_features(a_pos, c_pos, s1, other, workers=args.workers or None, log=log,
                            n_extra=int(use_emb))
    log(f"string features done in {time.time() - t0:.1f}s")
    if use_emb:
        log("computing embedding_cosine feature...")
        import embedding_blocking as eb
        X[:, -1] = fe.embedding_cosine(a_pos, c_pos, s1, other, eb.load_model(), log=log)

    score = np.empty(len(a_pos), dtype=np.float64)
    for s in range(0, len(a_pos), PREDICT_CHUNK):
        score[s:s + PREDICT_CHUNK] = clf.predict_proba(X[s:s + PREDICT_CHUNK])[:, 1]
    del X

    log("consolidating predictions (per-record argmax, gated by threshold)...")
    pred = consolidate(a_pos, c_pos, score, threshold)
    pred_df = pd.DataFrame({
        "anchor": a_pos[pred],
        "cid": other["entity_id"].to_numpy(dtype=object)[c_pos[pred]],
    })
    joined = pred_df.sort_values(["anchor", "cid"]).groupby("anchor", sort=False)["cid"].agg(",".join)
    matched = np.full(len(s1), "", dtype=object)
    matched[joined.index.to_numpy()] = joined.to_numpy()

    out_df = pd.DataFrame({
        "source1_entity_id": all_s1_ids,
        "matched_entity_ids": matched[s1_index.get_indexer(all_s1_ids)],
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_df.to_csv(args.output, sep="\t", index=False)
    n_with = int((out_df["matched_entity_ids"] != "").sum())
    log(f"wrote {args.output} ({len(out_df):,} rows, {n_with:,} with >=1 match, "
        f"{int(pred.sum()):,} matched pairs)")


if __name__ == "__main__":
    main()
