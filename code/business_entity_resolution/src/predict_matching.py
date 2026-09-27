"""
Applies a trained matching model (train_matching_model.py's output) to
test's candidate_pairs.tsv, producing matching_results.tsv in the exact
submission format: one row per test Source-1 entity (every one of them,
including those blocking found zero candidates for), matched_entity_ids
comma-joined with no duplicates, empty string when no match survives.
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import feature_engineering as fe
from consolidation import consolidate
from io_utils import read_parquet, read_tsv
from transliteration import add_compare_name


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True, help="Phase A output (test_*_clean.parquet)")
    ap.add_argument("--candidates-dir", required=True, help="Phase B output (test's candidate_pairs.tsv)")
    ap.add_argument("--model", required=True, help="path to the pickle saved by train_matching_model.py")
    ap.add_argument("--output", required=True, help="path to write matching_results.tsv")
    args = ap.parse_args()

    log(f"loading model from {args.model}...")
    with open(args.model, "rb") as f:
        saved = pickle.load(f)
    clf, threshold, feature_cols = saved["model"], saved["threshold"], saved["feature_cols"]
    log(f"model expects features: {feature_cols}, threshold={threshold}")

    log("loading processed test data...")
    s1 = add_compare_name(read_parquet(os.path.join(args.processed_dir, "test_source1_clean.parquet")))
    s2 = add_compare_name(read_parquet(os.path.join(args.processed_dir, "test_source2_clean.parquet")))
    s3 = add_compare_name(read_parquet(os.path.join(args.processed_dir, "test_source3_clean.parquet")))
    s1_lookup = fe.build_lookup(s1)
    other_lookup = fe.build_lookup(s2)
    other_lookup.update(fe.build_lookup(s3))
    log(f"lookups built: s1={len(s1_lookup):,} other={len(other_lookup):,}")

    log("loading candidate_pairs.tsv...")
    candidate_pairs = read_tsv(os.path.join(args.candidates_dir, "candidate_pairs.tsv"))
    all_s1_ids = candidate_pairs["source1_entity_id"].tolist()  # every test S1 entity, per the format spec
    pairs = fe.explode_candidate_pairs(candidate_pairs)
    log(f"scoring {len(pairs):,} candidate pairs across {len(all_s1_ids):,} S1 entities...")

    t0 = time.time()
    feat_df = fe.compute_features(pairs, s1_lookup, other_lookup)
    log(f"features computed in {time.time() - t0:.1f}s")

    if "embedding_cosine" in feature_cols:
        log("adding embedding_cosine feature...")
        import embedding_blocking as eb
        embed_model = eb.load_model()
        feat_df = fe.add_embedding_feature(feat_df, s1_lookup, other_lookup, embed_model)

    feat_df["score"] = clf.predict_proba(feat_df[feature_cols])[:, 1]
    log("consolidating predictions (per-record argmax, gated by threshold)...")
    pred_map = consolidate(feat_df, threshold)

    rows = [(s1_id, ",".join(sorted(pred_map.get(s1_id, set())))) for s1_id in all_s1_ids]
    out_df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, sep="\t", index=False)
    n_with_matches = (out_df["matched_entity_ids"] != "").sum()
    log(f"wrote {args.output} ({len(out_df):,} rows, {n_with_matches:,} with >=1 match)")


if __name__ == "__main__":
    main()
