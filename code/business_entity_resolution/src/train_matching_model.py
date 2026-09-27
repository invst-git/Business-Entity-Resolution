"""
Phase C entry point: trains the matching classifier on train's
candidate_pairs.tsv (from run_blocking.py) against train_ground_truth.tsv.

Split unit is the Source-1 anchor, which is a valid connected-component
split for this data: EDA confirmed full exclusivity (every Source-2/3 record
is a true match for AT MOST one Source-1 anchor, zero exceptions across the
full training ground truth), which makes the true-match graph a disjoint
union of stars, each centered on one anchor. Grouping by anchor is therefore
exactly connected-component grouping here, not an approximation of it.

Trains on the FULL candidate set as blocking actually produced it -- no
negative subsampling/rebalancing. The research is explicit that training at
a positive:negative ratio different from what inference actually produces
distorts the decision threshold; if training time later forces subsampling,
it must be done by anchor (drop whole anchors), never by dropping negatives
within a kept anchor's candidate set.

Threshold selection sweeps the ACTUAL macro per-anchor F0.5 metric with
consolidation applied inside the loop (see consolidation.py) -- not a
generic pairwise precision-recall curve, which is not guaranteed to pick the
threshold that maximizes this project's actual scoring function.
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import feature_engineering as fe
from consolidation import sweep_threshold
from io_utils import read_parquet, read_tsv
from scoring import macro_f_beta
from transliteration import add_compare_name

DEFAULT_THRESHOLDS = np.arange(0.05, 0.96, 0.05)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_processed(processed_dir, split):
    s1 = read_parquet(os.path.join(processed_dir, f"{split}_source1_clean.parquet"))
    s2 = read_parquet(os.path.join(processed_dir, f"{split}_source2_clean.parquet"))
    s3 = read_parquet(os.path.join(processed_dir, f"{split}_source3_clean.parquet"))
    s1 = add_compare_name(s1)
    s2 = add_compare_name(s2)
    s3 = add_compare_name(s3)
    return s1, s2, s3


def label_pairs(pairs_df: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """1 if candidate_entity_id is in the anchor's true match set, else 0.
    Ground truth here is exhaustive (the full training labels, not a partial
    sample), so every non-match pair really is a true negative -- no
    inter-label-noise risk, unlike a setting with incomplete labels."""
    true_map = {}
    for s1_id, matched in zip(ground_truth["source1_entity_id"], ground_truth["matched_entity_ids"]):
        true_map[s1_id] = set(matched.split(",")) if matched else set()
    labels = [
        int(cid in true_map.get(s1_id, set()))
        for s1_id, cid in zip(pairs_df["source1_entity_id"], pairs_df["candidate_entity_id"])
    ]
    out = pairs_df.copy()
    out["label"] = labels
    return out, true_map


def split_by_anchor(anchor_ids, val_fraction=0.2, seed=0):
    rng = np.random.RandomState(seed)
    unique_anchors = np.array(sorted(set(anchor_ids)))
    rng.shuffle(unique_anchors)
    n_val = int(len(unique_anchors) * val_fraction)
    val_anchors = set(unique_anchors[:n_val])
    train_anchors = set(unique_anchors[n_val:])
    return train_anchors, val_anchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True, help="Phase A output (train_*_clean.parquet)")
    ap.add_argument("--candidates-dir", required=True, help="Phase B output (train's candidate_pairs.tsv)")
    ap.add_argument("--ground-truth", required=True, help="path to train_ground_truth.tsv")
    ap.add_argument("--model-out", required=True, help="where to save the trained model + threshold")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--use-embeddings", action="store_true")
    args = ap.parse_args()

    log("loading processed train data...")
    s1, s2, s3 = load_processed(args.processed_dir, "train")
    s1_lookup = fe.build_lookup(s1)
    other_lookup = fe.build_lookup(s2)
    other_lookup.update(fe.build_lookup(s3))
    log(f"lookups built: s1={len(s1_lookup):,} other={len(other_lookup):,}")

    log("loading candidate_pairs.tsv and ground truth...")
    candidate_pairs = read_tsv(os.path.join(args.candidates_dir, "candidate_pairs.tsv"))
    ground_truth = read_tsv(args.ground_truth)

    log("exploding candidate pairs and labeling from ground truth...")
    pairs = fe.explode_candidate_pairs(candidate_pairs)
    pairs, true_map = label_pairs(pairs, ground_truth)
    log(f"total candidate pairs: {len(pairs):,} (positive={pairs['label'].sum():,})")

    # Every anchor in ground truth must be represented in true_map for
    # scoring, including anchors with zero surviving candidates (blocking
    # found nothing) -- true_map already covers all of them since it's built
    # directly from ground_truth, not from pairs.

    log("computing features...")
    t0 = time.time()
    feat_df = fe.compute_features(pairs[["source1_entity_id", "candidate_entity_id"]], s1_lookup, other_lookup)
    feat_df["label"] = pairs["label"].values[:len(feat_df)]
    log(f"features computed for {len(feat_df):,} pairs in {time.time() - t0:.1f}s")

    feature_cols = list(fe.FEATURE_COLS)
    if args.use_embeddings:
        log("adding embedding_cosine feature...")
        import embedding_blocking as eb
        model = eb.load_model()
        feat_df = fe.add_embedding_feature(feat_df, s1_lookup, other_lookup, model)
        feature_cols.append("embedding_cosine")

    log("splitting by anchor (connected-component split -- see module docstring)...")
    train_anchors, val_anchors = split_by_anchor(feat_df["source1_entity_id"], args.val_fraction)
    train_mask = feat_df["source1_entity_id"].isin(train_anchors)
    val_mask = feat_df["source1_entity_id"].isin(val_anchors)
    train_feat, val_feat = feat_df[train_mask], feat_df[val_mask]
    log(f"train pairs={len(train_feat):,} (anchors={len(train_anchors):,}) "
        f"val pairs={len(val_feat):,} (anchors={len(val_anchors):,})")

    log("training LightGBM classifier on the FULL train candidate set (no negative subsampling)...")
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(objective="binary", n_estimators=300, num_leaves=31,
                              learning_rate=0.05, random_state=0, verbosity=-1)
    clf.fit(train_feat[feature_cols], train_feat["label"])

    log("scoring validation candidates...")
    val_scored = val_feat[["source1_entity_id", "candidate_entity_id"]].copy()
    val_scored["score"] = clf.predict_proba(val_feat[feature_cols])[:, 1]

    val_true_map = {s1_id: true_map[s1_id] for s1_id in val_anchors}
    log(f"sweeping threshold against macro F0.5 on {len(val_true_map):,} validation anchors...")
    best_t, best_score, results = sweep_threshold(val_scored, val_true_map, DEFAULT_THRESHOLDS)
    log(f"best threshold={best_t:.2f} macro F0.5={best_score:.4f}")
    for t, s in results:
        log(f"  threshold={t:.2f} macro F0.5={s:.4f}")

    importances = sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1])
    log("feature importances:")
    for name, imp in importances:
        log(f"  {name}: {imp}")

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump({"model": clf, "threshold": best_t, "feature_cols": feature_cols}, f)
    log(f"saved model + threshold to {args.model_out}")


if __name__ == "__main__":
    main()
