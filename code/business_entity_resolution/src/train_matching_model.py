"""
Phase C entry point: trains the matching classifier on train's
candidate_pairs.tsv (from run_blocking.py) against train_ground_truth.tsv.

Run with plain `python`, NOT under `python -m cudf.pandas` (same reason as
run_blocking.py: cuDF's runtime-compiled kernels fail on the qBraid box).

Split unit is the Source-1 anchor, which is a valid connected-component
split for this data: EDA confirmed full exclusivity (every Source-2/3 record
is a true match for AT MOST one Source-1 anchor, zero exceptions across the
full training ground truth), which makes the true-match graph a disjoint
union of stars, each centered on one anchor. Grouping by anchor is therefore
exactly connected-component grouping here, not an approximation of it.

The split is drawn over EVERY ground-truth anchor, including ones blocking
found no candidates for, so the reported validation macro F0.5 is the
honest end-to-end number (blocking misses included), not a matcher-only one.

Trains on the FULL candidate set as blocking actually produced it -- no
negative subsampling/rebalancing. The research is explicit that training at
a positive:negative ratio different from what inference actually produces
distorts the decision threshold. If memory forces a smaller run,
--max-anchors drops WHOLE anchors (each kept anchor keeps its full candidate
set), which leaves that ratio unchanged.

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
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The model is fit on a plain float32 array with explicit feature names;
# sklearn's "X does not have valid feature names" on predict is just noise.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import pandas as pd

import feature_engineering as fe
import pair_table as pt
from consolidation import sweep_threshold
from io_utils import read_tsv

THRESHOLDS = np.round(np.arange(0.02, 0.99, 0.01), 2)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_recall(tag, r):
    log(f"{tag}: pair recall {r['pair_recall']:.2%}, full-set recall {r['full_set_recall']:.2%} "
        f"({r['anchors_with_matches']:,} of {r['anchors']:,} anchors have >=1 true match)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True, help="Phase A output (train_*_clean.parquet)")
    ap.add_argument("--candidates-dir", required=True, help="Phase B output (train's candidate_pairs.tsv)")
    ap.add_argument("--ground-truth", required=True, help="path to train_ground_truth.tsv")
    ap.add_argument("--model-out", required=True, help="where to save the trained model + threshold")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--max-anchors", type=int, default=0,
                    help="0 = all anchors; otherwise sample this many WHOLE anchors (memory relief)")
    ap.add_argument("--workers", type=int, default=0, help="feature processes (0 = all cores)")
    ap.add_argument("--use-embeddings", action="store_true")
    args = ap.parse_args()

    log("loading processed train tables...")
    s1, other = pt.load_tables(args.processed_dir, "train")
    s1_index, other_index = pd.Index(s1["entity_id"]), pd.Index(other["entity_id"])
    log(f"s1={len(s1):,} other(S2+S3)={len(other):,}")

    log("loading ground truth...")
    truth = pt.Truth(read_tsv(args.ground_truth), s1_index, other_index)
    log(f"ground truth: {len(truth.anchors):,} anchors, {len(truth.true_keys):,} true pairs")

    log("loading candidate_pairs.tsv...")
    a_pos, c_pos = pt.parse_id_lists(read_tsv(os.path.join(args.candidates_dir, "candidate_pairs.tsv")),
                                     "source1_entity_id", "candidate_entity_ids",
                                     s1_index, other_index, "candidate")
    label = truth.label(a_pos, c_pos)
    log(f"candidate pairs: {len(a_pos):,} (positive={int(label.sum()):,}, "
        f"{len(a_pos) / len(truth.anchors):.1f} per anchor)")
    log_recall("blocking recall ceiling (all train anchors)", truth.recall(a_pos, label))

    rng = np.random.RandomState(0)
    anchors = truth.anchors.copy()
    rng.shuffle(anchors)
    if args.max_anchors and args.max_anchors < len(anchors):
        anchors = anchors[:args.max_anchors]
        log(f"using a random subset of {len(anchors):,} whole anchors (--max-anchors)")
    n_val = int(len(anchors) * args.val_fraction)
    val_anchors, train_anchors = np.sort(anchors[:n_val]), np.sort(anchors[n_val:])

    # 0 = unused anchor, 1 = train, 2 = val. Pairs are reordered train-first
    # so the train/val feature blocks are contiguous views, not copies.
    role = np.zeros(len(s1), dtype=np.int8)
    role[train_anchors] = 1
    role[val_anchors] = 2
    pair_role = role[a_pos]
    order = np.flatnonzero(pair_role > 0)
    order = order[np.argsort(pair_role[order], kind="stable")]
    a_pos, c_pos, label = a_pos[order], c_pos[order], label[order]
    n_train = int((pair_role[order] == 1).sum())
    del pair_role, order
    log(f"train anchors={len(train_anchors):,} pairs={n_train:,} | "
        f"val anchors={len(val_anchors):,} pairs={len(a_pos) - n_train:,}")

    log("computing string features (process pool)...")
    t0 = time.time()
    X = fe.compute_features(a_pos, c_pos, s1, other, workers=args.workers or None, log=log,
                            n_extra=int(args.use_embeddings))
    log(f"string features done in {time.time() - t0:.1f}s")

    feature_cols = list(fe.FEATURE_COLS)
    if args.use_embeddings:
        log("computing embedding_cosine feature...")
        t0 = time.time()
        import embedding_blocking as eb
        X[:, -1] = fe.embedding_cosine(a_pos, c_pos, s1, other, eb.load_model(), log=log)
        feature_cols.append("embedding_cosine")
        log(f"embedding feature done in {time.time() - t0:.1f}s")

    log(f"training LightGBM on {n_train:,} pairs (full candidate sets, no negative subsampling)...")
    t0 = time.time()
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(objective="binary", n_estimators=300, num_leaves=31,
                              learning_rate=0.05, random_state=0, verbosity=-1)
    clf.fit(X[:n_train], label[:n_train], feature_name=feature_cols)
    log(f"trained in {time.time() - t0:.1f}s")

    log("scoring validation pairs...")
    va, vc, vl = a_pos[n_train:], c_pos[n_train:], label[n_train:]
    v_score = clf.predict_proba(X[n_train:])[:, 1]
    log_recall("blocking recall ceiling (val anchors)", truth.recall(va, vl, val_anchors))

    log(f"sweeping {len(THRESHOLDS)} thresholds against macro F0.5 over ALL "
        f"{len(val_anchors):,} val anchors (consolidation inside the loop)...")
    best_t, best_score, results = sweep_threshold(va, vc, v_score, vl, val_anchors, truth.n_true, THRESHOLDS)
    for t, s in results:
        if abs(round(t * 100) % 5) == 0 or t == best_t:
            log(f"  threshold={t:.2f} macro F0.5={s:.4f}")
    log(f"best threshold={best_t:.2f} validation macro F0.5={best_score:.4f}")

    log("feature importances (split count):")
    for name, imp in sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1]):
        log(f"  {name}: {imp}")

    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump({"model": clf, "threshold": best_t, "feature_cols": feature_cols}, f)
    log(f"saved model + threshold to {args.model_out}")


if __name__ == "__main__":
    main()
