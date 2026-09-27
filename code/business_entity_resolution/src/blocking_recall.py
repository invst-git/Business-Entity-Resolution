"""
Blocking recall ceiling on train: what share of true matches made it into
candidate_pairs.tsv. The matcher can only pick from these, so this bounds
Phase C. Run it after train blocking and BEFORE test blocking -- it is the
last point where k or the purge caps can still change for both splits.

CPU-only and light (reads only entity_id/country columns). Plain `python`.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import pair_table as pt
from io_utils import read_parquet, read_tsv


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--candidates-dir", required=True)
    ap.add_argument("--ground-truth", required=True)
    args = ap.parse_args()

    p = lambda src: os.path.join(args.processed_dir, f"train_{src}_clean.parquet")
    s1 = read_parquet(p("source1"), columns=["entity_id", "country"])
    other_ids = pd.concat([read_parquet(p(src), columns=["entity_id"])["entity_id"]
                           for src in ("source2", "source3")], ignore_index=True)
    s1_index, other_index = pd.Index(s1["entity_id"]), pd.Index(other_ids)

    truth = pt.Truth(read_tsv(args.ground_truth), s1_index, other_index)
    a_pos, c_pos = pt.parse_id_lists(read_tsv(os.path.join(args.candidates_dir, "candidate_pairs.tsv")),
                                     "source1_entity_id", "candidate_entity_ids",
                                     s1_index, other_index, "candidate")
    label = truth.label(a_pos, c_pos)
    log(f"{len(a_pos):,} candidate pairs, {len(a_pos) / len(s1):.1f} per anchor, "
        f"{int(label.sum()):,} true pairs retrieved of {len(truth.true_keys):,}")

    country = s1["country"].to_numpy()
    groups = [("ALL", truth.anchors)] + [
        (c, truth.anchors[country[truth.anchors] == c]) for c in sorted(pd.unique(country))
    ]
    for name, anchors in groups:
        r = truth.recall(a_pos, label, anchors)
        log(f"{name:>7}: pair recall {r['pair_recall']:.2%}, full-set recall {r['full_set_recall']:.2%} "
            f"({r['anchors_with_matches']:,} anchors with matches)")

    # Where the misses are: by anchor's true-set size.
    found = np.bincount(a_pos, weights=label, minlength=len(s1))
    nt = truth.n_true
    for size in range(1, int(nt.max()) + 1):
        sel = truth.anchors[nt[truth.anchors] == size]
        if len(sel):
            log(f"  true-set size {size:>2}: {len(sel):>9,} anchors, "
                f"full-set recall {(found[sel] == size).mean():.2%}")


if __name__ == "__main__":
    main()
