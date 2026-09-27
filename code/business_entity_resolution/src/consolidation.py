"""
Per-record argmax consolidation, and threshold selection built around it.

Not bipartite/Hungarian assignment: this project's structure is many-to-one
(one anchor absorbs several true matches -- mean 3.46, confirmed via EDA),
not one-to-one, so an assignment algorithm capping each anchor at a single
match would discard the majority of true matches. The only thing needing
"collective" resolution is the rare case where blocking retrieved the SAME
candidate record for more than one anchor (blocking doesn't know ground
truth, so this can happen even though a candidate can only be a TRUE match
for at most one anchor -- confirmed via EDA: zero duplicates across the
full training ground truth's matched ids). For that candidate, keep it only
for whichever anchor scores it highest, and only if that score clears the
decision threshold.

Threshold selection must run consolidation INSIDE the sweep, not after
picking a threshold from raw pair scores: the argmax step changes which
pairs survive, so macro F0.5 at threshold t has to be measured on the
CONSOLIDATED predictions at that same t, or the sweep optimizes a different
objective than the one actually being deployed.

Works on integer positions (pair_table) with numpy, so a sweep over tens of
millions of validation pairs takes seconds per threshold.
"""
import numpy as np

from scoring import macro_f_beta_arrays


def consolidate(anchor_pos: np.ndarray, other_pos: np.ndarray, score: np.ndarray,
                threshold: float) -> np.ndarray:
    """Boolean mask over the pairs: True where the pair is predicted as a
    match -- score >= threshold AND it is the highest-scoring surviving pair
    for its candidate record (ties keep the earliest pair)."""
    keep = np.zeros(len(score), dtype=bool)
    above = np.flatnonzero(score >= threshold)
    if len(above) == 0:
        return keep
    # Sort surviving pairs by candidate, then by descending score (lexsort is
    # stable, so equal scores stay in original order); the first pair of
    # each candidate run is its argmax.
    order = above[np.lexsort((-score[above], other_pos[above]))]
    cand_sorted = other_pos[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = cand_sorted[1:] != cand_sorted[:-1]
    keep[order[first]] = True
    return keep


def sweep_threshold(anchor_pos, other_pos, score, label, eval_anchors, n_true, thresholds):
    """For each candidate threshold: consolidate -> macro F0.5 over
    `eval_anchors` (EVERY anchor in the evaluation set, including ones
    blocking found no candidates for -- they score by the singleton rule
    either way). `label` marks true pairs, `n_true` is the per-anchor count
    of true matches (indexed by anchor position). Returns (best_threshold,
    best_score, [(t, score), ...])."""
    results = []
    for t in thresholds:
        pred = consolidate(anchor_pos, other_pos, score, t)
        s = macro_f_beta_arrays(eval_anchors, n_true, anchor_pos[pred], label[pred])
        results.append((float(t), s))
    best_t, best_score = max(results, key=lambda x: x[1])
    return best_t, best_score, results
