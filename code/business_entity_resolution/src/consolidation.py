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
"""
import pandas as pd

from scoring import macro_f_beta


def consolidate(scored_pairs: pd.DataFrame, threshold: float, score_col: str = "score") -> dict:
    """scored_pairs: DataFrame with source1_entity_id, candidate_entity_id,
    score_col (one row per candidate pair). Returns {source1_entity_id:
    set(candidate_entity_id)} -- anchors with no surviving candidate are
    simply absent, matching scoring.py's "missing key == empty prediction"
    convention."""
    above = scored_pairs[scored_pairs[score_col] >= threshold]
    if above.empty:
        return {}
    idx = above.groupby("candidate_entity_id")[score_col].idxmax()
    winners = above.loc[idx]
    result = {}
    for s1_id, cid in zip(winners["source1_entity_id"], winners["candidate_entity_id"]):
        result.setdefault(s1_id, set()).add(cid)
    return result


def sweep_threshold(scored_pairs: pd.DataFrame, true_map: dict, thresholds,
                     score_col: str = "score"):
    """For each candidate threshold: consolidate -> compute macro F0.5 against
    true_map. `true_map` must cover every anchor in the evaluation set,
    including ones with zero surviving candidates (they score via
    scoring.py's empty-prediction rule either way). Returns
    (best_threshold, best_score, [(t, score), ...] for every threshold tried).
    """
    results = []
    for t in thresholds:
        pred_map = consolidate(scored_pairs, t, score_col)
        score = macro_f_beta(true_map, pred_map)
        results.append((t, score))
    best_t, best_score = max(results, key=lambda x: x[1])
    return best_t, best_score, results
