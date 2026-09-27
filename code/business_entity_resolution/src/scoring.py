"""
The exact competition metric: F_0.5 = (1.25 * P * R) / (0.25 * P + R), computed
PER Source-1 entity, then macro-averaged across all entities -- including
singletons, where a correct empty prediction scores 1.0 and any false match
scores 0.0. Getting this exactly right matters more than anything else built
in Phase C: threshold selection, model comparison, and validation reporting
all depend on this function being correct, and a subtle bug here would
silently corrupt every decision downstream without ever raising an error.

Written once, tested against explicit edge cases (empty/empty, empty/non-
empty, non-empty/empty, partial overlap, full overlap, zero overlap) before
being used anywhere else in Phase C.
"""


def f_beta_one_entity(true_ids: set, pred_ids: set, beta: float = 0.5) -> float:
    """F_beta for a single Source-1 entity's true vs. predicted match sets.
    Mirrors the README's stated rule exactly: a true-empty entity scores 1.0
    for a correct empty prediction, 0.0 for any false match; a true-non-empty
    entity with an empty prediction scores 0.0 (zero recall)."""
    if not true_ids:
        return 1.0 if not pred_ids else 0.0
    if not pred_ids:
        return 0.0
    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def macro_f_beta(true_map: dict, pred_map: dict, beta: float = 0.5) -> float:
    """Macro-average across every key in `true_map` (every Source-1 entity in
    the evaluation set). `pred_map` may be missing a key entirely -- treated
    as an empty prediction for that entity, same as an explicit empty set."""
    if not true_map:
        return 0.0
    scores = [
        f_beta_one_entity(true_ids, pred_map.get(s1_id, set()), beta)
        for s1_id, true_ids in true_map.items()
    ]
    return sum(scores) / len(scores)


def macro_f_beta_arrays(eval_anchors, n_true, pred_anchor_pos, pred_is_true, beta: float = 0.5) -> float:
    """Vectorized macro_f_beta over integer anchor positions, for full-scale
    threshold sweeps. Same per-entity rule as f_beta_one_entity:
      eval_anchors     anchor positions being scored (every one counts)
      n_true           per-anchor true-match count, indexed by position
      pred_anchor_pos  anchor position of each predicted pair
      pred_is_true     1 where that predicted pair is a true match
    Checked against macro_f_beta on real validation data before use."""
    import numpy as np

    eval_anchors = np.asarray(eval_anchors)
    if len(eval_anchors) == 0:
        return 0.0
    size = len(n_true)
    n_pred = np.bincount(pred_anchor_pos, minlength=size)[eval_anchors]
    tp = np.bincount(pred_anchor_pos, weights=pred_is_true, minlength=size)[eval_anchors]
    nt = np.asarray(n_true)[eval_anchors]

    f = np.zeros(len(eval_anchors), dtype=np.float64)
    f[(nt == 0) & (n_pred == 0)] = 1.0
    ok = (nt > 0) & (tp > 0)
    p = tp[ok] / n_pred[ok]
    r = tp[ok] / nt[ok]
    b2 = beta * beta
    f[ok] = (1 + b2) * p * r / (b2 * p + r)
    return float(f.mean())
