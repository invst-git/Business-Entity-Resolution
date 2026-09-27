"""
Meta-blocking: unions candidate edges from all blocking channels into a
single weighted anchor-candidate graph, then prunes to a per-anchor budget K.

Uses plain (non-reciprocal) Cardinality Node Pruning, on the anchor side
only. An earlier design used reciprocal pruning (requiring both endpoints
to rank each other in their own top-K); that was wrong for this project's
bipartite S1-anchor / S2-or-S3-candidate structure and was corrected before
being built: most S2/S3 records are retrieved by very few anchors (many by
exactly one), so "does this anchor rank in the candidate's top-K" is close
to trivially true for them and does nothing useful, while it starts
INCORRECTLY dropping true matches on the hard cases -- a common business
name retrieved by many anchors -- which is exactly backwards. The TKDE
meta-blocking paper's own guidance, that node-centric caps are right "for
entity collections expected to contain a large portion of duplicate
profiles," describes this project's ANCHORS (most have a true match), not
its candidate pool (~26% of which are pure distractors with none at all).
Any second-side signal belongs in the matching-stage consolidation step
(per-record argmax across retrieving anchors, done later), not in blocking.
"""
import numpy as np
import scipy.sparse as sp


def union_edges(channel_edges, n_anchors, n_candidates):
    """channel_edges: list of (anchor_idx, candidate_idx, weight) numpy-array
    triples, one per channel. Aggregates by SUMMING weight across channels
    for the same (anchor, candidate) pair -- an edge found by multiple
    channels is stronger evidence than one found by a single channel, which
    is the redundancy signal this kind of pruning is meant to exploit."""
    non_empty = [(a, c, w) for a, c, w in channel_edges if len(a) > 0]
    if not non_empty:
        return sp.csr_matrix((n_anchors, n_candidates), dtype=np.float32)
    all_anchor = np.concatenate([a for a, c, w in non_empty])
    all_cand = np.concatenate([c for a, c, w in non_empty])
    all_weight = np.concatenate([w for a, c, w in non_empty])
    mat = sp.coo_matrix((all_weight, (all_anchor, all_cand)), shape=(n_anchors, n_candidates), dtype=np.float32)
    return mat.tocsr()  # duplicate (row, col) entries are summed on coo->csr conversion


def prune_to_budget(edge_matrix: sp.csr_matrix, k: int):
    """Per-anchor (per-row) top-k by weight -- plain Cardinality Node
    Pruning, anchor side only. Returns (anchor_idx, candidate_idx, weight)
    parallel arrays for surviving edges."""
    out_anchor, out_cand, out_weight = [], [], []
    n_rows = edge_matrix.shape[0]
    for i in range(n_rows):
        row_start, row_end = edge_matrix.indptr[i], edge_matrix.indptr[i + 1]
        if row_start == row_end:
            continue
        idx = edge_matrix.indices[row_start:row_end]
        val = edge_matrix.data[row_start:row_end]
        if len(val) > k:
            top = np.argpartition(val, -k)[-k:]
            idx = idx[top]
            val = val[top]
        out_anchor.append(np.full(len(idx), i, dtype=np.int64))
        out_cand.append(idx.astype(np.int64))
        out_weight.append(val.astype(np.float32))
    if not out_anchor:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=np.float32)
    return np.concatenate(out_anchor), np.concatenate(out_cand), np.concatenate(out_weight)
