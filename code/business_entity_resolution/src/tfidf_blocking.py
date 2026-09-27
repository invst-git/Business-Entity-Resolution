"""
Character n-gram TF-IDF blocking channel.

Fits ONE TfidfVectorizer per country, on the concatenation of that country's
S1 + S2 + S3 text -- fitting separately per source would give the same
n-gram different IDF weight on each side of a comparison and silently
distort cosine similarity between them. `max_df` prunes n-grams that appear
in a large fraction of that country's records (business-suffix fragments
like "lim", "ted", "pvt" are exactly this case) BEFORE the sparse similarity
computation, not after: an unpruned high-frequency n-gram makes that
vocabulary column dense against millions of rows, which blows up the size of
the sparse similarity product even though only ~30 results per row survive.
This is the feature-level analogue of the "purge oversized blocks" step
described in the blocking-methodology research.

Retrieval itself is a batched sparse matrix product (anchor batch times the
full candidate matrix, transposed), not a call to a generic nearest-neighbor
library: sparse @ sparse only computes nonzero entries for pairs sharing at
least one surviving (non-pruned) n-gram, which is the efficient equivalent of
an inverted-index lookup, and keeping this as plain scipy sparse arithmetic
(rather than a wrapped NearestNeighbors call) leaves room for a GPU-array
substitution later without restructuring the algorithm.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

NGRAM_RANGE = (3, 5)
MAX_DF = 0.3
MIN_DF = 2
BATCH_SIZE = 2000


def fit_vectorizer(texts, max_df=MAX_DF, min_df=MIN_DF, ngram_range=NGRAM_RANGE):
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=ngram_range,
        max_df=max_df, min_df=min_df, dtype=np.float32,
    )
    vec.fit(texts)
    return vec


def top_k_per_anchor(anchor_matrix: sp.csr_matrix, candidate_matrix: sp.csr_matrix,
                      k: int, batch_size: int = BATCH_SIZE):
    """Top-k candidate indices (into candidate_matrix's rows) and their cosine
    similarity, per anchor row (into anchor_matrix's rows). TfidfVectorizer
    output is already L2-normalized, so the sparse dot product IS the cosine
    similarity -- no separate normalization step needed.

    Returns three parallel 1-D numpy arrays: anchor_row_idx, candidate_row_idx,
    similarity -- one triple per surviving (anchor, candidate) edge.
    """
    n_anchors = anchor_matrix.shape[0]
    n_candidates = candidate_matrix.shape[0]
    if n_anchors == 0 or n_candidates == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=np.float32)

    k = min(k, n_candidates)
    candidate_T = candidate_matrix.T.tocsr()

    out_anchor, out_cand, out_sim = [], [], []
    for start in range(0, n_anchors, batch_size):
        end = min(start + batch_size, n_anchors)
        sims = (anchor_matrix[start:end] @ candidate_T).tocsr()
        for local_i in range(sims.shape[0]):
            row_start, row_end = sims.indptr[local_i], sims.indptr[local_i + 1]
            if row_start == row_end:
                continue
            row_idx = sims.indices[row_start:row_end]
            row_val = sims.data[row_start:row_end]
            if len(row_val) > k:
                top = np.argpartition(row_val, -k)[-k:]
                row_idx = row_idx[top]
                row_val = row_val[top]
            out_anchor.append(np.full(len(row_idx), start + local_i, dtype=np.int64))
            out_cand.append(row_idx.astype(np.int64))
            out_sim.append(row_val.astype(np.float32))

    if not out_anchor:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=np.float32)
    return np.concatenate(out_anchor), np.concatenate(out_cand), np.concatenate(out_sim)
