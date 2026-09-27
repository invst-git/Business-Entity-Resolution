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
import time

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

NGRAM_RANGE = (3, 5)
MAX_DF = 0.3
MIN_DF = 2
BATCH_SIZE = 1000
# Absolute cap on how many CANDIDATE records may share an n-gram for that
# n-gram to be used for retrieval (block purging). The relative MAX_DF above
# is not enough on its own: at ~3M candidates, 0.3 still permits n-gram
# "blocks" of ~900K records, so nearly every anchor shares some surviving
# trigram with nearly every candidate and the sparse product degenerates to
# near all-pairs. Extrapolated from a measured 1.3s on an 8.9K x 14.8K sample,
# the uncapped US x S2 retrieval would take ~11 hours. An absolute cap bounds
# the work per anchor (<= n-grams-per-name * cap) regardless of corpus size.
# Chosen from a measured recall/compute curve on a real 15K-GT-row sample
# (US, S1 vs S2), with the cap scaled proportionally to full size:
#   cap(full)   full-set recall   ~full multiply-adds (US x S2)
#   none        97.7%             2.1e12
#   ~41K        97.3%             5.4e11
#   ~20K        96.4%             2.7e11
#   ~10K        92.3%             8.3e10
#   ~1K         77.8%             3.4e9
# 40K keeps recall within ~0.4pp of uncapped at ~4x less work. Smaller caps
# (an earlier draft used 1000) cost far too much recall: short/generic
# business names are made almost entirely of common n-grams, so aggressive
# purging leaves them nothing to match on.
MAX_BLOCK_SIZE = 40000


def prune_common_terms(anchor_matrix: sp.csr_matrix, candidate_matrix: sp.csr_matrix,
                        max_block_size: int = MAX_BLOCK_SIZE):
    """Drops vocabulary columns whose candidate-side document frequency
    exceeds max_block_size, from BOTH matrices (same columns on both sides,
    so dot products stay comparable). Returns (anchor_matrix, candidate_matrix,
    n_kept, n_dropped). Dropping columns from L2-normalized rows keeps every
    dot product <= 1, so the scores remain valid ranking weights."""
    col_df = np.bincount(candidate_matrix.indices, minlength=candidate_matrix.shape[1])
    keep = col_df <= max_block_size
    return anchor_matrix[:, keep], candidate_matrix[:, keep], int(keep.sum()), int((~keep).sum())


def fit_vectorizer(texts, max_df=MAX_DF, min_df=MIN_DF, ngram_range=NGRAM_RANGE):
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=ngram_range,
        max_df=max_df, min_df=min_df, dtype=np.float32,
    )
    vec.fit(texts)
    return vec


GPU_BATCH_SIZE = 256
PROGRESS_EVERY = 0.05


def get_torch_cuda():
    """torch if a CUDA device is usable, else None. PyTorch, not CuPy: on the
    qBraid box, installing cuML pulled in a CUDA 12.9 runtime compiler
    (NVRTC) while the driver supports 12.8, so every kernel CuPy compiles at
    runtime fails with CUDA_ERROR_INVALID_IMAGE (hit live). PyTorch ships
    precompiled kernels and is already proven in the same process -- the
    embedding step runs on it."""
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


def _to_torch_csr(m: sp.csr_matrix, torch, device):
    # 32-bit indices: the most broadly supported form for GPU sparse-sparse
    # matmul (cuSPARSE SpGEMM). Every size here fits (candidate matrix nnz
    # ~1.6e8, dimensions ~1e6-3e6, per-batch product nnz < 256 * 3.1e6).
    m = m.tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(m.indptr.astype(np.int32)),
        torch.from_numpy(m.indices.astype(np.int32)),
        torch.from_numpy(m.data.astype(np.float32)),
        size=m.shape, device=device,
    )


def top_k_torch_batches(anchor_matrix: sp.csr_matrix, candidate_matrix: sp.csr_matrix, k: int,
                         torch, device, batch_size: int = GPU_BATCH_SIZE, log=None):
    """Batched top-k: (sparse anchor batch @ sparse candidates^T) on `device`,
    densified per batch, then torch.topk along each row -- one vectorized op
    per batch instead of a Python loop over sparse rows. The candidate
    matrix is moved to the device once; each anchor batch is row-sliced on
    the host (cheap) and moved per batch. Memory per batch is roughly
    batch_size * n_candidates * 4 bytes (256 x 3M ~= 3 GB) plus the sparse
    product, within an 80 GB A100.

    Runs identically on device="cpu" (used to verify this exact logic against
    the scipy path locally, where no GPU is available). Same output contract
    as top_k_per_anchor: only strictly positive similarities are kept."""
    n_anchors, n_candidates = anchor_matrix.shape[0], candidate_matrix.shape[0]
    k = min(k, n_candidates)
    anchor_matrix = anchor_matrix.tocsr()
    CT = _to_torch_csr(candidate_matrix.T.tocsr(), torch, device)

    out_anchor, out_cand, out_sim = [], [], []
    n_batches = (n_anchors + batch_size - 1) // batch_size
    report_every = max(1, int(n_batches * PROGRESS_EVERY))
    t_start = time.time()
    for b, start in enumerate(range(0, n_anchors, batch_size)):
        end = min(start + batch_size, n_anchors)
        A_b = _to_torch_csr(anchor_matrix[start:end], torch, device)
        dense = (A_b @ CT).to_dense()
        vals, top = torch.topk(dense, k, dim=1)
        keep = vals > 0
        rows = torch.arange(start, end, device=device).unsqueeze(1).expand_as(top)
        out_anchor.append(rows[keep].cpu().numpy().astype(np.int64))
        out_cand.append(top[keep].cpu().numpy().astype(np.int64))
        out_sim.append(vals[keep].cpu().numpy().astype(np.float32))
        del A_b, dense, vals, top, keep, rows
        if log is not None and ((b + 1) % report_every == 0 or b + 1 == n_batches):
            elapsed = time.time() - t_start
            eta = elapsed / (b + 1) * (n_batches - b - 1)
            log(f"      tfidf retrieval {b + 1:,}/{n_batches:,} batches, "
                f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    del CT
    return np.concatenate(out_anchor), np.concatenate(out_cand), np.concatenate(out_sim)


def top_k_per_anchor(anchor_matrix: sp.csr_matrix, candidate_matrix: sp.csr_matrix,
                      k: int, batch_size: int = BATCH_SIZE, log=None, use_gpu: bool = True):
    """Top-k candidate indices (into candidate_matrix's rows) and their cosine
    similarity, per anchor row (into anchor_matrix's rows). TfidfVectorizer
    output is already L2-normalized, so the sparse dot product IS the cosine
    similarity -- no separate normalization step needed.

    Uses the GPU (PyTorch, top_k_torch_batches) when available. The CPU path
    below is the fallback: correct, but at full scale it is single-threaded
    scipy -- measured ~5x10^7 multiply-adds/s, i.e. hours per large country
    block even with block-size purging.

    Returns three parallel 1-D numpy arrays: anchor_row_idx, candidate_row_idx,
    similarity -- one triple per surviving (anchor, candidate) edge.
    """
    n_anchors = anchor_matrix.shape[0]
    n_candidates = candidate_matrix.shape[0]
    if n_anchors == 0 or n_candidates == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=np.float32)

    torch = get_torch_cuda() if use_gpu else None
    if torch is not None:
        if log is not None:
            log("      tfidf retrieval backend: GPU (torch)")
        return top_k_torch_batches(anchor_matrix, candidate_matrix, k, torch, "cuda", log=log)
    if log is not None:
        log("      tfidf retrieval backend: CPU (scipy) -- slow at full scale")

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
