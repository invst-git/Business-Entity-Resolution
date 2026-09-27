"""
Frozen multilingual embedding blocking channel via multilingual-e5-large-instruct
(MIT-licensed, ~560M params, verified via HF model card to cover all Indic
scripts and French this project needs -- see the multilingual-models survey
in research_notes/).

Embeds core_name ONLY, not core_name+address concatenated: ~3.3% of S2/S3
records have no address at all (EDA finding), and address component order is
randomized, so concatenating would make part of the vector a function of
field-ordering noise and dilute the name signal this channel exists to
capture.

This channel is the semantic/cross-script robustness backstop the discrete
channels (n-gram, phonetic, exact-match) cannot provide -- but cross-script
matching (a Latin name vs. its native-script transliteration) is a
documented weak point for embeddings in the research, not a benchmark-backed
capability. It complements the transliteration branch; it doesn't replace it.

Compute cost is the single largest item in this phase (tens of millions of
records through a 560M-parameter encoder) and could not be measured without
the actual GPU -- run scripts/probe_embedding_throughput.py on the target
machine before committing to full-corpus embedding. If throughput is too
low, the fallback is restricting this channel to records where the lexical
channels (n-gram/phonetic/exact) found nothing, rather than running it over
every record.
"""
import time

import numpy as np

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
EMBED_BATCH_SIZE = 256
# e5 models: a uniform "query: " prefix is the model card's recommendation
# for symmetric similarity tasks (comparing two same-type texts), as opposed
# to the asymmetric "query:"/"passage:" split used for retrieval.
EMBED_PREFIX = "query: "


def load_model(device=None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)


def embed_texts(model, texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False):
    prefixed = [EMBED_PREFIX + (t or "") for t in texts]
    return model.encode(
        prefixed, batch_size=batch_size, show_progress_bar=show_progress_bar,
        normalize_embeddings=True, convert_to_numpy=True,
    )


SEARCH_BATCH_SIZE = 1024


def build_ann_index(candidate_embeddings: np.ndarray, device=None):
    """Exact cosine search index. GPU (PyTorch) when available: embeddings are
    L2-normalized, so cosine similarity is a plain matrix multiply, and exact
    search over ~3M x 1024 candidates is ~8x10^15 FLOPs for the US block --
    tens of seconds on an A100 in fp16, versus hours-to-days as a CPU flat
    search (hit live: a CPU FAISS search sat at 0% GPU for 25+ minutes).

    PyTorch rather than cuML/CuPy: on the qBraid box those runtime-compile
    kernels through a CUDA 12.9 NVRTC the 12.8 driver rejects
    (CUDA_ERROR_INVALID_IMAGE, hit live), and cuML additionally needed
    LD_LIBRARY_PATH surgery. PyTorch's kernels are precompiled and already
    proven in this process (the embedding step runs on it).

    `device` overrides auto-detection ("cpu" is used to test this exact code
    path locally). Falls back to FAISS CPU flat search when neither applies.
    Both paths are EXACT search -- a backend change, not an accuracy
    trade-off."""
    torch = None
    if device is None:
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                torch, device = _torch, "cuda"
        except Exception:
            pass
    else:
        import torch as _torch
        torch = _torch
    if torch is not None:
        dtype = torch.float16 if device == "cuda" else torch.float32
        cand = torch.from_numpy(candidate_embeddings.astype(np.float32)).to(device=device, dtype=dtype)
        return ("torch", (torch, device, dtype, cand))
    import faiss
    d = candidate_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(candidate_embeddings.astype(np.float32))
    return ("faiss", index)


def top_k_per_anchor(index, anchor_embeddings: np.ndarray, k: int, log=None):
    """Returns (anchor_idx, candidate_idx, similarity) parallel arrays -- same
    shape/contract as tfidf_blocking.top_k_per_anchor, so every channel feeds
    meta_blocking.union_edges identically. `index` is whatever
    build_ann_index() returned (a (backend, payload) tuple)."""
    backend, payload = index
    n_anchors = anchor_embeddings.shape[0]

    if backend == "torch":
        torch, device, dtype, cand = payload
        k_eff = min(k, cand.shape[0])
        if k_eff == 0 or n_anchors == 0:
            empty = np.array([], dtype=np.int64)
            return empty, empty, np.array([], dtype=np.float32)
        out_a, out_c, out_s = [], [], []
        n_batches = (n_anchors + SEARCH_BATCH_SIZE - 1) // SEARCH_BATCH_SIZE
        report_every = max(1, n_batches // 20)
        t0 = time.time()
        for b, start in enumerate(range(0, n_anchors, SEARCH_BATCH_SIZE)):
            end = min(start + SEARCH_BATCH_SIZE, n_anchors)
            q = torch.from_numpy(anchor_embeddings[start:end].astype(np.float32)).to(device=device, dtype=dtype)
            vals, idx = torch.topk(q @ cand.T, k_eff, dim=1)
            out_a.append(np.repeat(np.arange(start, end, dtype=np.int64), k_eff))
            out_c.append(idx.cpu().numpy().astype(np.int64).ravel())
            out_s.append(vals.float().cpu().numpy().astype(np.float32).ravel())
            if log is not None and ((b + 1) % report_every == 0 or b + 1 == n_batches):
                el = time.time() - t0
                log(f"      embedding search {b + 1:,}/{n_batches:,} batches, "
                    f"{el:.0f}s elapsed, ~{el / (b + 1) * (n_batches - b - 1):.0f}s remaining")
        return np.concatenate(out_a), np.concatenate(out_c), np.concatenate(out_s)

    k_eff = min(k, payload.ntotal)
    if k_eff == 0 or n_anchors == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=np.float32)
    sims, idxs = payload.search(anchor_embeddings.astype(np.float32), k_eff)
    anchor_idx = np.repeat(np.arange(n_anchors), k_eff)
    cand_idx = idxs.ravel()
    sim_flat = sims.ravel()
    valid = cand_idx >= 0  # faiss returns -1 when the index has fewer than k vectors
    return anchor_idx[valid].astype(np.int64), cand_idx[valid].astype(np.int64), sim_flat[valid].astype(np.float32)
