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


def build_ann_index(candidate_embeddings: np.ndarray):
    """Tries cuML's GPU brute-force KNN first, falls back to FAISS CPU flat
    search only when cuML isn't available (e.g. local/small-scale testing).

    This fallback matters: a CPU flat (exact) search is O(n_anchors *
    n_candidates * dim) with no GPU parallelism at all, and at this
    project's real scale that is not a "slower but viable" option, it is
    hours-to-days per country/target-source pair -- confirmed in practice,
    not just estimated (a single US-partition search stalled at 0% GPU
    utilization for 25+ minutes before being killed). GPU brute-force KNN
    turns the same computation (~4x10^15 FLOPs for the US partition alone)
    into a task an A100 finishes in seconds to low minutes. Both paths do
    EXACT search -- this is a backend change, not an accuracy trade-off; an
    approximate index (IVF/HNSW) remains a legitimate later optimization if
    even GPU exact search proves too slow, not a default assumed here."""
    try:
        from cuml.neighbors import NearestNeighbors  # noqa: F401 -- import-checked only
        return ("cuml", candidate_embeddings.astype(np.float32))
    except ImportError:
        import faiss
        d = candidate_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(candidate_embeddings.astype(np.float32))
        return ("faiss", index)


def top_k_per_anchor(index, anchor_embeddings: np.ndarray, k: int):
    """Returns (anchor_idx, candidate_idx, similarity) parallel arrays -- same
    shape/contract as tfidf_blocking.top_k_per_anchor, so every channel feeds
    meta_blocking.union_edges identically. `index` is whatever
    build_ann_index() returned (a (backend, payload) tuple)."""
    backend, payload = index
    n_anchors = anchor_embeddings.shape[0]

    if backend == "cuml":
        from cuml.neighbors import NearestNeighbors
        n_candidates = payload.shape[0]
        k_eff = min(k, n_candidates)
        if k_eff == 0 or n_anchors == 0:
            empty = np.array([], dtype=np.int64)
            return empty, empty, np.array([], dtype=np.float32)
        nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
        nn.fit(payload)
        dist, idxs = nn.kneighbors(anchor_embeddings.astype(np.float32))
        dist = np.asarray(dist)
        idxs = np.asarray(idxs)
        # Both sides are L2-normalized, so for unit vectors:
        # ||a-b||^2 = 2 - 2*cos_sim  =>  cos_sim = 1 - ||a-b||^2 / 2.
        # Used instead of asking cuML for cosine/inner-product directly
        # since 'euclidean' is the one metric guaranteed supported by every
        # cuML brute-force KNN version -- this identity gets the same exact
        # cosine ranking without depending on less-universal metric support.
        sims = 1.0 - (dist ** 2) / 2.0
        anchor_idx = np.repeat(np.arange(n_anchors), k_eff)
        cand_idx = idxs.ravel()
        sim_flat = sims.ravel()
        return anchor_idx.astype(np.int64), cand_idx.astype(np.int64), sim_flat.astype(np.float32)

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
