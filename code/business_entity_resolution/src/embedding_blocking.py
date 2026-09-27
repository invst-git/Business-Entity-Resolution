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
    """Flat inner-product index over L2-normalized vectors == exact cosine
    similarity search. Flat (exact), not an approximate index (IVF/HNSW), by
    default: correctness first -- an approximate index is a legitimate later
    optimization once flat search is confirmed too slow at real scale on the
    target hardware, not a default assumed up front."""
    import faiss
    d = candidate_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(candidate_embeddings.astype(np.float32))
    return index


def top_k_per_anchor(index, anchor_embeddings: np.ndarray, k: int):
    """Returns (anchor_idx, candidate_idx, similarity) parallel arrays -- same
    shape/contract as tfidf_blocking.top_k_per_anchor, so every channel feeds
    meta_blocking.union_edges identically."""
    k = min(k, index.ntotal)
    if k == 0 or anchor_embeddings.shape[0] == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=np.float32)
    sims, idxs = index.search(anchor_embeddings.astype(np.float32), k)
    n_anchors = anchor_embeddings.shape[0]
    anchor_idx = np.repeat(np.arange(n_anchors), k)
    cand_idx = idxs.ravel()
    sim_flat = sims.ravel()
    valid = cand_idx >= 0  # faiss returns -1 when the index has fewer than k vectors
    return anchor_idx[valid].astype(np.int64), cand_idx[valid].astype(np.int64), sim_flat[valid].astype(np.float32)
