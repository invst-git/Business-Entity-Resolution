"""
Computes the pairwise similarity feature vector for candidate (anchor,
candidate) pairs from run_blocking.py's candidate_pairs.tsv, to train and
later score the matching classifier.

Every name-similarity feature uses `core_name_compare`
(transliteration.add_compare_name) -- the SAME field blocking retrieved
candidates on, not a second, independently-derived one. Two romanization/
accent-folding code paths that could silently diverge would mean the model
trains on features computed from text that doesn't match what a candidate
was actually retrieved on. Embedding similarity instead uses raw
`core_name`, matching what embedding_blocking.py embeds during retrieval,
for the identical reason.

`country` is deliberately NOT a feature. Within a country-partitioned
blocking pass every pair already shares the same country -- it's constant
information there -- but a raw country feature would hand the classifier a
lever to memorize US/India-specific patterns, with no France-labeled data
available to catch that failure at training time.

Scale: pairs arrive as int32 row positions into the pair_table.load_tables
frames (never as per-pair strings or per-entity dicts). String features are
computed in fixed-size chunks across a process pool; each chunk's input
strings are gathered by position in the parent and shipped to a worker, so
worker memory is bounded by the chunk, not the corpus. Chunks are submitted
in bounded waves because Pool.imap would otherwise pull (and pickle) the
whole input at once. The embedding feature runs as a batched gather-and-dot
on the GPU.
"""
import gc
import os
import time

import numpy as np
import jellyfish
from rapidfuzz.distance import JaroWinkler, Levenshtein

FEATURE_COLS = [
    "name_jaro_winkler", "name_levenshtein", "name_char_ngram_jaccard",
    "name_token_jaccard", "name_phonetic_match", "name_len_ratio",
    "suffix_overlap",
    "address_token_jaccard", "address_comparable",
    "postal_match", "postal_both_present",
    "house_number_match", "house_number_both_present",
    "state_norm_match",
]

# Per-side fields, in the order _pair_features takes them.
PAIR_FIELDS = [
    "core_name_compare", "name_suffix_tags", "address_tokens", "address_ascii",
    "postal_code", "house_number", "state_norm",
]

_NGRAM_N = 3
CHUNK_PAIRS = 100_000
CHUNKS_PER_WAVE_PER_WORKER = 4


def _char_ngrams(s: str, n: int = _NGRAM_N) -> set:
    if not s:
        return set()
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _phonetic_code(s: str) -> str:
    """Metaphone of the first 3 tokens, not the whole string: a single
    whole-string phonetic code is unstable under the word-order transposition
    noise this dataset actually has (e.g. 'Gonzalez Beijing Inc' vs
    'Gonzalez Inc Beijing' -- see EDA), since transposing tokens changes the
    entire encoded string even though the same words are present."""
    tokens = [t for t in s.split() if t]
    if not tokens:
        return ""
    return " ".join(sorted(jellyfish.metaphone(t) for t in tokens[:3]))


def _pair_features(name_a, tags_a, addr_tok_a, addr_ascii_a, postal_a, house_a, state_a,
                   name_c, tags_c, addr_tok_c, addr_ascii_c, postal_c, house_c, state_c) -> list:
    jw = JaroWinkler.normalized_similarity(name_a, name_c)
    lev = Levenshtein.normalized_similarity(name_a, name_c)
    ngram_jac = _jaccard(_char_ngrams(name_a), _char_ngrams(name_c))
    tok_jac = _jaccard(set(name_a.split()), set(name_c.split()))
    phon_match = float(_phonetic_code(name_a) == _phonetic_code(name_c) and name_a != "" and name_c != "")
    len_a, len_c = len(name_a), len(name_c)
    len_ratio = min(len_a, len_c) / max(len_a, len_c) if max(len_a, len_c) > 0 else 0.0

    set_tags_a = set(tags_a.split(";")) - {""}
    set_tags_c = set(tags_c.split(";")) - {""}
    suffix_overlap = float(bool(set_tags_a & set_tags_c)) if (set_tags_a or set_tags_c) else 0.0

    addr_jac = _jaccard(set(addr_tok_a.split()), set(addr_tok_c.split()))
    addr_comparable = float(bool(addr_ascii_a) and bool(addr_ascii_c))

    postal_both = bool(postal_a) and bool(postal_c)
    postal_match = float(postal_both and postal_a == postal_c)

    house_both = bool(house_a) and bool(house_c)
    house_match = float(house_both and house_a == house_c)

    state_match = float(bool(state_a) and state_a == state_c)

    return [
        jw, lev, ngram_jac, tok_jac, phon_match, len_ratio, suffix_overlap,
        addr_jac, addr_comparable,
        postal_match, float(postal_both),
        house_match, float(house_both),
        state_match,
    ]


def _features_chunk(columns) -> np.ndarray:
    """Worker entry point: `columns` is the anchor-side PAIR_FIELDS arrays
    followed by the candidate-side ones, all the same length."""
    rows = [_pair_features(*vals) for vals in zip(*columns)]
    return np.asarray(rows, dtype=np.float32).reshape(-1, len(FEATURE_COLS))


def _gather(s1_cols, other_cols, anchor_pos, other_pos):
    return ([s1_cols[f][anchor_pos] for f in PAIR_FIELDS]
            + [other_cols[f][other_pos] for f in PAIR_FIELDS])


def compute_features(anchor_pos: np.ndarray, other_pos: np.ndarray, s1, other,
                      workers: int = None, log=None, n_extra: int = 0) -> np.ndarray:
    """(n_pairs, len(FEATURE_COLS) + n_extra) float32 matrix, row i for pair
    (s1 row anchor_pos[i], other row other_pos[i]). `s1`/`other` are the
    pair_table.load_tables frames. The n_extra trailing columns are left for
    the caller to fill (the embedding feature), so the full matrix is never
    re-allocated. Must run BEFORE anything initializes CUDA in this process
    (the embedding feature): the pool forks on Linux."""
    import multiprocessing as mp

    n = len(anchor_pos)
    n_str = len(FEATURE_COLS)
    out = np.empty((n, n_str + n_extra), dtype=np.float32)
    if n == 0:
        return out
    s1_cols = {f: s1[f].to_numpy(dtype=object) for f in PAIR_FIELDS}
    other_cols = {f: other[f].to_numpy(dtype=object) for f in PAIR_FIELDS}
    workers = workers or os.cpu_count() or 1
    starts = list(range(0, n, CHUNK_PAIRS))
    wave = max(1, workers * CHUNKS_PER_WAVE_PER_WORKER)
    t0 = time.time()
    # The parent's big string arrays are not gc-tracked containers, and
    # freezing the rest keeps forked workers' garbage collection from
    # touching (and so copying) the parent's heap pages.
    gc.collect()
    gc.freeze()
    try:
        with mp.Pool(workers) as pool:
            for w in range(0, len(starts), wave):
                batch = starts[w:w + wave]
                tasks = [_gather(s1_cols, other_cols, anchor_pos[s:s + CHUNK_PAIRS], other_pos[s:s + CHUNK_PAIRS])
                         for s in batch]
                for s, feats in zip(batch, pool.map(_features_chunk, tasks)):
                    out[s:s + len(feats), :n_str] = feats
                del tasks
                if log is not None:
                    done = min(batch[-1] + CHUNK_PAIRS, n)
                    el = time.time() - t0
                    log(f"      features {done:,}/{n:,} pairs, {el:.0f}s elapsed, "
                        f"~{el / done * (n - done):.0f}s remaining")
    finally:
        gc.unfreeze()
    return out


EMBED_BLOCK = 262_144
DOT_CHUNK = 500_000


def embedding_cosine(anchor_pos: np.ndarray, other_pos: np.ndarray, s1, other, embed_model,
                      log=None) -> np.ndarray:
    """float32 cosine similarity per pair between e5 embeddings of raw
    `core_name` -- matching what embedding_blocking.py embeds during
    retrieval. Embeds only the entities that actually occur in the pairs,
    once each, into a preallocated fp16 tensor on the GPU (fp32 on CPU);
    per-pair cosine is then a batched gather + row-wise dot there. No
    per-entity host-side dict or full fp32 host copy of the embeddings."""
    import torch
    import embedding_blocking as eb

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    dim = embed_model.get_sentence_embedding_dimension()

    def embed_unique(positions, frame, label):
        uniq = np.unique(positions)
        texts = frame["core_name"].to_numpy(dtype=object)[uniq]
        table = torch.empty((len(uniq), dim), dtype=dtype, device=device)
        t0 = time.time()
        for s in range(0, len(uniq), EMBED_BLOCK):
            embs = eb.embed_texts(embed_model, texts[s:s + EMBED_BLOCK].tolist())
            table[s:s + len(embs)] = torch.from_numpy(embs).to(device=device, dtype=dtype)
            if log is not None:
                done = min(s + EMBED_BLOCK, len(uniq))
                el = time.time() - t0
                log(f"      embedded {done:,}/{len(uniq):,} {label}, {el:.0f}s elapsed, "
                    f"~{el / done * (len(uniq) - done):.0f}s remaining")
        remap = np.full(len(frame), -1, dtype=np.int64)
        remap[uniq] = np.arange(len(uniq))
        return table, remap

    a_table, a_remap = embed_unique(anchor_pos, s1, "S1 entities")
    c_table, c_remap = embed_unique(other_pos, other, "S2/S3 entities")
    out = np.empty(len(anchor_pos), dtype=np.float32)
    for s in range(0, len(anchor_pos), DOT_CHUNK):
        ia = torch.from_numpy(a_remap[anchor_pos[s:s + DOT_CHUNK]]).to(device)
        ic = torch.from_numpy(c_remap[other_pos[s:s + DOT_CHUNK]]).to(device)
        cos = (a_table[ia].float() * c_table[ic].float()).sum(dim=1)  # both L2-normalized -> dot == cosine
        out[s:s + len(cos)] = cos.cpu().numpy()
    del a_table, c_table
    if device == "cuda":
        torch.cuda.empty_cache()
    return out
