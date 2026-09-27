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
"""
import numpy as np
import pandas as pd
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

LOOKUP_COLS = [
    "entity_id", "core_name_compare", "core_name", "address_tokens", "address_ascii",
    "postal_code", "house_number", "state_norm", "name_suffix_tags",
]

_NGRAM_N = 3


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


def build_lookup(df: pd.DataFrame) -> dict:
    """entity_id -> dict of the fields features need, for O(1) access per
    pair instead of repeated DataFrame indexing (which is what made
    vocab_builder catastrophically slow earlier in this project -- see
    vocab_builder.py's docstring). Call once per source file."""
    cols = [c for c in LOOKUP_COLS if c in df.columns]
    records = df[cols].to_dict("records")
    return {r["entity_id"]: r for r in records}


def explode_candidate_pairs(candidate_pairs_df: pd.DataFrame) -> pd.DataFrame:
    """candidate_pairs.tsv, as read (columns source1_entity_id,
    candidate_entity_ids, comma-joined) -> one row per (anchor, candidate)
    pair. Rows with no candidates are dropped (nothing to score)."""
    non_empty = candidate_pairs_df[candidate_pairs_df["candidate_entity_ids"] != ""]
    exploded = non_empty.assign(
        candidate_entity_id=non_empty["candidate_entity_ids"].str.split(",")
    ).explode("candidate_entity_id")
    return exploded[["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)


def _one_pair_features(a: dict, c: dict) -> list:
    name_a, name_c = a["core_name_compare"], c["core_name_compare"]
    jw = JaroWinkler.normalized_similarity(name_a, name_c)
    lev = Levenshtein.normalized_similarity(name_a, name_c)
    ngram_jac = _jaccard(_char_ngrams(name_a), _char_ngrams(name_c))
    tok_a, tok_c = set(name_a.split()), set(name_c.split())
    tok_jac = _jaccard(tok_a, tok_c)
    phon_match = float(_phonetic_code(name_a) == _phonetic_code(name_c) and name_a != "" and name_c != "")
    len_a, len_c = len(name_a), len(name_c)
    len_ratio = min(len_a, len_c) / max(len_a, len_c) if max(len_a, len_c) > 0 else 0.0

    tags_a = set(a["name_suffix_tags"].split(";")) - {""}
    tags_c = set(c["name_suffix_tags"].split(";")) - {""}
    suffix_overlap = float(bool(tags_a & tags_c)) if (tags_a or tags_c) else 0.0

    addr_a_tok = set(a["address_tokens"].split())
    addr_c_tok = set(c["address_tokens"].split())
    addr_jac = _jaccard(addr_a_tok, addr_c_tok)
    addr_comparable = float(bool(a["address_ascii"]) and bool(c["address_ascii"]))

    postal_both = bool(a["postal_code"]) and bool(c["postal_code"])
    postal_match = float(postal_both and a["postal_code"] == c["postal_code"])

    house_both = bool(a["house_number"]) and bool(c["house_number"])
    house_match = float(house_both and a["house_number"] == c["house_number"])

    state_match = float(bool(a["state_norm"]) and a["state_norm"] == c["state_norm"])

    return [
        jw, lev, ngram_jac, tok_jac, phon_match, len_ratio, suffix_overlap,
        addr_jac, addr_comparable,
        postal_match, float(postal_both),
        house_match, float(house_both),
        state_match,
    ]


def compute_features(pairs_df: pd.DataFrame, s1_lookup: dict, other_lookup: dict) -> pd.DataFrame:
    """pairs_df: [source1_entity_id, candidate_entity_id]. other_lookup must
    contain BOTH S2 and S3 entities (merge the two lookups before calling, or
    call once per target source and concat results). Rows whose anchor or
    candidate is missing from the lookups are dropped (should not happen for
    correctly-sourced input, guarded rather than silently producing garbage
    features)."""
    feature_rows = []
    keep_mask = []
    for s1_id, cand_id in zip(pairs_df["source1_entity_id"], pairs_df["candidate_entity_id"]):
        a = s1_lookup.get(s1_id)
        c = other_lookup.get(cand_id)
        if a is None or c is None:
            keep_mask.append(False)
            continue
        keep_mask.append(True)
        feature_rows.append(_one_pair_features(a, c))

    kept = pairs_df[pd.Series(keep_mask, index=pairs_df.index)].reset_index(drop=True)
    feat_df = pd.DataFrame(feature_rows, columns=FEATURE_COLS)
    return pd.concat([kept, feat_df], axis=1)


def add_embedding_feature(feat_df: pd.DataFrame, s1_lookup: dict, other_lookup: dict,
                           embed_model) -> pd.DataFrame:
    """Adds an `embedding_cosine` column. Embeds raw `core_name` (NOT
    core_name_compare) for the unique anchor/candidate texts appearing in
    feat_df -- matching what embedding_blocking.py embeds during retrieval,
    the same "one source of truth" reasoning as core_name_compare. Only
    embeds the unique texts actually present in this candidate set (a small
    fraction of the full corpus), not the whole corpus again."""
    import embedding_blocking as eb

    unique_s1 = feat_df["source1_entity_id"].unique()
    unique_other = feat_df["candidate_entity_id"].unique()

    s1_texts = [s1_lookup[i]["core_name"] for i in unique_s1]
    other_texts = [other_lookup[i]["core_name"] for i in unique_other]

    s1_embs = eb.embed_texts(embed_model, s1_texts)
    other_embs = eb.embed_texts(embed_model, other_texts)

    s1_emb_map = dict(zip(unique_s1, s1_embs))
    other_emb_map = dict(zip(unique_other, other_embs))

    cos = [
        float(np.dot(s1_emb_map[s1_id], other_emb_map[cid]))  # both L2-normalized -> dot == cosine
        for s1_id, cid in zip(feat_df["source1_entity_id"], feat_df["candidate_entity_id"])
    ]
    out = feat_df.copy()
    out["embedding_cosine"] = cos
    return out
