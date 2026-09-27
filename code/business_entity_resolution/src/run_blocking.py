"""
Phase B entry point: candidate generation / blocking. Reads the Phase A
*_clean.parquet files, runs all blocking channels per country partition,
unions and prunes them via meta_blocking, and writes candidate_pairs.tsv in
the exact submission format -- this file is the last stage before a matching
model scores pairs, per the challenge README, so every S1 entity must appear
exactly once (empty string when no candidates survived), IDs must be
comma-separated with no duplicates, and only S2-/S3-prefixed IDs are valid.

Country partition is a validated-safe hard block (zero cross-country matches
found in the EDA); each country is processed independently and in isolation,
which is also what bounds memory at this record count.

Run under `python -m cudf.pandas run_blocking.py ...` for the same reason as
Phase A: everything here is written against the plain pandas API. In
practice the heaviest per-country step (sparse TF-IDF retrieval) is CPU-bound
scipy/sklearn regardless -- cudf.pandas does not accelerate scipy.sparse or
sklearn calls, only pandas ones -- so the GPU benefit here is smaller than in
Phase A; it's kept for consistency and because reading/writing the parquet
files at this scale IS a pandas operation that benefits.
"""
import argparse
import gc
import os
import sys
import time

# Same fix Phase A's run_preprocessing.py needed: cudf.pandas loads this file
# via runpy.run_path(), which does not add the script's own directory to
# sys.path the way `python script.py` does, so the bare sibling imports below
# would raise ModuleNotFoundError unless invoked from this exact directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import embedding_blocking
import exact_blocking
import meta_blocking
import tfidf_blocking
import transliteration
from io_utils import read_parquet

COUNTRIES = ["US", "India", "France"]
CANDIDATE_BUDGET = 30  # upper end of the report's 20-30/anchor recommendation


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_blocking_text(df: pd.DataFrame) -> pd.DataFrame:
    """core_name_blocking: romanized for India rows (so the n-gram/phonetic
    channels have Latin text to compare against the English-spelled S1
    anchors), core_name as-is otherwise. A new column -- core_name itself is
    left untouched for the exact-match and embedding channels."""
    out = df.copy()
    out["core_name_blocking"] = out["core_name"]
    india_mask = out["country"] == "India"
    if india_mask.any():
        out.loc[india_mask, "core_name_blocking"] = transliteration.romanized_series(
            out.loc[india_mask, "core_name"]
        )
    return out


def block_one_target(s1_c: pd.DataFrame, target_c: pd.DataFrame, tfidf_vec, k: int,
                      embed_model=None, s1_embeddings=None):
    """Runs the TF-IDF, exact-match, and (if enabled) embedding channels for
    one (country, target source) pair, returns (anchor_idx, candidate_idx,
    weight) edges pruned to budget k. `s1_embeddings` is precomputed once per
    country and passed in, since it's shared across both target sources."""
    if len(s1_c) == 0 or len(target_c) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=np.float32)

    s1_mat = tfidf_vec.transform(s1_c["core_name_blocking"])
    target_mat = tfidf_vec.transform(target_c["core_name_blocking"])
    tfidf_edges = tfidf_blocking.top_k_per_anchor(s1_mat, target_mat, k)

    exact_df = exact_blocking.exact_match_candidates(s1_c, target_c)
    if len(exact_df):
        exact_edges = (
            exact_df["anchor_row"].to_numpy(dtype=np.int64),
            exact_df["candidate_row"].to_numpy(dtype=np.int64),
            np.ones(len(exact_df), dtype=np.float32),
        )
    else:
        empty = np.array([], dtype=np.int64)
        exact_edges = (empty, empty, np.array([], dtype=np.float32))

    channels = [tfidf_edges, exact_edges]
    if embed_model is not None:
        target_embeddings = embedding_blocking.embed_texts(embed_model, target_c["core_name"].tolist())
        index = embedding_blocking.build_ann_index(target_embeddings)
        channels.append(embedding_blocking.top_k_per_anchor(index, s1_embeddings, k))

    merged = meta_blocking.union_edges(channels, len(s1_c), len(target_c))
    return meta_blocking.prune_to_budget(merged, k)


def run_country(country: str, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, k: int,
                 embed_model=None) -> dict:
    """Returns {s1_entity_id: set(candidate_entity_id)} for this country."""
    s1_c = s1[s1["country"] == country].reset_index(drop=True)
    s2_c = s2[s2["country"] == country].reset_index(drop=True)
    s3_c = s3[s3["country"] == country].reset_index(drop=True)
    result = {eid: set() for eid in s1_c["entity_id"]}
    if len(s1_c) == 0:
        return result

    log(f"[{country}] s1={len(s1_c):,} s2={len(s2_c):,} s3={len(s3_c):,} -- adding blocking text...")
    s1_c = add_blocking_text(s1_c)
    s2_c = add_blocking_text(s2_c)
    s3_c = add_blocking_text(s3_c)

    all_text = pd.concat([s1_c["core_name_blocking"], s2_c["core_name_blocking"], s3_c["core_name_blocking"]])
    log(f"[{country}] fitting shared TF-IDF vectorizer on {len(all_text):,} texts...")
    tfidf_vec = tfidf_blocking.fit_vectorizer(all_text)
    log(f"[{country}] vocab size={len(tfidf_vec.vocabulary_):,}")
    del all_text
    gc.collect()

    s1_embeddings = None
    if embed_model is not None:
        log(f"[{country}] embedding {len(s1_c):,} S1 anchors...")
        s1_embeddings = embedding_blocking.embed_texts(embed_model, s1_c["core_name"].tolist())

    for tag, target_c in (("S2", s2_c), ("S3", s3_c)):
        if len(target_c) == 0:
            continue
        log(f"[{country}] blocking against {tag} ({len(target_c):,} candidates)...")
        t0 = time.time()
        a_idx, c_idx, _ = block_one_target(s1_c, target_c, tfidf_vec, k,
                                            embed_model=embed_model, s1_embeddings=s1_embeddings)
        log(f"[{country}] {tag}: {len(a_idx):,} candidate edges in {time.time() - t0:.1f}s")
        s1_ids = s1_c["entity_id"].to_numpy()
        cand_ids = target_c["entity_id"].to_numpy()
        for ai, ci in zip(a_idx, c_idx):
            result[s1_ids[ai]].add(cand_ids[ci])

    return result


def write_candidate_pairs(all_s1_ids, candidate_map: dict, out_path: str):
    """One row per S1 entity (ALL of them, per the format spec), comma-
    joined candidate IDs, empty string when none survived blocking."""
    rows = []
    for eid in all_s1_ids:
        cands = candidate_map.get(eid, set())
        rows.append((eid, ",".join(sorted(cands))))
    df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    df.to_csv(out_path, sep="\t", index=False)
    log(f"wrote {out_path} ({len(df):,} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True, help="Phase A output dir (*_clean.parquet)")
    ap.add_argument("--output-dir", required=True, help="where candidate_pairs.tsv is written")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--k", type=int, default=CANDIDATE_BUDGET)
    ap.add_argument("--use-embeddings", action="store_true",
                     help="enable the embedding-ANN channel (heaviest compute item -- "
                          "run scripts/probe_embedding_throughput.py first to size it)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.split

    embed_model = None
    if args.use_embeddings:
        log(f"loading embedding model {embedding_blocking.EMBEDDING_MODEL_NAME}...")
        embed_model = embedding_blocking.load_model()

    log(f"loading {prefix} parquet files...")
    s1 = read_parquet(os.path.join(args.processed_dir, f"{prefix}_source1_clean.parquet"))
    s2 = read_parquet(os.path.join(args.processed_dir, f"{prefix}_source2_clean.parquet"))
    s3 = read_parquet(os.path.join(args.processed_dir, f"{prefix}_source3_clean.parquet"))
    log(f"loaded s1={len(s1):,} s2={len(s2):,} s3={len(s3):,}")

    all_s1_ids = s1["entity_id"].tolist()
    candidate_map = {}
    for country in COUNTRIES:
        t0 = time.time()
        country_result = run_country(country, s1, s2, s3, args.k, embed_model=embed_model)
        candidate_map.update(country_result)
        log(f"[{country}] done in {time.time() - t0:.1f}s")
        gc.collect()

    out_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    write_candidate_pairs(all_s1_ids, candidate_map, out_path)


if __name__ == "__main__":
    main()
