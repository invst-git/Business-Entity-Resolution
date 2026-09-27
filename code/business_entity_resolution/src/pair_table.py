"""
Integer-position representation of candidate pairs and ground truth, so
Phase C scales to the full candidate set (~100M+ pairs at k=30 per target
source across ~2M anchors).

The first Phase C draft kept one Python string per pair side and one Python
dict per entity (to_dict("records")). That was fine on the 15K-row sample
but at full scale it is tens of GB of Python objects before a single feature
is computed. Here every entity is identified by its row position in its
table, pairs are two int32 arrays, and string ids exist only transiently
while candidate_pairs.tsv / the ground truth are parsed chunk by chunk.

S2 and S3 are concatenated into ONE "other" table (their id prefixes are
disjoint), so a candidate is a single position regardless of source.
"""
import os

import numpy as np
import pandas as pd

from io_utils import read_parquet
from transliteration import add_compare_name

# Columns the pairwise features read. Anything else from Phase A is dropped
# at load time to bound memory.
TABLE_COLS = [
    "entity_id", "core_name_compare", "core_name", "address_tokens", "address_ascii",
    "postal_code", "house_number", "state_norm", "name_suffix_tags",
]
PARSE_CHUNK_ROWS = 200_000


def load_tables(processed_dir: str, split: str):
    """Returns (s1, other) DataFrames restricted to TABLE_COLS, each with a
    fresh RangeIndex -- the row position IS the entity's integer id."""
    frames = []
    for src in ("source1", "source2", "source3"):
        df = read_parquet(os.path.join(processed_dir, f"{split}_{src}_clean.parquet"))
        df = add_compare_name(df)
        frames.append(df[TABLE_COLS].reset_index(drop=True))
    s1 = frames[0]
    other = pd.concat(frames[1:], ignore_index=True)
    return s1, other


def _ids_to_positions(ids, index: pd.Index, what: str) -> np.ndarray:
    pos = index.get_indexer(ids)
    n_missing = int((pos < 0).sum())
    if n_missing:
        raise ValueError(f"{n_missing:,} {what} ids not found in the processed tables "
                         f"(e.g. {list(pd.Series(ids)[pos < 0][:3])})")
    return pos.astype(np.int32)


def _explode(anchor_ids: pd.Series, joined_ids: pd.Series):
    """(anchor id, 'a,b,c') rows -> parallel (anchor id, candidate id) arrays."""
    non_empty = joined_ids != ""
    anchor_ids, joined_ids = anchor_ids[non_empty], joined_ids[non_empty]
    if len(joined_ids) == 0:
        empty = np.array([], dtype=object)
        return empty, empty
    lists = joined_ids.str.split(",")
    counts = lists.str.len().to_numpy()
    return np.repeat(anchor_ids.to_numpy(), counts), np.concatenate(lists.to_list())


def parse_id_lists(df: pd.DataFrame, anchor_col: str, list_col: str,
                   s1_index: pd.Index, other_index: pd.Index, what: str):
    """Parses a (source1_entity_id, comma-joined ids) table -- either
    candidate_pairs.tsv or the ground truth -- into int32 position arrays
    (anchor_pos, other_pos), in chunks so the exploded string form never
    exists for the whole file at once."""
    a_parts, c_parts = [], []
    for start in range(0, len(df), PARSE_CHUNK_ROWS):
        chunk = df.iloc[start:start + PARSE_CHUNK_ROWS]
        a_ids, c_ids = _explode(chunk[anchor_col], chunk[list_col])
        if len(a_ids) == 0:
            continue
        a_parts.append(_ids_to_positions(a_ids, s1_index, f"{what} anchor"))
        c_parts.append(_ids_to_positions(c_ids, other_index, f"{what} candidate"))
    if not a_parts:
        empty = np.array([], dtype=np.int32)
        return empty, empty
    return np.concatenate(a_parts), np.concatenate(c_parts)


def pair_keys(anchor_pos: np.ndarray, other_pos: np.ndarray, n_other: int) -> np.ndarray:
    return anchor_pos.astype(np.int64) * np.int64(n_other) + other_pos.astype(np.int64)


class Truth:
    """Ground truth in position form: the set of true (anchor, other) pair
    keys, the true-match count per anchor (for recall, including matches
    blocking missed), and the list of anchors the ground truth covers
    (every S1 entity, singletons included)."""

    def __init__(self, gt_df: pd.DataFrame, s1_index: pd.Index, other_index: pd.Index):
        self.anchors = _ids_to_positions(gt_df["source1_entity_id"].to_numpy(), s1_index, "ground-truth anchor")
        a_pos, c_pos = parse_id_lists(gt_df, "source1_entity_id", "matched_entity_ids",
                                      s1_index, other_index, "ground-truth")
        self.n_other = len(other_index)
        self.true_keys = np.unique(pair_keys(a_pos, c_pos, self.n_other))
        self.n_true = np.bincount(a_pos, minlength=len(s1_index)).astype(np.int32)

    def label(self, anchor_pos: np.ndarray, other_pos: np.ndarray) -> np.ndarray:
        return np.isin(pair_keys(anchor_pos, other_pos, self.n_other), self.true_keys).astype(np.int8)

    def recall(self, anchor_pos: np.ndarray, label: np.ndarray, anchors: np.ndarray = None) -> dict:
        """Blocking recall ceiling over `anchors` (default: every ground-
        truth anchor). pair_recall = true pairs retrieved / all true pairs;
        full_set_recall = share of anchors with >=1 true match whose ENTIRE
        true set was retrieved (the matcher can't recover the rest)."""
        anchors = self.anchors if anchors is None else anchors
        found = np.bincount(anchor_pos, weights=label, minlength=len(self.n_true))[anchors]
        nt = self.n_true[anchors]
        has = nt > 0
        return {
            "anchors": int(len(anchors)),
            "anchors_with_matches": int(has.sum()),
            "pair_recall": float(found.sum() / max(nt.sum(), 1)),
            "full_set_recall": float((found[has] == nt[has]).mean()) if has.any() else 0.0,
        }
