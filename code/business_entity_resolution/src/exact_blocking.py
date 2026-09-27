"""
Exact-match blocking channel: postal code and house-number+state agreement.
Narrow and high-precision by construction (an exact postal-code or
house-number match is strong evidence on its own), and cheap: this is a
plain merge/join, not a similarity computation, mirroring the pattern from
the blocking-methodology research where a small exact-identifier channel
unioned with a broader semantic channel recovered true matches the semantic
channel missed entirely, at negligible added candidate volume since exact-
match blocks are inherently small.

Two keys, not one, because either alone is weak: postal codes are present in
only ~6.5% of US addresses and <1% of India addresses (EDA finding), while
house numbers are more often present but far less selective alone (many
businesses share a common house number). Combining "same house number AND
same state" is a meaningfully more selective join key than either alone,
without requiring a full address match (which the confirmed component-
reordering noise pattern makes unreliable as an exact key).

Every key is block-purged: a key shared by more than MAX_KEY_BLOCK records
on either side is dropped before joining. Without this, the join emits the
full product for common keys -- house number "100" in California alone can
sit on hundreds of anchors and thousands of candidates -- which at full
scale is plausibly hundreds of millions of rows for the US partition. A key
shared by that many records isn't exact evidence of anything anyway.
"""
import pandas as pd

MAX_KEY_BLOCK = 50


def _purged_join(anchors: pd.DataFrame, candidates: pd.DataFrame, key_cols: list,
                 max_key_block: int) -> pd.DataFrame:
    a = anchors[key_cols].reset_index().rename(columns={"index": "anchor_row"})
    c = candidates[key_cols].reset_index().rename(columns={"index": "candidate_row"})
    for col in key_cols:
        a = a[a[col] != ""]
        c = c[c[col] != ""]
    if a.empty or c.empty:
        return pd.DataFrame(columns=["anchor_row", "candidate_row"])
    a_sizes = a.groupby(key_cols).size()
    c_sizes = c.groupby(key_cols).size()
    ok_keys = a_sizes[a_sizes <= max_key_block].index.intersection(c_sizes[c_sizes <= max_key_block].index)
    if len(ok_keys) == 0:
        return pd.DataFrame(columns=["anchor_row", "candidate_row"])
    a = a.set_index(key_cols).loc[a.set_index(key_cols).index.isin(ok_keys)].reset_index()
    c = c.set_index(key_cols).loc[c.set_index(key_cols).index.isin(ok_keys)].reset_index()
    merged = a.merge(c, on=key_cols, how="inner")
    return merged[["anchor_row", "candidate_row"]]


def postal_code_candidates(anchors: pd.DataFrame, candidates: pd.DataFrame,
                            max_key_block: int = MAX_KEY_BLOCK) -> pd.DataFrame:
    """Anchors and candidates joined on exact non-empty postal_code."""
    return _purged_join(anchors, candidates, ["postal_code"], max_key_block)


def house_state_candidates(anchors: pd.DataFrame, candidates: pd.DataFrame,
                            max_key_block: int = MAX_KEY_BLOCK) -> pd.DataFrame:
    """Anchors and candidates joined on exact (house_number, state_norm)."""
    return _purged_join(anchors, candidates, ["house_number", "state_norm"], max_key_block)


def exact_match_candidates(anchors: pd.DataFrame, candidates: pd.DataFrame,
                            max_key_block: int = MAX_KEY_BLOCK) -> pd.DataFrame:
    """Union of both exact-match sub-channels, deduplicated."""
    postal = postal_code_candidates(anchors, candidates, max_key_block)
    house = house_state_candidates(anchors, candidates, max_key_block)
    combined = pd.concat([postal, house], ignore_index=True)
    return combined.drop_duplicates()
