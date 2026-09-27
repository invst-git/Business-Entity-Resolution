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
"""
import pandas as pd


def postal_code_candidates(anchors: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Anchors and candidates joined on exact non-empty postal_code.
    Returns columns [anchor_row, candidate_row]."""
    a = anchors[anchors["postal_code"] != ""][["postal_code"]].reset_index().rename(columns={"index": "anchor_row"})
    c = candidates[candidates["postal_code"] != ""][["postal_code"]].reset_index().rename(columns={"index": "candidate_row"})
    if a.empty or c.empty:
        return pd.DataFrame(columns=["anchor_row", "candidate_row"])
    merged = a.merge(c, on="postal_code", how="inner")
    return merged[["anchor_row", "candidate_row"]]


def house_state_candidates(anchors: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Anchors and candidates joined on exact (house_number, state_norm), both
    non-empty. Weaker signal alone than postal code but has far higher
    coverage (postal codes are rare in this data; house numbers are not)."""
    key_cols = ["house_number", "state_norm"]
    a = anchors[(anchors["house_number"] != "") & (anchors["state_norm"] != "")][key_cols].reset_index().rename(columns={"index": "anchor_row"})
    c = candidates[(candidates["house_number"] != "") & (candidates["state_norm"] != "")][key_cols].reset_index().rename(columns={"index": "candidate_row"})
    if a.empty or c.empty:
        return pd.DataFrame(columns=["anchor_row", "candidate_row"])
    merged = a.merge(c, on=key_cols, how="inner")
    return merged[["anchor_row", "candidate_row"]]


def exact_match_candidates(anchors: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Union of both exact-match sub-channels, deduplicated."""
    postal = postal_code_candidates(anchors, candidates)
    house = house_state_candidates(anchors, candidates)
    combined = pd.concat([postal, house], ignore_index=True)
    return combined.drop_duplicates()
