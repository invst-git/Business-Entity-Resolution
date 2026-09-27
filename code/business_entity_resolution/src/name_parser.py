"""
Business-name parsing: rename/DBA markers, honorific prefixes, legal-suffix
extraction -> a comparison-ready `core_name`.

Order matters and was chosen from qualitative EDA on train_ground_truth matched
groups (see the EDA report): every observed "fka X" / "formerly X" / "trading as X"
record had its TRUE match on the text AFTER the marker, with the text before it
being an unrelated decoy name (e.g. "Pyrakor fka Gonzalez Beijing Inc" truly
matches "Gonzalez Beijing Inc", not "Pyrakor"). So the post-marker segment, not the
raw field, is what gets carried into suffix/honorific stripping.
"""
import numpy as np
import pandas as pd

import text_normalize as tn
from config import HONORIFIC_RE, LEGAL_SUFFIX_COMPILED, RENAME_MARKER_RE


def split_rename_marker(name_series: pd.Series):
    parts = name_series.str.split(RENAME_MARKER_RE.pattern, n=1, expand=True, regex=True)
    if parts.shape[1] == 1:
        had_marker = pd.Series(False, index=name_series.index)
        working = name_series
        decoy_prefix = pd.Series("", index=name_series.index)
    else:
        had_marker = parts[1].notna()
        working = tn.normalize_whitespace(parts[1].fillna(name_series))
        decoy_prefix = parts[0].where(had_marker, "").fillna("")
    return working, had_marker, decoy_prefix


def strip_honorific(name_series: pd.Series):
    honorific_flag = name_series.str.contains(HONORIFIC_RE, regex=True)
    stripped = name_series.str.replace(HONORIFIC_RE, "", regex=True)
    return tn.normalize_whitespace(stripped), honorific_flag.fillna(False)


def extract_legal_suffixes(name_series: pd.Series):
    working = name_series
    suffix_tags = pd.Series("", index=name_series.index)
    for tag, pattern in LEGAL_SUFFIX_COMPILED:
        matched = working.str.contains(pattern, regex=True).fillna(False)
        addition = np.where(matched, tag + ";", "")
        suffix_tags = suffix_tags + addition
        working = working.str.replace(pattern, " ", regex=True)
    working = tn.normalize_whitespace(working)
    suffix_tags = suffix_tags.str.rstrip(";")
    return working, suffix_tags


def parse_names(df: pd.DataFrame, name_col: str = "business_name") -> pd.DataFrame:
    out = df.copy()
    folded = tn.fold(out[name_col])
    folded = tn.clean_literal_null_whole(folded)

    working, had_marker, decoy_prefix = split_rename_marker(folded)
    stripped, honorific_flag = strip_honorific(working)
    core_name, suffix_tags = extract_legal_suffixes(stripped)

    out["name_is_ascii"] = tn.is_ascii_series(out[name_col])
    out["name_had_rename_marker"] = had_marker
    out["name_decoy_prefix"] = decoy_prefix
    out["name_honorific_flag"] = honorific_flag
    out["name_suffix_tags"] = suffix_tags
    out["core_name"] = core_name
    out["core_name_ascii"] = tn.strip_accents(core_name)
    return out
