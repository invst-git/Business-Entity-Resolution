"""
IO helpers. Written against the `pandas` API only (no direct `cudf` import) so the
whole pipeline runs unmodified on CPU pandas or, on qBraid, GPU-accelerated via the
`cudf.pandas` proxy (`python -m cudf.pandas run_preprocessing.py ...`). See
scripts/setup_env.sh and the README for how that proxy is enabled.
"""
import pandas as pd

TSV_DTYPE = str


def read_tsv(path, columns=None, nrows=None):
    """Read a challenge TSV with literal empty-string semantics preserved.

    keep_default_na=False is required: pandas' default na_values list includes
    "NULL", "null", "None", "N/A", "NA", "n/a" etc. Silently turning those into NaN
    on read would destroy exactly the literal-"null"-string noise (~131K rows in
    each of train_source2/3) that text_normalize.clean_literal_nulls is meant to
    detect and clean. Reading as plain strings keeps that signal visible.

    `nrows`, when given, is passed straight to pandas so a `--sample` smoke test
    actually skips reading the rest of a 500MB+ file -- reading in full and then
    truncating in memory (e.g. via `.head()`) pays the same disk/parse cost either
    way and defeats the point of a fast sanity check.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=TSV_DTYPE,
        keep_default_na=False,
        na_filter=False,
        usecols=columns,
        nrows=nrows,
    )
    return df


def write_parquet(df, path):
    df.to_parquet(path, index=False)


def read_parquet(path, columns=None):
    """`columns`, when given, is passed straight to pandas so reading just the
    slim entity_id/country/state_raw columns for vocab-building doesn't
    deserialize every other column of a multi-million-row parquet file --
    genuinely valuable here since parquet is columnar and can skip the rest
    at read time, unlike a TSV."""
    return pd.read_parquet(path, columns=columns)
