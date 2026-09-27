"""
Generic string-normalization primitives shared by name and address parsing.

Every function takes and returns a pandas Series and is written as vectorized
`.str` calls so it gets GPU acceleration for free under `cudf.pandas`. Row-wise
`.apply` with a Python callback (used only where a vectorized op doesn't exist,
e.g. Unicode-aware token filtering) falls back to CPU under cudf.pandas -- that is
expected and fine; it only touches the minority of steps that truly need it.
"""
import re

import pandas as pd

from config import LITERAL_NULL_RE

_STRAY_SYMBOLS_RE = re.compile(r"#+")
_REPEATED_PUNCT_RE = re.compile(r"([.,;:])\1+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def nfkc_normalize(series: pd.Series) -> pd.Series:
    """Unicode NFKC normalization: folds compatibility/full-width variants and
    composes combining accents, so 'e' + combining-acute and precomposed 'é'
    compare equal. Not vectorizable in cuDF; cudf.pandas transparently runs this
    one on CPU while everything else below stays on GPU."""
    return series.str.normalize("NFKC")


def strip_stray_symbols(series: pd.Series) -> pd.Series:
    s = series.str.replace(_STRAY_SYMBOLS_RE, "", regex=True)
    s = s.str.replace(_REPEATED_PUNCT_RE, r"\1", regex=True)
    return s


def normalize_whitespace(series: pd.Series) -> pd.Series:
    s = series.str.replace(_MULTI_SPACE_RE, " ", regex=True)
    return s.str.strip()


def clean_literal_null_whole(series: pd.Series) -> pd.Series:
    """Blank out a field that, after stripping, is ONLY a null-like token
    ("null", "N/A", "nil", ...). Component-level cleanup for comma-separated
    addresses (e.g. '..., null, ...') is handled in address_parser, which is
    comma-structure aware."""
    is_null_only = series.str.fullmatch(LITERAL_NULL_RE.pattern, case=False, na=False)
    return series.mask(is_null_only, "")


def fold(series: pd.Series) -> pd.Series:
    """Full normalization pipeline for building a comparison-only '_norm' field.
    Callers should keep the original column for display/audit."""
    s = nfkc_normalize(series)
    s = s.str.lower()
    s = strip_stray_symbols(s)
    s = normalize_whitespace(s)
    return s


def strip_accents(series: pd.Series) -> pd.Series:
    """Accent-fold companion transform: NFD-decompose (e -> e + combining-acute)
    then drop everything that doesn't survive an ASCII round-trip, which strips
    the now-detached combining marks. NFKC alone (used by `fold` above) composes
    accents, it doesn't remove them -- EDA found accent corruption as an explicit
    noise operator ("Ínstitute", "PRÍVATE", "schárities.com", "pártners"), and
    without this, those don't exact-match their unaccented counterparts. Also
    gives French region/street names (Émile, Loire-Atlantique, Théâtre, ...)
    something plain-ASCII to match against downstream. Fully vectorized, so this
    stays GPU-eligible under cudf.pandas even though it isn't the identity-named
    'accent stripping' cuDF ships -- it's built from ops cuDF's string API does
    support (normalize, encode, decode)."""
    decomposed = series.str.normalize("NFD")
    return decomposed.str.encode("ascii", errors="ignore").str.decode("ascii")


def is_ascii_series(series: pd.Series) -> pd.Series:
    """True where the string is pure ASCII (char length == byte length).
    Used to flag transliteration / non-Latin-script fields."""
    filled = series.fillna("")
    return filled.str.len() == filled.str.encode("utf-8").str.len()
