"""
Address parsing. Addresses are treated as an unordered bag of comma-separated
components (EDA confirmed component reordering is a real, frequent noise pattern,
e.g. "NC, DUNN, 241 BRINKLEY RD" vs "241 Brinkley Road, Dunn, NC") -- so this module
never assumes a fixed field position except as a fallback heuristic for house
number (usually still leads) and state (usually still trails).

The per-row logic (variable comma-count, conditional regex) doesn't vectorize
cleanly in cuDF, so it runs as a single combined `.apply` pass per file (one Python
pass over ~2-5M rows, not five) -- this is a case where cudf.pandas is expected to
fall back to CPU. The bulk NFKC/case/whitespace normalization before it, and the
street-abbreviation expansion after it, ARE vectorized `.str` ops and get real GPU
acceleration under cudf.pandas.
"""
import re

import pandas as pd

import text_normalize as tn
from config import (
    COUNTRY_STATE_TABLES,
    HOUSE_NUMBER_TOKEN_RE,
    IN_PIN_RE,
    LITERAL_NULL_RE,
    STREET_ABBREV_BY_COUNTRY,
    US_ZIP_RE,
)

# \w+ (Python's re is Unicode-aware by default) rather than [a-z0-9]+: the ASCII
# class produced an EMPTY token bag for any address written entirely in a native
# script (Devanagari/Tamil/...), since none of those characters are in a-z0-9.
# \w+ still matches Latin+digits identically, and picks up non-Latin letters too
# -- it fragments some conjunct scripts at internal combining marks rather than
# returning whole "words", but fragments are still consistent tokens: two
# records in the same script produce the same fragments, which is all a token-
# overlap blocking signal needs. Non-empty and consistent beats empty.
_TOKEN_RE = re.compile(r"\w+")

# Precomputed once at import time, not per-row: rebuilding this set on every one
# of 5M+ row calls was the dominant cost of _extract_state (measured ~4x slowdown
# before this fix).
_KNOWN_STATE_TOKENS = {
    country: ({k.lower() for k in table} | {v.lower() for v in table.values()})
    for country, table in COUNTRY_STATE_TABLES.items()
}


def _parse_components(addr: str):
    if not addr:
        return [], ""
    raw_parts = [p.strip() for p in addr.split(",")]
    parts = [p for p in raw_parts if p and not LITERAL_NULL_RE.fullmatch(p)]
    cleaned = ", ".join(parts)
    return parts, cleaned


def _is_foreign_script(s: str) -> bool:
    """True for a genuinely non-Latin script (Devanagari, Tamil, ...), false for
    plain or accented Latin text (French "Émile", "Loire-Atlantique", ...).
    Plain isascii() conflates the two -- accented Latin is non-ASCII too -- which
    made tier 3 below misfire on French addresses (every accented component
    looked "foreign", so it grabbed a street name instead of the actual
    region/departement). U+0250 is the start of IPA Extensions, safely above the
    Latin blocks (Latin-1 Supplement, Latin Extended-A/B) that cover every
    accented character seen in the training/test data's Latin-script countries.
    """
    return any(ord(ch) >= 0x250 for ch in s)


def _extract_state(parts, country, native_map):
    """Component reordering (a confirmed noise pattern -- see EDA) means the state
    isn't reliably the last component, so this scans all components rather than
    indexing positionally. Tiers, most confident first:
      1. exact match against the hardcoded state/abbreviation table for `country`
      2. exact match against `native_map` -- despite the name this covers two
         distinct learned mappings, merged by the caller: the native-script ->
         Latin state map (India, from train_ground_truth) and the frequency-based
         locality vocabulary (France, from test_source1 itself, mapped identity
         token->token since no script translation is needed there)
      3. any foreign-script component (best-effort fallback when tiers 1-2 miss a
         genuine non-Latin state name: see the EDA finding that ~80% of
         non-ASCII-name addresses are otherwise fully ASCII/Latin, so an
         unrecognized foreign-script component is very likely the state)
      4. positional last-resort (previous naive behavior)
    Scans from the end since state still trails in the common case, so ties
    between equally-plausible components resolve in favor of the later one.
    """
    if not parts:
        return ""
    known = _KNOWN_STATE_TOKENS.get(country, frozenset())
    for comp in reversed(parts):
        if comp in known:  # `parts` is already lowercased by the tn.fold() upstream
            return comp
    if native_map:
        for comp in reversed(parts):
            if comp in native_map:
                return comp
    for comp in reversed(parts):
        if _is_foreign_script(comp):
            return comp
    return parts[-1]


def _extract_house_number(parts):
    for comp in parts[:2]:
        m = HOUSE_NUMBER_TOKEN_RE.search(comp)
        if m:
            token = m.group(0)
            is_range = "-" in token
            canonical = token.split("-")[0].lstrip("0") or "0"
            return canonical, is_range
    return "", False


def _extract_postal(cleaned, country, house_number):
    pattern = IN_PIN_RE if country == "India" else US_ZIP_RE
    matches = [m.group(0) for m in pattern.finditer(cleaned)]
    matches = [m for m in matches if m.split("-")[0] != house_number]
    if not matches:
        return "", ""
    postal_type = "IN_PIN" if country == "India" else ("FR_POSTAL" if country == "France" else "US_ZIP")
    return matches[-1], postal_type


def _parse_row(addr, country, native_map):
    parts, cleaned = _parse_components(addr)
    house_number, is_range = _extract_house_number(parts)
    postal_code, postal_type = _extract_postal(cleaned, country, house_number)
    state_raw = _extract_state(parts, country, native_map)
    return cleaned, house_number, is_range, postal_code, postal_type, state_raw


def tokenize_series(series: pd.Series) -> pd.Series:
    """Vectorized dedupe-sorted token bag. Deliberately built from the POST-
    street-abbreviation-expansion address (address_expanded), not the raw cleaned
    one -- tokenizing before expansion would leave the "rd" vs "road" mismatch
    the expansion step exists to remove still present in the token bag blocking
    is meant to consume."""
    token_lists = series.str.findall(_TOKEN_RE)
    return token_lists.apply(lambda toks: " ".join(sorted(set(toks))))


def expand_street_abbreviations(series: pd.Series, country_series: pd.Series) -> pd.Series:
    out = series.copy()
    for country, mapping in STREET_ABBREV_BY_COUNTRY.items():
        mask = country_series == country
        if not mask.any():
            continue
        sub = out[mask]
        for abbrev, full in mapping.items():
            sub = sub.str.replace(rf"\b{abbrev}\b", full, regex=True, case=False)
        out[mask] = sub
    return out


def _canonical_map_for(country: str) -> dict:
    """abbrev -> full and full -> full, both lowercase, so any variant maps to the
    canonical lowercase full state/region name in one dict lookup. Every table in
    COUNTRY_STATE_TABLES is {ABBREV: full_lowercase} -- see config.py's note on
    why this direction must stay consistent across countries."""
    table = COUNTRY_STATE_TABLES.get(country, {})
    canon = {}
    for abbrev, full in table.items():
        canon[abbrev.lower()] = full
        canon[full.lower()] = full
    return canon


def state_confident_hit_series(state_raw: pd.Series, country_series: pd.Series, native_map: dict) -> pd.Series:
    """True where state_raw resolved via tier 1 (hardcoded table) or tier 2
    (learned map) rather than falling through to the tier-3/4 guesses. For QA:
    `state_norm` is non-empty for nearly any row with an address at all (tier 4
    always returns the last raw component), so "state_resolved" is not a
    meaningful signal on its own -- this is."""
    out = pd.Series(False, index=state_raw.index)
    for country, known in _KNOWN_STATE_TOKENS.items():
        mask = country_series == country
        if mask.any():
            out[mask] = state_raw[mask].isin(known)
    if native_map:
        out |= state_raw.isin(native_map.keys())
    return out


def normalize_state_series(state_raw: pd.Series, country_series: pd.Series, native_map: dict) -> pd.Series:
    """Vectorized per-country-group state canonicalization (map(), not apply()).
    Falls back to the (already-folded, lowercase) raw token when no hardcoded or
    learned mapping resolves it -- still usable as a blocking key, just not
    canonicalized."""
    out = state_raw.copy()
    for country in country_series.unique():
        mask = country_series == country
        canon = _canonical_map_for(country)
        if native_map:
            canon.update({k: v for k, v in native_map.items() if k not in canon})
        mapped = state_raw[mask].map(canon)
        out[mask] = mapped.where(mapped.notna(), state_raw[mask])
    return out


def parse_addresses(df: pd.DataFrame, addr_col: str = "business_address",
                     country_col: str = "country", native_state_map: dict = None) -> pd.DataFrame:
    out = df.copy()
    native_state_map = native_state_map or {}

    out["address_is_ascii"] = tn.is_ascii_series(out[addr_col])
    folded = tn.fold(out[addr_col])
    folded = tn.clean_literal_null_whole(folded)

    def _row(addr, country):
        return _parse_row(addr, country, native_state_map)

    parsed = folded.combine(out[country_col], _row)
    parsed_df = pd.DataFrame(
        parsed.tolist(),
        index=out.index,
        columns=["address_clean", "house_number", "house_number_is_range",
                 "postal_code", "postal_code_type", "state_raw"],
    )
    out = pd.concat([out, parsed_df], axis=1)

    out["address_expanded"] = expand_street_abbreviations(out["address_clean"], out[country_col])
    # NOT chained: strip_accents (NFD + ASCII-encode) doesn't just fold accents,
    # it drops every character that fails the ASCII round-trip -- for a
    # Devanagari/Tamil/etc. address that means the WHOLE string, emptying the
    # token bag exactly for the non-ASCII-address minority (~18% of non-ASCII-
    # name records per EDA) that address-based blocking is supposed to rescue.
    # address_ascii stays independent, for accent-insensitive Latin-text
    # comparison only; tokens always come from the (unstripped) expanded form.
    out["address_ascii"] = tn.strip_accents(out["address_expanded"])
    out["address_tokens"] = tokenize_series(out["address_expanded"])
    out["state_norm"] = normalize_state_series(out["state_raw"], out[country_col], native_state_map)
    out["has_address"] = out["address_clean"].str.len() > 0
    return out
