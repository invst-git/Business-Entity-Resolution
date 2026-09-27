"""
Static configuration for the preprocessing pipeline: vocabularies, regex patterns.

Design note on the "no external lookup" rule: the US state / Indian state / French
region tables below are hardcoded because they are closed, standard, non-business
reference facts (like knowing "Inc" abbreviates "Incorporated") -- not a business
identity lookup. The one thing that IS learned rather than hardcoded is the
native-script (Devanagari/Bengali/Tamil/Gujarati/...) <-> Latin state-name mapping,
because that is not public/obvious domain knowledge to bake in. It is derived from
train_ground_truth.tsv matched pairs by vocab_builder.py -- see that module.
"""
import re

LITERAL_NULL_RE = re.compile(r"(?i)\b(null|none|n/?a|nil)\b")

RENAME_MARKER_RE = re.compile(
    # Longer phrases before their shorter substrings (alternation is first-match,
    # not longest-match): "formerly known as" must be tried before "formerly" or
    # the trailing "known as" is left stuck onto the real name. Optional trailing
    # colon ("DBA: Real Name") is absorbed via lookahead, not a literal \b after
    # it -- colon-then-space are both non-word chars, so \b would never fire
    # there (same class of bug as the legal-suffix trailing-period fix).
    r"(?i)\b(?:fka|f/k/a|formerly known as|formerly|trading as|t/a|d/b/a|dba|aka|a/k/a):?(?=\W|$)"
)

HONORIFIC_RE = re.compile(
    r"(?i)^\s*(?:sri|shri|smt|dr|mr|mrs|ms|m/s)\.?\s+"
)

# Legal-suffix vocab: canonical_tag -> regex (case-insensitive, word-boundary).
# Acronym-style suffixes allow optional dots between letters (S.A.R.L. == SARL).
LEGAL_SUFFIX_PATTERNS = {
    "LLC": r"\bL\.?L\.?C\.?(?=\W|$)",
    "LLP": r"\bL\.?L\.?P\.?(?=\W|$)",
    "INC": r"\bINC(?:ORPORATED)?\.?(?=\W|$)",
    "CORP": r"\bCORP(?:ORATION)?\.?(?=\W|$)",
    "CO": r"\bCO(?:MPANY)?\.?(?=\W|$)",
    "LTD": r"\bLTD\.?(?=\W|$)|\bLIMITED\b",
    "PVT": r"\bPVT\.?(?=\W|$)|\bPRIVATE\b",
    "PLLC": r"\bP\.?L\.?L\.?C\.?(?=\W|$)",
    "PA": r"\bP\.?A\.?(?=\W|$)",
    "PC": r"\bP\.?C\.?(?=\W|$)",
    # France (test-only country, no training labels)
    "SARL": r"\bS\.?A\.?R\.?L\.?(?=\W|$)",
    "SASU": r"\bS\.?A\.?S\.?U\.?(?=\W|$)",
    "SAS": r"\bS\.?A\.?S\.?(?=\W|$)",
    "EURL": r"\bE\.?U\.?R\.?L\.?(?=\W|$)",
    "SCI": r"\bS\.?C\.?I\.?(?=\W|$)",
    "SNC": r"\bS\.?N\.?C\.?(?=\W|$)",
    "SA_FR": r"\bS\.?A\.?(?=\W|$)",
}
# Match order matters: try longer/more-specific acronyms before their substrings
# (SASU before SAS, PLLC before LLC, etc.) so "SASU" isn't first consumed as "SAS".
LEGAL_SUFFIX_ORDER = [
    "SASU", "SARL", "EURL", "SNC", "SCI", "PLLC", "LLC", "LLP",
    "INC", "CORP", "LTD", "PVT", "PA", "PC", "CO", "SAS", "SA_FR",
]
LEGAL_SUFFIX_COMPILED = [
    (tag, re.compile(f"(?i){LEGAL_SUFFIX_PATTERNS[tag]}")) for tag in LEGAL_SUFFIX_ORDER
]

US_ZIP_RE = re.compile(r"(?<!\d)\d{5}(-\d{4})?(?!\d)")
IN_PIN_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")
HOUSE_NUMBER_TOKEN_RE = re.compile(r"\b\d{1,6}[a-zA-Z]?(?:-\d{1,6})?\b")

# Split by locale: a bare "r" -> "rue" is only safe for French addresses (it would
# false-positive on stray single-letter tokens like "Apt R" elsewhere).
STREET_ABBREV_EXPANSION_EN = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane",
    "ct": "court", "hwy": "highway", "pkwy": "parkway", "pl": "place",
    "sq": "square", "ter": "terrace", "cir": "circle",
}
STREET_ABBREV_EXPANSION_FR = {
    "r": "rue", "av": "avenue", "bd": "boulevard", "pl": "place",
    "che": "chemin", "all": "allee", "imp": "impasse",
}
STREET_ABBREV_BY_COUNTRY = {
    "US": STREET_ABBREV_EXPANSION_EN,
    "India": STREET_ABBREV_EXPANSION_EN,
    "France": STREET_ABBREV_EXPANSION_FR,
}

# Every table below is oriented {ABBREVIATION: canonical lowercase full name} --
# consistently, on purpose. address_parser._canonical_map_for() assumes this one
# direction for every country; do not add a table in the opposite orientation
# (an earlier version had US and India in opposite orientations, which silently
# produced two different "canonical" strings -- e.g. "nc" staying "nc" while
# "north carolina" resolved to "NC" -- for the same real state. Caught only by
# testing US records specifically, since the India-oriented table happened to
# match the (buggy) assumed direction and looked correct in India-only testing.)
US_STATE_ABBREV = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas",
    "CA": "california", "CO": "colorado", "CT": "connecticut", "DE": "delaware",
    "FL": "florida", "GA": "georgia", "HI": "hawaii", "ID": "idaho",
    "IL": "illinois", "IN": "indiana", "IA": "iowa", "KS": "kansas",
    "KY": "kentucky", "LA": "louisiana", "ME": "maine", "MD": "maryland",
    "MA": "massachusetts", "MI": "michigan", "MN": "minnesota", "MS": "mississippi",
    "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada",
    "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico", "NY": "new york",
    "NC": "north carolina", "ND": "north dakota", "OH": "ohio", "OK": "oklahoma",
    "OR": "oregon", "PA": "pennsylvania", "RI": "rhode island",
    "SC": "south carolina", "SD": "south dakota", "TN": "tennessee", "TX": "texas",
    "UT": "utah", "VT": "vermont", "VA": "virginia", "WA": "washington",
    "WV": "west virginia", "WI": "wisconsin", "WY": "wyoming", "DC": "washington dc",
    # Not an abbreviation, but a full-form synonym that needs the same canonical
    # target: an address that spells DC out as two trailing components ("...,
    # Washington, District of Columbia") would otherwise have "washington" (the
    # state's own full name) match first during the reversed-component scan,
    # miscategorizing DC as Washington state.
    "DISTRICT OF COLUMBIA": "washington dc",
}

INDIAN_STATES = {
    "AP": "andhra pradesh", "AR": "arunachal pradesh", "AS": "assam", "BR": "bihar",
    "CG": "chhattisgarh", "GA": "goa", "GJ": "gujarat", "HR": "haryana",
    "HP": "himachal pradesh", "JH": "jharkhand", "KA": "karnataka", "KL": "kerala",
    "MP": "madhya pradesh", "MH": "maharashtra", "MN": "manipur", "ML": "meghalaya",
    "MZ": "mizoram", "NL": "nagaland", "OD": "odisha", "PB": "punjab",
    "RJ": "rajasthan", "SK": "sikkim", "TN": "tamil nadu", "TG": "telangana",
    "TR": "tripura", "UP": "uttar pradesh", "UK": "uttarakhand", "WB": "west bengal",
    "DL": "delhi", "JK": "jammu and kashmir", "LA": "ladakh", "PY": "puducherry",
    "CH": "chandigarh",
}

COUNTRIES_TRAIN = {"US", "India"}
COUNTRY_STATE_TABLES = {
    "US": US_STATE_ABBREV,
    "India": INDIAN_STATES,
    # France: deliberately absent. Test-only, no ground truth to learn a
    # native-script map from (moot anyway -- French is Latin-script), and its
    # trailing address slot mixes regions AND departements (confirmed in EDA:
    # "Nouvelle-Aquitaine" alongside "Loire-Atlantique", "Nord", "Gironde").
    # Hardcoding one or the other would be guessing at world knowledge rather
    # than reading it off the data. vocab_builder.build_locality_vocab derives
    # this country's table directly from the frequency of trailing components
    # in test_source1.tsv -- no labels needed, so it works despite France
    # having zero training rows.
}
