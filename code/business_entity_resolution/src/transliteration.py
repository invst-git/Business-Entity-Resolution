"""
Brahmic-script -> Latin romanization for India-country records, so the
character-n-gram and phonetic blocking channels have something to compare
against the English-spelled Source-1 anchors. Uses `indic_transliteration`
(MIT-licensed, PyPI: indic-transliteration) -- a mature, community-maintained
Sanscript transliteration engine -- rather than hand-rolling Unicode
code-point arithmetic: the underlying regularity (Brahmic scripts sharing
relative code-point offsets) is real, but a maintained library handles each
script's specific consonant-cluster/vowel-sign quirks correctly, which a
from-scratch implementation would not get right on the first pass.

Verified against real matched pairs from this project's training data before
being wired in (see conversation record): e.g. Devanagari
'ब्राइट फर्स्ट टेक प्रा. लि.' (a real S2 match for S1's 'Bright First Tech
Pvt Ltd') romanizes to 'brAiTa pharsTa Teka prA. li.' -- not a perfect
phonetic reproduction (schwa vowels survive, 'ph' represents an /f/ sound),
but close enough for character-n-gram and phonetic-code matching to find it.
Deliberately NOT post-processed with substitutions like ph->f: that
regularity is systematic enough that n-gram matching already tolerates it,
and a blanket substitution would corrupt the majority of India records that
are already Latin-script and pass through unchanged.
"""
import re
from collections import Counter

import pandas as pd
from indic_transliteration import sanscript

import text_normalize as tn

# Unicode block ranges for the Brahmic scripts observed in this project's
# training data (see EDA: native_script_state_map.json's derived tokens span
# exactly this set).
SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "gurmukhi": (0x0A00, 0x0A7F),
    "gujarati": (0x0A80, 0x0AFF),
    "oriya": (0x0B00, 0x0B7F),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
}

_SCHEME_ATTR = {name: name.upper() for name in SCRIPT_RANGES}
_WS_RE = re.compile(r"\s+")


def detect_dominant_script(text: str):
    """Returns the Brahmic script with the most characters in `text`, or None
    if the string is already Latin/ASCII/other (the common case -- ~80% of
    non-ASCII-name India records still have a fully-Latin address per EDA,
    and plenty of India names are already Latin-spelled too)."""
    if not text:
        return None
    counts = Counter()
    for ch in text:
        cp = ord(ch)
        for script, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[script] += 1
                break
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def romanize(text: str) -> str:
    """Best-effort romanization for blocking purposes, not a linguistically
    exact transliteration. Mixed-script strings (a native-script name next to
    an already-Latin word, the common case per EDA) are handled correctly:
    Sanscript's engine passes through characters that don't match the source
    script's patterns unchanged, so 'Dynamic કન્સલ્ટન્સી Private Limited'
    romanizes to 'Dynamic kansalTansI Private Limited', not just the Gujarati
    portion in isolation. A string mixing TWO different Brahmic scripts
    (rare) only gets the dominant one transliterated -- an accepted edge case
    given how uncommon this is in the data.
    """
    if not text:
        return text
    script = detect_dominant_script(text)
    if script is None:
        return text
    from_scheme = getattr(sanscript, _SCHEME_ATTR[script])
    try:
        out = sanscript.transliterate(text, from_scheme, sanscript.ITRANS)
    except Exception:
        return text
    return out


def romanized_series(core_name: pd.Series) -> pd.Series:
    """Per-row (romanize() is inherently row-wise -- variable script per
    string) romanization, then run through the SAME accent-folding used
    elsewhere (NFD decompose + ASCII round-trip) to fold ITRANS's diacritic
    marks (e.g. the grave accent in 'iனvèsDhmèNDhs') down to plain ASCII for
    the n-gram/phonetic channels. A handful of untransliterated characters
    from a rare mixed-script edge case get silently dropped by that same
    ASCII round-trip -- acceptable, matches the accent-folding behavior
    already used throughout this pipeline (see text_normalize.strip_accents).
    Writes a NEW representation, never overwrites `core_name`: exact-match
    and embedding channels need the original untouched.
    """
    romanized = core_name.fillna("").apply(romanize)
    return tn.strip_accents(romanized).str.lower()
