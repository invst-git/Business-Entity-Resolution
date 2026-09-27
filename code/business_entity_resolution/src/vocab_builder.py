"""
Derives the native-script (Devanagari/Bengali/Tamil/Gujarati/...) <-> Latin
state-name mapping from train_ground_truth.tsv matched pairs. This is NOT
hardcoded like the US/India state tables in config.py: unlike a state's English
name and postal abbreviation (public, standard reference data), a translation
table from script to script is not something to bake in from outside knowledge.
It is learned only from the provided training labels, run once on train, then
applied to both train and test.

Only needs three slim columns (entity_id, country, state_raw) per file, so this
step is cheap regardless of GPU/CPU and doesn't need the full parsed dataframe.
"""
import json
from collections import Counter, defaultdict

import pandas as pd

MIN_COUNT = 5
MIN_AGREEMENT = 0.9


def build_native_state_map(s1_state: pd.DataFrame, s2_state: pd.DataFrame,
                            s3_state: pd.DataFrame, ground_truth: pd.DataFrame) -> dict:
    votes = defaultdict(Counter)

    s1_map = s1_state.set_index("entity_id")
    s2_map = s2_state.set_index("entity_id")
    s3_map = s3_state.set_index("entity_id")

    for s1_id, matched in ground_truth.itertuples(index=False, name=None):
        if not matched or s1_id not in s1_map.index:
            continue
        s1_state_val = s1_map.at[s1_id, "state_raw"]
        if not s1_state_val or not s1_state_val.isascii():
            continue
        for mid in matched.split(","):
            src_map = s2_map if mid.startswith("S2-") else s3_map
            if mid not in src_map.index:
                continue
            other_state = src_map.at[mid, "state_raw"]
            if other_state and not other_state.isascii():
                votes[other_state][s1_state_val] += 1

    native_map = {}
    for native_token, counter in votes.items():
        total = sum(counter.values())
        best_state, best_count = counter.most_common(1)[0]
        if total >= MIN_COUNT and best_count / total >= MIN_AGREEMENT:
            native_map[native_token] = best_state
    return native_map


def build_locality_vocab(folded_addresses: pd.Series, min_count: int = 50) -> dict:
    """Frequency-based locality vocabulary for a country with no hardcoded state
    table and no training labels to learn a translation from -- built for France,
    which has neither (test-only, and Latin-script so there's no script to align
    anyway). Takes the last comma-component of each address as a candidate state/
    region/departement token and keeps whatever appears at least `min_count`
    times; EDA confirmed France's trailing slot mixes both regions
    ("Nouvelle-Aquitaine") and departements ("Loire-Atlantique", "Nord"), so this
    reads the vocabulary directly off the data's own usage rather than assuming
    one administrative level. Returns an identity mapping ({token: token}) so it
    plugs into the same native_map argument address_parser already threads
    through for the (translated) India case -- no translation needed here, just
    "is this a recognized locality" confidence.

    `folded_addresses` must already be NFKC+lowercased (text_normalize.fold), to
    match the case/form of state_raw at lookup time.
    """
    last = folded_addresses.str.rsplit(",", n=1).str[-1].str.strip()
    last = last[last != ""]
    counts = last.value_counts()
    vocab = counts[counts >= min_count].index.tolist()
    return {tok: tok for tok in vocab}


def save_map(native_map: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(native_map, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_map(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
