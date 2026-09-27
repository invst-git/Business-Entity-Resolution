"""
Entry point. Run under `python -m cudf.pandas run_preprocessing.py ...` on a
qBraid GPU environment for acceleration, or plain `python run_preprocessing.py ...`
anywhere (CPU pandas) -- identical code path either way. See ../README.md.
"""
import argparse
import gc
import os
import time

import text_normalize as tn
from address_parser import normalize_state_series, parse_addresses, state_confident_hit_series
from io_utils import read_parquet, read_tsv, write_parquet
from name_parser import parse_names
import vocab_builder

SOURCE_SUFFIXES = ["source1", "source2", "source3"]
STATE_COLS = ["entity_id", "country", "state_raw"]


def process_file(path, native_map, sample=None):
    df = read_tsv(path, nrows=sample)
    df = parse_names(df)
    df = parse_addresses(df, native_state_map=native_map)
    return df


def qa_report(name, df, native_map):
    n = len(df)
    print(f"[{name}] rows={n:,}")
    print(
        f"  name: rename_marker={int(df['name_had_rename_marker'].sum()):,} "
        f"honorific={int(df['name_honorific_flag'].sum()):,} "
        f"non_ascii={int((~df['name_is_ascii']).sum()):,} "
        f"empty_core_name={int((df['core_name'].str.len() == 0).sum()):,}"
    )
    # state_norm is non-empty for nearly any row with an address (tier 4 always
    # falls back to the last raw component) -- that alone is not evidence the
    # value is *correct*. What matters for blocking confidence is how often we
    # resolved via a real tier-1/2 hit (known table or learned map) rather than
    # falling through to the tier-3/4 guesses.
    tier12_hit = state_confident_hit_series(df["state_raw"], df["country"], native_map)
    n_has_addr = int(df["has_address"].sum())
    print(
        f"  address: has_address={n_has_addr:,} "
        f"postal_found={int((df['postal_code'] != '').sum()):,} "
        f"state_tier1or2_hit={int(tier12_hit.sum()):,}/{n_has_addr:,} "
        f"non_ascii={int((~df['address_is_ascii']).sum()):,}"
    )


def build_france_locality_map(input_dir, sample=None):
    path = os.path.join(input_dir, "test", "test_source1.tsv")
    if not os.path.isfile(path):
        return {}
    df = read_tsv(path, columns=["business_address", "country"], nrows=sample)
    fr = df.loc[df["country"] == "France", "business_address"]
    if fr.empty:
        return {}
    folded = tn.clean_literal_null_whole(tn.fold(fr))
    return vocab_builder.build_locality_vocab(folded)


def run_train(args, native_map):
    t0 = time.time()
    slim = {}
    for suffix in SOURCE_SUFFIXES:
        path = os.path.join(args.input_dir, "train", f"train_{suffix}.tsv")
        df = process_file(path, native_map, sample=args.sample)
        qa_report(f"train_{suffix}", df, native_map)
        write_parquet(df, os.path.join(args.output_dir, f"train_{suffix}_clean.parquet"))
        slim[suffix] = df[STATE_COLS].copy()
        del df
        gc.collect()
    print(f"train name/address parsing done in {time.time() - t0:.1f}s")

    if native_map:
        return native_map

    gt_path = os.path.join(args.input_dir, "train", "train_ground_truth.tsv")
    gt = read_tsv(gt_path, nrows=args.sample)
    if args.sample:
        # Under --sample, source1/2/3 and ground_truth are each truncated to
        # their own first N rows independently, so overlap (and therefore the
        # derived vocab) will likely be sparse or empty -- expected for a fast
        # wiring smoke-test, not a substitute for a full run.
        ids = set(slim["source1"]["entity_id"])
        gt = gt[gt["source1_entity_id"].isin(ids)]

    native_map = vocab_builder.build_native_state_map(
        slim["source1"], slim["source2"], slim["source3"], gt,
    )
    vocab_path = os.path.join(args.artifacts_dir, "native_script_state_map.json")
    vocab_builder.save_map(native_map, vocab_path)
    print(f"derived native-script state map: {len(native_map)} entries -> {vocab_path}")
    del slim
    gc.collect()

    print("re-applying learned state map to already-written train parquet files...")
    for suffix in SOURCE_SUFFIXES:
        out_path = os.path.join(args.output_dir, f"train_{suffix}_clean.parquet")
        df = read_parquet(out_path)
        df["state_norm"] = normalize_state_series(df["state_raw"], df["country"], native_map)
        tier12_hit = state_confident_hit_series(df["state_raw"], df["country"], native_map)
        n_has_addr = int(df["has_address"].sum())
        print(f"  [train_{suffix}] after vocab: state_tier1or2_hit={int(tier12_hit.sum()):,}/{n_has_addr:,}")
        write_parquet(df, out_path)
        del df
        gc.collect()

    return native_map


def run_test(args, native_map):
    t0 = time.time()
    fr_map = build_france_locality_map(args.input_dir, sample=args.sample)
    if fr_map:
        print(f"derived France locality vocab: {len(fr_map)} entries (not persisted -- cheap to rebuild)")
        native_map = {**native_map, **fr_map}

    for suffix in SOURCE_SUFFIXES:
        path = os.path.join(args.input_dir, "test", f"test_{suffix}.tsv")
        df = process_file(path, native_map, sample=args.sample)
        qa_report(f"test_{suffix}", df, native_map)
        write_parquet(df, os.path.join(args.output_dir, f"test_{suffix}_clean.parquet"))
        del df
        gc.collect()
    print(f"test name/address parsing done in {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="dataset/ root (contains train/ and test/)")
    ap.add_argument("--output-dir", required=True, help="where *_clean.parquet files are written")
    ap.add_argument("--artifacts-dir", required=True, help="where the derived vocab JSON is written/read")
    ap.add_argument("--only", choices=["train", "test", "all"], default="all")
    ap.add_argument("--skip-vocab", action="store_true",
                     help="reuse artifacts/native_script_state_map.json instead of rebuilding it from train")
    ap.add_argument("--sample", type=int, default=None,
                     help="process only the first N rows per file (smoke-test before a full run)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)
    vocab_path = os.path.join(args.artifacts_dir, "native_script_state_map.json")

    native_map = {}
    if args.skip_vocab and os.path.exists(vocab_path):
        native_map = vocab_builder.load_map(vocab_path)
        print(f"loaded existing native-script state map ({len(native_map)} entries)")

    if args.only in ("train", "all"):
        native_map = run_train(args, native_map)

    if args.only in ("test", "all"):
        run_test(args, native_map)


if __name__ == "__main__":
    main()
