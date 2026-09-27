"""
Recovery entry point: builds and applies the native-script state vocab against
ALREADY-WRITTEN train_source{1,2,3}_clean.parquet files, without re-reading or
re-parsing the raw TSVs.

Use this when run_preprocessing.py finished writing the train parquet files
(you'll see three "writing parquet..." lines in its log) but was interrupted,
killed, or -- as happened before the vocab_builder.py performance fix -- hung
before reaching "re-applying learned state map...". Re-running
run_preprocessing.py --only all from scratch in that situation redoes the
train parsing for no reason; this picks up exactly where it left off instead.

After this finishes, run:
  python -m cudf.pandas run_preprocessing.py --only test --skip-vocab \
    --input-dir ... --output-dir ... --artifacts-dir ...
which loads the vocab this script just saved (via --skip-vocab) and processes
test using it, without re-touching train at all.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from address_parser import normalize_state_series, state_confident_hit_series
from io_utils import read_parquet, read_tsv, write_parquet
import vocab_builder

SOURCE_SUFFIXES = ["source1", "source2", "source3"]
STATE_COLS = ["entity_id", "country", "state_raw"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="dataset/ root (needs train/train_ground_truth.tsv)")
    ap.add_argument("--output-dir", required=True, help="where train_source*_clean.parquet already are")
    ap.add_argument("--artifacts-dir", required=True, help="where native_script_state_map.json gets written")
    args = ap.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)

    slim = {}
    for suffix in SOURCE_SUFFIXES:
        path = os.path.join(args.output_dir, f"train_{suffix}_clean.parquet")
        if not os.path.isfile(path):
            print(f"ERROR: {path} not found -- nothing to resume from.", file=sys.stderr)
            sys.exit(1)
        print(f"reading {path} (slim columns only)...")
        slim[suffix] = read_parquet(path, columns=STATE_COLS)

    print("reading train_ground_truth.tsv...")
    gt_path = os.path.join(args.input_dir, "train", "train_ground_truth.tsv")
    gt = read_tsv(gt_path)

    print("building native-script state map...")
    native_map = vocab_builder.build_native_state_map(
        slim["source1"], slim["source2"], slim["source3"], gt,
    )
    vocab_path = os.path.join(args.artifacts_dir, "native_script_state_map.json")
    vocab_builder.save_map(native_map, vocab_path)
    print(f"derived native-script state map: {len(native_map)} entries -> {vocab_path}")
    del slim

    print("re-applying learned state map to train parquet files...")
    for suffix in SOURCE_SUFFIXES:
        out_path = os.path.join(args.output_dir, f"train_{suffix}_clean.parquet")
        df = read_parquet(out_path)
        df["state_norm"] = normalize_state_series(df["state_raw"], df["country"], native_map)
        tier12_hit = state_confident_hit_series(df["state_raw"], df["country"], native_map)
        n_has_addr = int(df["has_address"].sum())
        print(f"  [train_{suffix}] after vocab: state_tier1or2_hit={int(tier12_hit.sum()):,}/{n_has_addr:,}")
        write_parquet(df, out_path)
        del df

    print("\nDone. Train parquet files are fully finished. Next, run test with --skip-vocab:")
    print(
        f"  python -m cudf.pandas run_preprocessing.py --only test --skip-vocab "
        f"--input-dir {args.input_dir} --output-dir {args.output_dir} --artifacts-dir {args.artifacts_dir}"
    )


if __name__ == "__main__":
    main()
