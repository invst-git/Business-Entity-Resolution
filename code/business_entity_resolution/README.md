# Business Entity Resolution -- Preprocessing (Phase A)

Phase A only: text/address normalization and feature-ready cleanup that the
blocking and matching phases (next) will consume. No blocking, no candidate
generation, no model training happens here.

## Directory layout (on qBraid, project root)

```
student_resource/                      <- clone/copy this whole repo to qBraid
├── dataset/                           <- populated by scripts/download_dataset.sh
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── config.py            # vocabularies: legal suffixes, honorifics, state tables
│       │   ├── io_utils.py          # TSV/parquet read-write
│       │   ├── text_normalize.py    # NFKC, case-fold, literal-"null" cleanup
│       │   ├── name_parser.py       # rename-marker split, honorific strip, suffix extraction
│       │   ├── address_parser.py    # component parsing, house#/postal/state extraction
│       │   ├── vocab_builder.py     # learns native-script<->state map (train) + France locality vocab (test)
│       │   ├── run_preprocessing.py # CLI entry point
│       │   └── resume_vocab_from_parquet.py  # recovery: see "If a run gets interrupted" below
│       ├── scripts/
│       │   ├── download_dataset.sh          # Drive (public link) -> dataset/, via gdown
│       │   ├── download_dataset_rclone.sh   # Drive (private) -> dataset/, via rclone
│       │   └── setup_env.sh                 # installs RAPIDS cudf.pandas + requirements.txt
│       ├── requirements.txt
│       └── README.md                # this file
├── artifacts/                         <- vocab_builder output (derived from TRAIN only)
│   └── native_script_state_map.json
├── processed/                         <- run_preprocessing.py output, ready for blocking
│   ├── train_source1_clean.parquet
│   ├── train_source2_clean.parquet
│   ├── train_source3_clean.parquet
│   ├── test_source1_clean.parquet
│   ├── test_source2_clean.parquet
│   └── test_source3_clean.parquet
├── output/                            <- final submission outputs (later phase, not here)
├── README.md                          # original challenge README
├── Documentation_template.md
└── utils/validate_submission.py
```

## 1. Get the dataset onto qBraid

The zip lives in Google Drive. Pick one:

- **Public link is fine:** share the zip as "Anyone with the link", grab its file
  ID from the share URL, then from `code/business_entity_resolution/scripts/`:
  ```bash
  bash download_dataset.sh <GOOGLE_DRIVE_FILE_ID>
  ```
- **Keep it private:** one-time `rclone config` (instructions in the script's
  header), then:
  ```bash
  bash download_dataset_rclone.sh "path/on/drive/to/the.zip"
  ```

Either way you end up with `dataset/train/*.tsv` and `dataset/test/*.tsv` at the
project root. The script detects and flattens a nested `dataset/` folder inside
the zip automatically, and warns if any expected file is missing.

## 2. Set up the GPU environment

```bash
cd code/business_entity_resolution/scripts
bash setup_env.sh
```

This detects your CUDA version from `nvidia-smi` and installs RAPIDS
`cudf.pandas` (the pip wheel matching CUDA 11 or 12), then the rest of
`requirements.txt`. If CUDA auto-detection fails, it prints a link to RAPIDS'
own install-command selector (https://docs.rapids.ai/install/) -- use that
instead of a hardcoded version, since pinning one here would go stale.

**Why `cudf.pandas` and not hand-written cuDF code:** every module in `src/` is
written against the plain `pandas` API. Run it as-is on CPU, or launch it with
`python -m cudf.pandas run_preprocessing.py ...` on a GPU box and the exact same
code gets GPU acceleration for the vectorized `.str` operations (NFKC
normalization, case folding, regex suffix/honorific extraction, street-abbreviation
expansion) -- which is the bulk of the runtime at 5M+ rows/file. A handful of
operations are inherently row-wise (address component splitting -- comma count
and field order both vary per row, confirmed in EDA) and fall back to CPU
automatically under `cudf.pandas`; that's expected, not a bug.

## 3. Run preprocessing

Smoke-test first (finishes in seconds, catches path/environment problems early):

```bash
cd ../src
python -m cudf.pandas run_preprocessing.py \
  --input-dir ../../../dataset \
  --output-dir ../../../processed \
  --artifacts-dir ../../../artifacts \
  --sample 20000
```

Then the full run:

```bash
python -m cudf.pandas run_preprocessing.py \
  --input-dir ../../../dataset \
  --output-dir ../../../processed \
  --artifacts-dir ../../../artifacts
```

This processes train first (building `artifacts/native_script_state_map.json`
from train_ground_truth.tsv along the way -- see "What gets learned vs.
hardcoded" below), then test using that map. If you need to iterate on just one
side, `--only train` / `--only test`, and `--skip-vocab` reuses an
already-built `native_script_state_map.json` instead of rebuilding it.

Train specifically runs in two passes to keep peak memory bounded: parse and
write each of the 3 train files to parquet immediately (keeping only the slim
`entity_id`/`country`/`state_raw` columns resident afterward, not the full
~20-column, up-to-5.3M-row dataframe), derive the vocab from those slim frames,
then re-open each parquet just to recompute `state_norm` with the now-final map
and overwrite it in place -- you'll see a "re-applying learned state map..."
line for this second pass. Only one full dataframe is ever resident at a time.

**Performance expectations.** Measured on an ordinary (non-GPU) dev machine at
30K rows/file: ~3,600 rows/sec combined across the 3 train files, ~2,750/sec
across the 3 test files -- extrapolating, roughly **1-1.5 hours for the full
~24M rows** (train+test) on CPU alone. The row-wise address-component parsing
(comma splitting, state-tier matching -- inherently per-row since comma count
and field order both vary) is the bulk of that cost and does **not** speed up
under `cudf.pandas`, since it isn't vectorizable; it's expected to fall back to
CPU there too. The name-parsing side (NFKC/case-fold/regex suffix and honorific
extraction, the other big cost) *is* fully vectorized and should see a real
GPU speedup under `cudf.pandas`, so your actual qBraid wall-clock should beat
this CPU estimate, but don't expect GPU to erase the address-parsing floor.
Plan to run the full job as a detached/background process (`nohup ... &`, or a
long-running notebook cell) rather than watching a terminal.

Each file prints a QA summary as it finishes: rename-marker hits, honorific
hits, non-ASCII rate, postal-code extraction rate, and **`state_tier1or2_hit`**
-- the fraction of addressed rows whose state resolved via the hardcoded table
or a learned map (tier 1/2), as opposed to falling through to the foreign-script
or positional-last guesses (tier 3/4). Track this one specifically: `state_norm`
itself is non-empty for almost every row with any address (tier 4 always
returns something), so a non-empty `state_norm` is not evidence it's *correct*
-- the tier1or2 hit rate is. Sanity-check these against the EDA report before
moving on; a sudden drop where you expect ~90%+ usually means a path or
encoding problem, not real data.

## If a run gets interrupted after train finishes parsing

`run_train()` writes each `train_source*_clean.parquet` as soon as that file's
name/address parsing completes, *before* building the native-script vocab.
If the vocab-building step (or anything after it) gets killed, hangs, or
errors out -- train parsing already succeeded and doesn't need to be redone.
Resume from there instead of re-running the whole thing from scratch:

```bash
python resume_vocab_from_parquet.py \
  --input-dir ../../../dataset \
  --output-dir ../../../processed \
  --artifacts-dir ../../../artifacts
```

This reads the three already-written train parquet files (slim columns only),
builds and saves the vocab, and re-applies it to those same files in place --
exactly the second half of `run_train()`, just skipped straight to. It prints
the exact follow-up command for test when it finishes:

```bash
python -m cudf.pandas run_preprocessing.py --only test --skip-vocab \
  --input-dir ../../../dataset --output-dir ../../../processed --artifacts-dir ../../../artifacts
```

No need for `cudf.pandas` on the resume step itself -- it's a handful of
seconds of plain Python/pandas work at this scale, not worth any GPU-proxy
overhead.

## What gets learned vs. hardcoded (fair-play note)

`config.py` hardcodes US state abbreviations and Indian state names -- static,
public reference facts (like knowing "Inc" abbreviates "Incorporated"), not a
business-identity lookup, so they're fine to bake in directly per the
challenge's external-lookup rule. Both tables are kept in a single, consistent
`{ABBREVIATION: canonical lowercase full name}` orientation on purpose -- an
earlier draft had them in opposite orientations, which silently produced two
different "canonical" strings for the same real state (e.g. "nc" staying "nc"
while "north carolina" resolved to "NC"); only caught by testing US records
specifically; India's table happened to be written in the direction the bug
assumed, so India-only testing looked fine. If you extend either table, keep
the orientation.

Two things are **learned, not hardcoded**, both by `vocab_builder.py`:

- **Native-script -> Latin state mapping** (India). Derived purely from
  `train_ground_truth.tsv` matched pairs -- for each true match where the
  Source-1 side names a recognized English state, whatever non-ASCII token sits
  in the matched record's state slot gets a vote for that state name; tokens
  with at least 5 consistent votes (>=90% agreement) make it into
  `artifacts/native_script_state_map.json`. Verified on real training data this
  correctly recovers all major Indian scripts, e.g. `'महाराष्ट्र' ->
  'maharashtra'`, `'தமிழ்நாடு' -> 'tamil nadu'`, `'ಕರ್ನಾಟಕ' -> 'karnataka'`,
  `'ਪੰਜਾਬ' -> 'punjab'` -- with zero external translation resource involved.
- **Locality vocabulary** (France). France has no hardcoded state table at
  all: it's test-only (no ground truth to learn a translation from), and
  doesn't need one anyway since French is Latin-script, not a different script
  to align. `vocab_builder.build_locality_vocab` instead reads the vocabulary
  directly off `test_source1.tsv`'s own address field -- whatever trailing
  comma-component appears at least 50 times becomes a recognized locality
  token, mapped to itself (no translation needed, just "is this a place name I
  should trust"). EDA showed France's trailing slot mixes regions
  ("Nouvelle-Aquitaine") and departements ("Loire-Atlantique", "Nord",
  "Gironde") inconsistently; reading the frequency table directly sidesteps
  having to hardcode (i.e., assume) either administrative level. This is
  recomputed fresh each run (not persisted) since it's cheap and doesn't depend
  on train.

## Known limitations (by design, not oversights)

- **Legal-suffix regexes are Latin-script only.** A suffix written in a native
  script (e.g. Tamil "எல்எல்பி" for "LLP") is not stripped from `core_name`.
  Addressed downstream by leaning on address-based blocking for such records
  rather than name-suffix matching.
- **House-number extraction takes the first digit run in the first two address
  components.** When a record has an injected numbering prefix ("Door No 162
  1401, ...", a documented noise pattern), this can grab the prefix instead of
  the "true" number. No reliable way to disambiguate without ground truth: left
  as a directional signal for the matching-stage features, not an identity key.
- **State extraction falls through to a foreign-script or positional-last
  guess** when no component matches the hardcoded table or a learned map (tiers
  3/4 in `address_parser._extract_state`) -- this is exactly what
  `state_tier1or2_hit` in the QA output measures the rate of. Good enough as a
  blocking key when it does; don't treat `state_norm` as ground truth. One
  specific known miss: a rare/low-frequency French departement below the
  France locality vocab's 50-occurrence threshold falls through to tier 4.
- Free-text noise beyond scope: phone numbers embedded in name fields, stray
  decorative characters (`>>`, `[...]`), and similar one-off artifacts are left
  in `core_name` untouched. `core_name_ascii` (name_parser) and `address_ascii`
  (address_parser) add accent-folded companion columns for the "Ínstitute" /
  "pártners" / "énterprises" class of accent-corruption noise, but do not
  address the ones above.

## Next phase (not in this script)

`processed/*_clean.parquet` is the input to blocking/candidate generation
(Phase B), which is a separate script to be added next -- it consumes
`core_name` (or `core_name_ascii` for accent-insensitive matching),
`address_tokens`, `state_norm`, `postal_code`, and `country` to build the
candidate pairs, budgeted to roughly 20-30 candidates per Source-1 entity per
the EDA's observed match-count distribution (mean 3.46, max 11).
