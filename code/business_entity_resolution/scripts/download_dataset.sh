#!/usr/bin/env bash
# Downloads the challenge dataset zip from Google Drive and unpacks it to
# <project_root>/dataset, the layout run_preprocessing.py expects:
#   dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
#   dataset/test/{test_source1,test_source2,test_source3}.tsv
#
# Requires the zip to be reachable via a Drive share link (gdown needs at least
# "Anyone with the link -> Viewer"). If your Drive stays private, use
# download_dataset_rclone.sh instead.
#
# Usage:
#   1. In Drive: right-click the zip -> Share -> General access -> "Anyone with
#      the link" (Viewer). You can revert this after the download finishes.
#   2. Copy the file ID from the share link:
#        https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
#   3. Run:  bash download_dataset.sh <FILE_ID>
set -euo pipefail

FILE_ID="${1:?Usage: bash download_dataset.sh <GOOGLE_DRIVE_FILE_ID>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASET_DIR="$PROJECT_ROOT/dataset"
ZIP_PATH="$PROJECT_ROOT/_dataset_download.zip"

pip show gdown >/dev/null 2>&1 || pip install -q gdown

echo "Downloading dataset zip (file id: $FILE_ID) ..."
gdown "$FILE_ID" -O "$ZIP_PATH"

echo "Unzipping into $DATASET_DIR ..."
mkdir -p "$DATASET_DIR"
unzip -oq "$ZIP_PATH" -d "$DATASET_DIR"

# The zip's internal layout depends entirely on how it was created locally --
# contents zipped directly, the dataset/ folder zipped (one extra level), or a
# whole project folder zipped (dataset/student_resource/dataset/..., two extra
# levels). Rather than assume a specific nesting depth, find the actual train/
# and test/ directories by name, wherever they landed, and move them to the top
# of $DATASET_DIR.
FOUND_TRAIN=$(find "$DATASET_DIR" -type d -name train | head -1)
FOUND_TEST=$(find "$DATASET_DIR" -type d -name test | head -1)

if [ -z "$FOUND_TRAIN" ] || [ -z "$FOUND_TEST" ]; then
    echo "ERROR: could not find both a 'train' and 'test' folder anywhere inside the extracted zip." >&2
    echo "Extracted contents:" >&2
    find "$DATASET_DIR" -maxdepth 4 >&2
    exit 1
fi

if [ "$FOUND_TRAIN" != "$DATASET_DIR/train" ]; then
    echo "Found train/ nested at ${FOUND_TRAIN#"$DATASET_DIR"/} -- moving to dataset/train"
    rm -rf "$DATASET_DIR/train"
    mv "$FOUND_TRAIN" "$DATASET_DIR/train"
fi
if [ "$FOUND_TEST" != "$DATASET_DIR/test" ]; then
    echo "Found test/ nested at ${FOUND_TEST#"$DATASET_DIR"/} -- moving to dataset/test"
    rm -rf "$DATASET_DIR/test"
    mv "$FOUND_TEST" "$DATASET_DIR/test"
fi

# Clean up whatever empty wrapper directories the move left behind.
find "$DATASET_DIR" -mindepth 1 -maxdepth 4 -type d -empty -delete

rm -f "$ZIP_PATH"

echo
echo "Done. Files present:"
find "$DATASET_DIR" -name "*.tsv" | sort

for f in train/train_source1.tsv train/train_source2.tsv train/train_source3.tsv \
         train/train_ground_truth.tsv test/test_source1.tsv test/test_source2.tsv \
         test/test_source3.tsv; do
    if [ ! -f "$DATASET_DIR/$f" ]; then
        echo "WARNING: expected $DATASET_DIR/$f but it is missing." >&2
    fi
done
