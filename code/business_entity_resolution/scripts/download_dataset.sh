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

# Handle both possible zip layouts: contents zipped directly (train/, test/ at the
# top level) or the whole dataset/ folder zipped (one extra nesting level).
if [ -d "$DATASET_DIR/dataset/train" ] && [ ! -d "$DATASET_DIR/train" ]; then
    echo "Zip contained a nested dataset/ folder -- flattening..."
    mv "$DATASET_DIR/dataset"/* "$DATASET_DIR/"
    rmdir "$DATASET_DIR/dataset"
fi

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
