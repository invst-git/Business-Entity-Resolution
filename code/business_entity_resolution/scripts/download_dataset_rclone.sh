#!/usr/bin/env bash
# Alternative to download_dataset.sh for a Drive file you don't want to make
# public. Uses rclone with a one-time OAuth login instead of a share link.
#
# One-time setup (run once per qBraid environment):
#   1. pip install --user rclone  # or: curl https://rclone.org/install.sh | sudo bash
#   2. rclone config
#        n) New remote
#        name> gdrive
#        Storage> drive (Google Drive)
#        client_id / client_secret> leave blank to use rclone's own
#        scope> 1 (full access) or 2 (read only) is enough
#        root_folder_id> leave blank
#        service_account_file> leave blank
#        Edit advanced config?> No
#        Use auto config?> If qBraid has no browser you can open, choose No and
#          follow the printed instructions to authorize from your local machine,
#          then paste the resulting token back into this prompt.
#   This writes a token to ~/.config/rclone/rclone.conf -- reusable across runs.
#
# Usage:
#   bash download_dataset_rclone.sh "<path/to/zip on your Drive>"
#   e.g. bash download_dataset_rclone.sh "student_resource_dataset.zip"
set -euo pipefail

DRIVE_PATH="${1:?Usage: bash download_dataset_rclone.sh <path-on-drive-to-zip>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASET_DIR="$PROJECT_ROOT/dataset"
ZIP_PATH="$PROJECT_ROOT/_dataset_download.zip"

command -v rclone >/dev/null 2>&1 || { echo "rclone not found -- see the setup comment at the top of this script." >&2; exit 1; }

echo "Copying $DRIVE_PATH from Drive remote 'gdrive' ..."
rclone copy "gdrive:$DRIVE_PATH" "$PROJECT_ROOT" --progress
mv "$PROJECT_ROOT/$(basename "$DRIVE_PATH")" "$ZIP_PATH"

mkdir -p "$DATASET_DIR"
unzip -oq "$ZIP_PATH" -d "$DATASET_DIR"
if [ -d "$DATASET_DIR/dataset/train" ] && [ ! -d "$DATASET_DIR/train" ]; then
    mv "$DATASET_DIR/dataset"/* "$DATASET_DIR/"
    rmdir "$DATASET_DIR/dataset"
fi
rm -f "$ZIP_PATH"

echo "Done. Files present:"
find "$DATASET_DIR" -name "*.tsv" | sort
