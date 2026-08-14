#!/usr/bin/env bash
# ASVspoof2019 LA is hosted on the University of Edinburgh DataShare and requires a
# click-through license acceptance in a browser — it cannot be scripted end-to-end.
#
# Manual steps:
#   1. Open https://datashare.ed.ac.uk/handle/10283/3336 in a browser.
#   2. Accept the Open Data Commons Attribution License and download
#      "LA.zip" (~7.1 GB, contains train/dev/eval flac audio + protocol files).
#   3. Move/extract it so the layout matches what manifests.py expects:
#        data/raw/asvspoof2019_la/
#          ASVspoof2019_LA_train/flac/*.flac
#          ASVspoof2019_LA_dev/flac/*.flac
#          ASVspoof2019_LA_eval/flac/*.flac
#          ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt
#          ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt
#          ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt
#
# This script just does the local extraction step once you've manually downloaded the
# zip into data/raw/ (checks the expected layout and unzips if needed).

set -euo pipefail

RAW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"
TARGET_DIR="$RAW_DIR/asvspoof2019_la"
ZIP_PATH="$RAW_DIR/LA.zip"

if [ -d "$TARGET_DIR/ASVspoof2019_LA_cm_protocols" ]; then
    echo "Already extracted at $TARGET_DIR"
    exit 0
fi

if [ ! -f "$ZIP_PATH" ]; then
    echo "Expected to find $ZIP_PATH"
    echo "Download LA.zip manually from https://datashare.ed.ac.uk/handle/10283/3336"
    echo "and place it at: $ZIP_PATH"
    exit 1
fi

mkdir -p "$TARGET_DIR"
echo "Extracting $ZIP_PATH -> $TARGET_DIR ..."
unzip -q "$ZIP_PATH" -d "$TARGET_DIR"
echo "Done. Verify layout matches the comment at the top of this script."
