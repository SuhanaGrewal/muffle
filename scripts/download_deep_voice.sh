#!/usr/bin/env bash
# DEEP-VOICE is hosted on Kaggle (birdy654/deep-voice-deepfake-voice-recognition):
# 8 American public figures, real audio vs. RVC voice-conversion fakes, ~62 minutes
# total, well under 1GB. Free to download, but Kaggle requires an account + API token
# (not a scriptable anonymous download).
#
# Manual steps:
#   1. Create a free Kaggle account if you don't have one, then generate an API token:
#      https://www.kaggle.com/settings -> "Create New Token" -> downloads kaggle.json.
#      Place it at ~/.kaggle/kaggle.json (chmod 600).
#   2. Either run this script (uses the kaggle CLI), or download the zip manually from
#      https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition
#      via the "Download" button and place it at data/raw/deep_voice.zip.
#
# Expected layout after extraction:
#   data/raw/deep_voice/REAL/*.wav
#   data/raw/deep_voice/FAKE/*.wav

set -euo pipefail

RAW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"
TARGET_DIR="$RAW_DIR/deep_voice"

if [ -d "$TARGET_DIR/REAL" ] && [ -d "$TARGET_DIR/FAKE" ]; then
    echo "Already extracted at $TARGET_DIR"
    exit 0
fi

mkdir -p "$TARGET_DIR"

if command -v kaggle >/dev/null 2>&1; then
    echo "Downloading via kaggle CLI..."
    kaggle datasets download -d birdy654/deep-voice-deepfake-voice-recognition -p "$RAW_DIR" --unzip -o
    # The kaggle CLI unzips directly into $RAW_DIR; move contents into $TARGET_DIR if needed.
    exit 0
fi

ZIP_PATH="$RAW_DIR/deep_voice.zip"
if [ ! -f "$ZIP_PATH" ]; then
    echo "kaggle CLI not found and $ZIP_PATH not present."
    echo "Either 'pip install kaggle' and configure ~/.kaggle/kaggle.json, or download"
    echo "the zip manually from the Kaggle page and place it at: $ZIP_PATH"
    exit 1
fi

unzip -q "$ZIP_PATH" -d "$TARGET_DIR"
echo "Done. Verify layout matches the comment at the top of this script."
