# muffle

AI-generated voice (deepfake audio) detection, built to serve as a fraud-prevention
signal in voice-based KYC (identity verification) flows. Given an audio clip, the
system classifies it as human speech or AI-generated/synthetic speech (TTS or voice
clone), trained and evaluated on public anti-spoofing datasets.

## Status

Research/prototype project. Not production-hardened. Local-only training on CPU/MPS
(Apple Silicon, no CUDA) — architecture and dataset choices are sized to fit that.

## Why this is hard

The core difficulty isn't classifying attacks you trained on — it's generalizing to
voice-generation systems you *didn't* train on. A detector trained only on one
TTS/vocoder family (e.g. ASVspoof's 2019 attacks) commonly fails against unrelated
generators or against a system released after training data was collected. This
project's evaluation protocol is built around measuring that gap explicitly
(cross-dataset generalization), not just reporting in-domain accuracy.

## Datasets

| Dataset | Role | License |
|---|---|---|
| DEEP-VOICE | Combined into training -- 8 American public figures, real audio vs. RVC voice-conversion fakes | Kaggle, freely downloadable |
| garystafford/deepfake-audio-detection | Combined into training -- 933 real / 933 commercial-TTS fakes, materialized locally from HF streaming (see `scripts/materialize_garystafford.py`) | CC-BY-4.0 |
| ASVspoof2019 LA | Combined into training -- adds volume and diversity (121k rows) but reintroduces British/VCTK-accented speakers into the training mix | Open Data Commons Attribution (free, no gate) |
| ASVspoof2021 DF | Cross-dataset generalization test (codec-compressed, closer to phone audio) | Zenodo, free registration |
| WaveFake | Cross-dataset generalization test (different vocoder family) | CC-BY-SA 4.0 |
| In-the-Wild | Held-out benchmark only, never trained on (real-world deepfakes) | Verify license before any commercial use |

**DEEP-VOICE is tiny on its own** (~62 minutes total: 8 real clips + 56 RVC-converted
fakes) -- not enough to train or evaluate anything trustworthy by itself (an eval split
with 1 example per class is a coin flip, not a measurement). Combined with garystafford
and ASVspoof2019 LA it contributes real diversity (a different attack type, RVC voice
conversion) to a much bigger combined pool. **LJSpeech + WaveFake's LJSpeech-conditioned
subsets** (also US-accented, ~29GB, one non-partitionable Zenodo zip) remains a documented
but not-yet-pursued option if more data is needed later.

**MLAAD is deliberately excluded** (CC-BY-NC 4.0, non-commercial) — not used here even
though this is a research project, to avoid any future re-licensing risk.

ASVspoof5 and FakeAVCeleb were considered but not included in this phase (gated/unclear
access terms) — worth revisiting if the project needs multilingual or audio-visual
coverage later.

Datasets are downloaded on demand (see `scripts/`) and are gitignored. All three
(DEEP-VOICE, garystafford, ASVspoof2019 LA) are combined for the current baseline
training run -- see the Usage section below.

## Architecture

- **Phase 1 baseline:** LFCC/CQT hand-crafted features + a small CNN. Cheap to train
  locally, validates the full pipeline end-to-end.
- **Phase 2 (implemented):** a frozen pretrained SSL model (`microsoft/wavlm-base-plus`
  by default) as a feature extractor, feeding a small trainable attentive-pooling + MLP
  head (`src/muffle/models/ssl_head.py`). The SSL backbone is *frozen*, not fine-tuned —
  full fine-tuning needs GPU budget this project doesn't assume. Frozen-SSL-features is
  the practical lever for cross-dataset generalization without that compute.
  `src/muffle/factory.py` dispatches on each config's `model_type` (`cnn_baseline` or
  `ssl_head`) so `train.py`/`evaluate.py` are architecture-agnostic.
- **Stretch goal, not built:** AASIST / RawNet2, the dedicated raw-waveform anti-
  spoofing architecture, as a from-repo reference implementation, if compute allows
  later (see `clovaai/aasist`).

## Project layout

```
src/muffle/
├── data/          # download, manifest-building, torch Dataset classes
├── features/      # LFCC/CQT and SSL (wav2vec2/WavLM) feature extraction
├── models/        # CNN baseline, SSL+head model
├── factory.py     # builds extractor+model from a config's model_type
├── train.py       # training entrypoint
├── evaluate.py    # EER / min t-DCF / cross-dataset eval reporting
└── metrics.py     # EER / min t-DCF, ported from the official ASVspoof eval kit
service/           # FastAPI inference API: app.py (POST /detect, GET /health),
                    # inference.py (model loading + preprocessing), schemas.py
configs/           # per-model training configs (yaml)
scripts/           # dataset download scripts, eval-runner scripts
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# 1. Download DEEP-VOICE (needs a free Kaggle account + API token)
bash scripts/download_deep_voice.sh
python -m muffle.data.manifests --dataset deep_voice

# 2. Materialize garystafford/deepfake-audio-detection from HF streaming (no bulk
#    download -- pulls samples one at a time and writes them locally)
python scripts/materialize_garystafford.py
python -m muffle.data.manifests --dataset garystafford

# 3. Download ASVspoof2019 LA (manual click-through -- see scripts/download_asvspoof2019.sh)
bash scripts/download_asvspoof2019.sh
python -m muffle.data.manifests --dataset asvspoof2019_la

# 4. Combine all three into one training manifest
python -m muffle.data.manifests --combine \
    data/processed/deep_voice_manifest.csv \
    data/processed/garystafford_manifest.csv \
    data/processed/asvspoof2019_la_manifest.csv \
    --out data/processed/combined_manifest.csv

# 5. Train the Phase 1 baseline
python -m muffle.train --config configs/baseline_lfcc_cnn.yaml

# 4. Evaluate (EER / min t-DCF)
python -m muffle.evaluate --config configs/baseline_lfcc_cnn.yaml --checkpoint checkpoints/baseline_lfcc_cnn/best.pt

# 5. Or train the Phase 2 SSL-head model instead
python -m muffle.train --config configs/ssl_wavlm_head.yaml

# 6. Run the inference API (env vars pick which trained checkpoint to serve)
MUFFLE_CONFIG=configs/baseline_lfcc_cnn.yaml \
MUFFLE_CHECKPOINT=checkpoints/baseline_lfcc_cnn/best.pt \
uvicorn service.app:app --reload
curl -F file=@sample.wav http://localhost:8000/detect
```

## Evaluation methodology

- **EER** (equal error rate): the threshold at which false-accept rate equals
  false-reject rate. Primary headline metric.
- **min t-DCF**: ASVspoof's application-weighted metric that accounts for a
  downstream speaker-verification system's errors alongside the countermeasure's own
  errors. Implemented in `src/muffle/metrics.py`, ported and unit-tested against the
  official reference implementation's known values — not reimplemented from scratch.
- **Cross-dataset generalization**: train on ASVspoof2019 LA, evaluate (no fine-tuning)
  on ASVspoof2021 DF, WaveFake, and In-the-Wild. This is the headline result — it
  indicates whether the detector generalizes or just memorized ASVspoof-specific
  artifacts.
- **Telephony realism**: re-run evaluation after simulating phone codecs (G.711/AMR,
  8kHz downsampling) on the audio, since training data is studio-quality but the real
  KYC use case is phone audio.

## Known limitations / future work

- **Streaming/real-time detection is not built.** This system analyzes a
  recorded/uploaded clip, not a live call. Moving to real-time would require:
  a fixed-size sliding-window chunking strategy (e.g. 1-2s windows with overlap)
  instead of whole-clip analysis, a sub-second per-chunk latency budget (which likely
  rules out the frozen-SSL model in favor of the lighter CNN or a distilled model),
  a websocket/gRPC streaming endpoint instead of file upload, and incremental
  score smoothing across chunks to avoid flip-flopping verdicts.
- **New TTS/voice-cloning systems keep emerging.** A static training snapshot degrades
  over time. Any real deployment should periodically re-evaluate against freshly
  generated audio from current voice-generation systems not present in training data,
  as an ongoing practice rather than a one-time task.
- **License re-audit needed before any commercial use.** Dataset licenses here were
  chosen to be safe for a research prototype; re-verify every dataset's actual terms
  (and any pretrained checkpoint's license) before using this in a commercial product.
