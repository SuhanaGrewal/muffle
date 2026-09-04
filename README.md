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
| In-the-Wild | Held-out benchmark only, never trained on (real-world deepfakes) | CC-BY-SA 4.0 -- commercial use OK with attribution + share-alike |

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

# 6. Evaluate (EER / min t-DCF)
python -m muffle.evaluate --config configs/baseline_lfcc_cnn.yaml --checkpoint checkpoints/baseline_lfcc_cnn/best.pt

# 7. Or train the Phase 2 SSL-head model instead
python -m muffle.train --config configs/ssl_wavlm_head.yaml

# 8. Run the inference API (env vars pick which trained checkpoint to serve)
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
- **Cross-dataset generalization**: originally planned as train-on-A/eval-on-B across
  separate datasets. **Currently not what's happening** -- DEEP-VOICE, garystafford, and
  ASVspoof2019 LA are all pooled into one combined train/dev/eval split (for training
  volume), so the current eval numbers are in-domain, not a generalization test. A
  held-out-dataset eval (train on two, test purely on the third, never seen during
  training) is still the real test of whether this generalizes and is not yet done.
- **Telephony realism**: re-run evaluation after simulating phone codecs (G.711/AMR,
  8kHz downsampling) on the audio, since training data is studio-quality but the real
  KYC use case is phone audio.

## Results

EER on the combined dataset's eval split (in-domain, not cross-dataset -- see
limitations above):

| Model | Training data | Dev EER | Eval EER |
|---|---|---|---|
| CNN baseline (LFCC) | full combined manifest (123k rows) | 0.94% | 16.75% |
| WavLM + head | shrunk subsample (4.8k rows, 18x less data) | -- | 12.88% |

WavLM already generalizes better than the CNN despite far less training data, and the
two models fail on different attack types (WavLM notably stronger on A12/A13, CNN
stronger on A14/A17/A19) -- evidence of complementary rather than correlated errors.

**Ensemble** (`scripts/ensemble_eval.py`): scoring both models' checkpoints against the
same held-out 14,950-row eval subset (`data/processed/eval_subset_for_comparison.csv`,
built with `subsample_manifest`) and averaging z-score-normalized scores:

| Model | EER (shared subset) |
|---|---|
| CNN alone | 16.58% |
| WavLM alone | 13.16% |
| **Ensemble (normalized score average)** | **6.68%** |

Averaging cuts EER by ~60% relative to the CNN alone and ~50% relative to WavLM alone --
consistent with the two models' error patterns being complementary rather than
redundant. Scores must come from `scripts/score_for_ensemble.py` run against the exact
same manifest for both models (row order must match) before combining.

**Fusion experiment (negative result, kept for honesty):** `scripts/fit_fusion.py` fits a
logistic regression on dev-set CNN+WavLM scores instead of assuming a 50/50 average, then
applies it to the eval subset. This scored *worse* -- 8.62% EER, vs. 6.68% for the plain
average. Likely cause: dev shares attack types with train (A01-A06), while eval introduces
unseen attack types (A07-A19) -- the fitted weights captured a CNN/WavLM relationship
specific to dev's attack distribution that didn't transfer to eval's unseen attacks. Plain
averaging turned out more robust to that shift than a model fit to seen-attack data. Kept
as the reported result: simple averaging, not learned fusion.

Next step: retrain WavLM on the full combined manifest (not the 4.8k-row subsample) for
a fair standalone comparison and likely a stronger ensemble -- not yet done (needs 8-16
hours of uninterrupted local compute; deferred for now given machine time constraints).

### Cross-dataset generalization (the real test)

Everything above is in-domain -- DEEP-VOICE, garystafford, and ASVspoof2019 LA are all
pooled into one train/dev/eval split, so those eval numbers measure how well each model
fits that shared distribution, not whether it generalizes. The actual test: score both
already-trained checkpoints, cold, against **In-the-Wild** (Muller et al.) -- 31,779
real-world deepfake clips of public figures, never seen during training, downloaded and
scored via `scripts/score_for_ensemble.py` against `data/processed/in_the_wild_manifest.csv`.

| Model | Cross-dataset EER (In-the-Wild) | In-domain eval EER |
|---|---|---|
| CNN (LFCC) | **55.28%** -- chance level | 16.75% |
| WavLM + head | **14.85%** | 12.88% |
| Ensemble (CNN+WavLM average) | 32.64% -- worse than WavLM alone | 6.68% |

This is the project's central finding. The CNN's strong in-domain number was misleading
-- it memorized artifacts specific to its training distribution (studio-quality audio,
a narrow set of TTS/vocoder families) and is statistically indistinguishable from random
guessing on real-world audio it wasn't trained on. WavLM's frozen self-supervised
features hold up close to their in-domain number even cross-dataset -- the generalization
bet behind using a frozen SSL backbone (see Architecture) actually paid off.

**The ensemble that helped in-domain actively hurts cross-dataset.** Averaging a
chance-level CNN score into WavLM's good score drags the combined result down --
ensembling isn't free, and blending in a component that isn't generalizing makes things
worse, not better. **`service/app.py` now defaults to serving WavLM alone**, not the
ensemble, based on this result.

### Data-diversity experiment (negative result, kept for honesty)

WavLM's original training subsample (`combined_manifest_small.csv`) turned out to be
86% ASVspoof2019 LA by row count -- `subsample_manifest` pooled all datasets' rows before
capping, so ASVspoof2019 LA's much larger pool crowded out garystafford (6 modern TTS
platforms, including 173 ElevenLabs clips) down to ~23 rows and DEEP-VOICE down to ~5,
almost by chance. Two fixes: `scripts/materialize_garystafford.py` now preserves original
filenames (previously renamed to sequential numbers, discarding which TTS platform made
each clip), and `subsample_manifest` now stratifies by dataset too, so every source
contributes everything it has up to the cap instead of being pooled against ASVspoof's
volume. This gave WavLM v2 a ~2x larger, much more diverse training set (all 745
garystafford train rows across 6 platforms, ~137 ElevenLabs specifically, vs. ~23 of any
platform before).

| Model | Cross-dataset EER (In-the-Wild) | In-domain eval EER |
|---|---|---|
| WavLM v1 (ASVspoof-dominated, ~23 garystafford rows) | **14.85%** | 12.75% |
| WavLM v2 (dataset-stratified, ~745 garystafford rows) | 19.25% | 12.58% |

**v2 is worse on the number that matters**, despite better in-domain and dev EER (8.83%).
The generalization gap (in-domain -> cross-dataset) widened from +2.1 points (v1) to
+6.67 points (v2) -- the hallmark of overfitting, on a model that v1's numbers suggested
wasn't prone to it. Best working explanation: giving the small trainable head more spoof
examples, even diverse ones, from a still-closed set of 7 known generators let it fit
more tightly to the union of "known fake" patterns, at some cost to the more abstract
signal that happened to transfer well when there was less to specifically fit to. More
training-data diversity did not automatically buy better generalization here.

**`service/app.py` stays on WavLM v1** (`checkpoints/ssl_wavlm_head/`), not v2. The v2
config/checkpoint (`configs/ssl_wavlm_head_v2.yaml`, `checkpoints/ssl_wavlm_head_v2/`)
is kept for reference, not deployed.

## Known limitations / future work

- **`num_workers > 0` stalls badly on macOS with the MPS device** -- DataLoader worker
  processes spun up but did almost no work over ~20 minutes real time. Root cause not
  investigated further; `configs/*.yaml` set `num_workers: 0`, which is fast enough at
  this model/data scale. Worth revisiting only if training throughput actually matters.
- **Long-running local training needs `caffeinate` (or equivalent).** A training run
  left running while the Mac was asleep accumulated 9.5 hours of wall-clock time for
  ~23 minutes of actual CPU work -- background processes get paused, not killed, on
  sleep. Wrap long training/download commands in `caffeinate -i` to prevent this.
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
