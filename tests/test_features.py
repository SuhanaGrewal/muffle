import torch

from muffle.features.lfcc import LFCCExtractor


def test_lfcc_output_shape():
    extractor = LFCCExtractor(sample_rate=16_000, n_lfcc=60)
    waveform = torch.randn(3, 16_000 * 4)  # batch of 3, 4s @ 16kHz

    feats = extractor(waveform)

    assert feats.shape[0] == 3
    assert feats.shape[1] == 60 * 3  # static + delta + delta-delta


def test_lfcc_accepts_unbatched_waveform():
    extractor = LFCCExtractor(sample_rate=16_000, n_lfcc=60)
    waveform = torch.randn(16_000 * 2)  # single 2s clip, no batch dim

    feats = extractor(waveform)

    assert feats.shape[0] == 1
    assert feats.shape[1] == 180


def test_lfcc_is_deterministic():
    extractor = LFCCExtractor(sample_rate=16_000)
    waveform = torch.randn(1, 16_000 * 2)

    feats_a = extractor(waveform)
    feats_b = extractor(waveform)

    assert torch.allclose(feats_a, feats_b)
