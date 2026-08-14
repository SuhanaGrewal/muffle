from muffle.data.manifests import build_deep_voice_manifest


def _touch_wavs(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"")


def test_build_deep_voice_manifest_labels_and_splits(tmp_path):
    _touch_wavs(tmp_path / "REAL", [f"real_{i}.wav" for i in range(4)])
    _touch_wavs(tmp_path / "FAKE", [f"fake_{i}.wav" for i in range(6)])

    manifest = build_deep_voice_manifest(tmp_path)

    assert len(manifest) == 10
    assert set(manifest["label"]) == {"bonafide", "spoof"}
    assert (manifest["label"] == "bonafide").sum() == 4
    assert (manifest["label"] == "spoof").sum() == 6

    bonafide_splits = manifest.loc[manifest["label"] == "bonafide", "split"].tolist()
    assert bonafide_splits.count("eval") == 1
    assert bonafide_splits.count("dev") == 1
    assert bonafide_splits.count("train") == 2
