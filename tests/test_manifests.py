from muffle.data.manifests import build_deep_voice_manifest


def _touch_wavs(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"")
