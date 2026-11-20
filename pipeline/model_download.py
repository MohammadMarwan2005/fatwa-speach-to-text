"""Download a Whisper model once (resumable) and verify it against its published SHA-256."""
import hashlib
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from . import config


def _known(path: Path):
    """The registry entry for this file name, if it is one of ours."""
    return next((m for m in config.WHISPER_MODELS.values() if m["file"] == path.name), None)


def is_ready(path: Path) -> bool:
    if not path.exists():
        return False
    info = _known(path)
    return info is None or path.stat().st_size == info["size"]  # a custom model file is trusted as-is


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(path: Path, progress: Optional[Callable[[float], None]] = None, attempts: int = 30) -> None:
    """Download to `path` if missing, resuming partial files; raise if the checksum is wrong."""
    if is_ready(path):
        return
    info = _known(path)
    if info is None:
        raise RuntimeError(f"The Whisper model file {path} doesn't exist.")
    size, url = info["size"], config.MODEL_URL.format(file=info["file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    for _ in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        if have >= size:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-", "User-Agent": "fatwa-transcript"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(part, "ab" if resp.status == 206 else "wb") as f:
                done = have if resp.status == 206 else 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done / size)
        except OSError:
            continue  # network hiccup: resume from what we have
    if not part.exists() or part.stat().st_size != size:
        raise RuntimeError("The Whisper model download did not finish. Check the internet connection and try again.")
    if sha256(part) != info["sha256"]:
        part.unlink()
        raise RuntimeError("The downloaded Whisper model failed its checksum and was deleted. Try again.")
    part.replace(path)
