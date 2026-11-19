"""Download the Whisper model once (resumable) and verify it against its published SHA-256."""
import hashlib
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from . import config


def is_ready(path: Path) -> bool:
    return path.exists() and path.stat().st_size == config.WHISPER_MODEL_SIZE


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
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    for _ in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        if have >= config.WHISPER_MODEL_SIZE:
            break
        req = urllib.request.Request(config.WHISPER_MODEL_URL, headers={"Range": f"bytes={have}-"})
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
                        progress(done / config.WHISPER_MODEL_SIZE)
        except OSError:
            continue  # network hiccup: resume from what we have
    if not part.exists() or part.stat().st_size != config.WHISPER_MODEL_SIZE:
        raise RuntimeError("The Whisper model download did not finish. Check the internet connection and try again.")
    if sha256(part) != config.WHISPER_MODEL_SHA256:
        part.unlink()
        raise RuntimeError("The downloaded Whisper model failed its checksum and was deleted. Try again.")
    part.rename(path)
