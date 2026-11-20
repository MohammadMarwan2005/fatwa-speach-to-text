"""Download a Whisper model once (resumable) and verify it against its published SHA-256."""
import hashlib
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from . import config


def _ssl_context() -> ssl.SSLContext:
    """Trusted certificates for HTTPS. The system's list plus certifi's: the installed app's Python on
    macOS has no certificate list of its own, so without certifi every download fails verification."""
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except Exception:
        pass
    return ctx


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


def _is_certificate_error(e: Exception) -> bool:
    reason = getattr(e, "reason", e)
    return isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(e)


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
    ctx = _ssl_context()
    last_error = None
    for attempt in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        if have >= size:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-", "User-Agent": "fatwa-transcript"})
        try:
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp, \
                    open(part, "ab" if resp.status == 206 else "wb") as f:
                done = have if resp.status == 206 else 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done / size)
        except (OSError, ssl.SSLError) as e:
            last_error = e
            if _is_certificate_error(e):
                break  # retrying won't help; report it
            time.sleep(min(2 * (attempt + 1), 15))  # network hiccup: wait a little, then resume
    got = part.stat().st_size if part.exists() else 0
    if got != size:
        what = f"{info['file']}, {size / 1e6:.0f} MB"
        why = f" ({type(last_error).__name__}: {getattr(last_error, 'reason', last_error)})" if last_error else ""
        if last_error is not None and _is_certificate_error(last_error):
            raise RuntimeError(f"The Whisper model download ({what}) was blocked by a certificate check{why}. "
                               "If you are on a work or school network, it may inspect secure connections.")
        raise RuntimeError(f"The Whisper model download ({what}) stopped at {got / 1e6:.0f} MB{why}. "
                           f"Check the internet connection and try again; it continues where it stopped. "
                           f"Or download the file yourself into {path.parent}")
    if sha256(part) != info["sha256"]:
        part.unlink()
        raise RuntimeError("The downloaded Whisper model failed its checksum and was deleted. Try again.")
    part.replace(path)
