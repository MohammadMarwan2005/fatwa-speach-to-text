"""Step 2: convert any audio/video file to the 16 kHz mono WAV that Whisper reads."""
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
import imageio_ffmpeg

# Light cleanup, used only by the optional third run. On clean recordings it can make Whisper worse.
CLEAN_FILTER = "highpass=f=70,afftdn=nr=10:nf=-40,loudnorm=I=-18:TP=-2:LRA=11"


def ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def duration_seconds(src: Path) -> float:
    """Read the duration ffmpeg prints for the input (0.0 if it can't tell)."""
    out = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(src)], capture_output=True, text=True).stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if not m:
        return 0.0
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def to_wav(src: Path, dst: Path, clean: bool = False) -> None:
    cmd = [ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000"]
    if clean:
        cmd += ["-af", CLEAN_FILTER]
    cmd += ["-c:a", "pcm_s16le", str(dst)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg could not read this file: {res.stderr.strip()[:300] or 'unknown error'}")


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1:
            raise ValueError("expected 16 kHz mono WAV")
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
