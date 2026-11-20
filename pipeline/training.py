"""Training data from checked transcripts, for fine-tuning Whisper later.

A line counts as training data only when a person vouched for it: they corrected it by hand, or marked it
as checked (listened and found it right). Unchecked lines may still hold Whisper's own mistakes, and
training on those would teach Whisper its errors.

The export is a zip in the common "audiofolder" layout that Hugging Face datasets and most Whisper
fine-tuning scripts read directly:
    metadata.csv          file_name,transcription,whisper_text,source,job,line,start,end
    audio/<job>_<line>.wav  16 kHz mono clip of the line (with 0.2 s of padding)
    README.txt
"""
import csv
import io
import json
import time
import wave
import zipfile
from pathlib import Path

from . import config

PAD = 0.2  # seconds of audio kept before and after each line


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def items() -> list:
    """Every checked or hand-corrected line of every job: what goes into the training data."""
    out = []
    if not config.JOBS_DIR.exists():
        return out
    for d in sorted(config.JOBS_DIR.iterdir()):
        edits = _read(d / "edits.json", None) if d.is_dir() else None
        if not edits:
            continue
        payload = _read(d / "result.json", None)
        if not payload or not (d / "plain.wav").exists():
            continue
        edited = edits.get("lines", {})
        checked = {str(x) for x in edits.get("checked", [])} | set(edited)
        whisper = {}  # what Whisper's base run heard on each line, for comparison
        for ln in payload.get("compared", {}).get("lines", []):
            whisper[str(ln["id"])] = ln.get("base", "")
        for ln in payload["result"]["lines"]:
            lid = str(ln["id"])
            if lid not in checked:
                continue
            text = edited.get(lid, ln["text"]).strip()
            if text:
                out.append({"job": d.name, "line": lid, "start": ln["start"], "end": ln["end"], "text": text,
                            "whisper_text": whisper.get(lid, ""), "source": "corrected" if lid in edited else "checked",
                            "wav": d / "plain.wav"})
    return out


def stats() -> dict:
    its = items()
    return {"lines": len(its), "seconds": round(sum(max(0.0, i["end"] - i["start"]) for i in its), 1),
            "transcripts": len({i["job"] for i in its}), "corrected": sum(i["source"] == "corrected" for i in its)}


def _clip(wav: Path, start: float, end: float) -> bytes:
    with wave.open(str(wav), "rb") as w:
        rate, n = w.getframerate(), w.getnframes()
        a = max(0, int((start - PAD) * rate))
        b = min(n, int((end + PAD) * rate))
        w.setpos(a)
        frames = w.readframes(max(0, b - a))
        params = w.getparams()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setparams(params)
        out.writeframes(frames)
    return buf.getvalue()


README = """Fatwa Transcript training data
================================
Each row of metadata.csv is one line of a transcript that a person corrected or marked as checked,
with its audio clip in audio/. "transcription" is the person's text; "whisper_text" is what Whisper
heard, for comparison.

This is the "audiofolder" layout: datasets.load_dataset("audiofolder", data_dir="<this folder>") reads it.

Before training: keep some recordings aside as a test set (whole recordings, not random lines), measure
Whisper's error rate on it before and after, and only use recordings you have permission to use.
"""


def export_zip(dest: Path) -> dict:
    """Write the training data zip to dest. Returns stats about what was written."""
    its = items()
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = io.StringIO()
    writer = csv.writer(rows)
    writer.writerow(["file_name", "transcription", "whisper_text", "source", "job", "line", "start", "end"])
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for it in its:
            name = f"audio/{it['job']}_{int(it['line']):05d}.wav"
            z.writestr(zipfile.ZipInfo(name, time.localtime()[:6]), _clip(it["wav"], it["start"], it["end"]),
                       compress_type=zipfile.ZIP_STORED)
            writer.writerow([name, it["text"], it["whisper_text"], it["source"], it["job"], it["line"],
                             it["start"], it["end"]])
        z.writestr("metadata.csv", "﻿" + rows.getvalue())  # BOM so Excel shows Arabic correctly
        z.writestr("README.txt", README)
    return {"lines": len(its), "bytes": dest.stat().st_size}
