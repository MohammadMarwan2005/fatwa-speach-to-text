"""Runs one transcription job and records every step in jobs/<id>/state.json.

The server starts it as:   python -m pipeline.job jobs/<id>
From a terminal:           python -m pipeline.job --file voice.m4a [--model claude-opus-5-5|none] [--runs 2]
A job whose state has "redo": true skips steps 1-3 and reuses its saved Whisper runs (runs/*.json).
"""
import argparse
import json
import os
import re
import secrets
import shutil
import sys
import time
import traceback
from pathlib import Path

from . import audio, compare, config, model_download, outputs
from .correct import CorrectionError, correct

STEPS = [
    ("upload", "Get the audio in"),
    ("convert", "Convert the audio"),
    ("whisper", "Speech to text"),
    ("compare", "Score and compare"),
    ("correct", "Correct and polish"),
    ("save", "Save the files"),
]
REDO_STEPS = ("compare", "correct", "save")


def _safe_name(name: str) -> str:
    name = Path(name or "audio").name
    return re.sub(r"[^\w.\- ()؀-ۿ]", "_", name)[:120] or "audio"


def _clean_options(options: dict) -> dict:
    model = options.get("model") or config.default_claude_model()
    if model not in config.CLAUDE_MODELS and model != config.FREE_MODE:
        model = config.default_claude_model()
    effort = options.get("effort") if options.get("effort") in config.EFFORTS else config.default_effort()
    try:
        runs = min(3, max(1, int(options.get("runs", 2))))
    except (TypeError, ValueError):
        runs = 2
    return {"model": model, "effort": effort, "runs": runs}


def create_job(filename: str, data: bytes = None, src: Path = None, options: dict = None) -> Path:
    """Make jobs/<id>/ with the input file and a queued state.json."""
    job_id = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
    d = config.JOBS_DIR / job_id
    d.mkdir(parents=True)
    filename = _safe_name(filename)
    ext = Path(filename).suffix.lower()[:8] or ".bin"
    target = d / f"input{ext}"
    if data is not None:
        target.write_bytes(data)
    else:
        shutil.copyfile(src, target)
    state = {
        "id": job_id, "filename": filename, "input": target.name, "size": target.stat().st_size,
        "created": time.time(), "status": "queued", "error": "", "warning": "", "duration": 0,
        "options": _clean_options(options or {}), "redo": False,
        "steps": [{"key": k, "title": t, "status": "pending", "detail": "", "fraction": 0,
                   "started": None, "ended": None} for k, t in STEPS],
    }
    (d / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


def saved_runs(d: Path) -> dict:
    """Whisper runs already saved in a job folder, in run order."""
    runs = {}
    for r in ("A", "B", "C"):
        p = d / "runs" / f"{r}.json"
        if p.exists():
            runs[r] = json.loads(p.read_text(encoding="utf-8"))
    return runs


def queue_redo(d: Path, model: str, effort: str) -> dict:
    """Mark a finished job to run steps 4-6 again with another model, reusing its Whisper runs."""
    st = json.loads((d / "state.json").read_text(encoding="utf-8"))
    opts = _clean_options({"model": model, "effort": effort, "runs": st["options"]["runs"]})
    st["options"].update(model=opts["model"], effort=opts["effort"])
    st.update(status="queued", redo=True, error="", warning="")
    for s in st["steps"]:
        if s["key"] in REDO_STEPS:
            s.update(status="pending", detail="", fraction=0, started=None, ended=None)
    (d / "state.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    return st


class Job:
    def __init__(self, d: Path):
        self.dir = d
        self.state = json.loads((d / "state.json").read_text(encoding="utf-8"))
        self.current = "upload"
        self._last_save = 0.0

    def save(self, force: bool = True) -> None:
        now = time.time()
        if not force and now - self._last_save < 0.5:
            return
        self._last_save = now
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.dir / "state.json")

    def step(self, key: str) -> dict:
        return next(s for s in self.state["steps"] if s["key"] == key)

    def start(self, key: str, detail: str = "") -> None:
        self.current = key
        self.step(key).update(status="running", started=time.time(), ended=None, detail=detail, fraction=0)
        self.save()

    def progress(self, key: str, fraction: float, detail: str = None) -> None:
        s = self.step(key)
        s["fraction"] = round(max(0.0, min(1.0, fraction)), 3)
        if detail is not None:
            s["detail"] = detail
        self.save(force=False)

    def finish(self, key: str, detail: str = "", status: str = "done") -> None:
        self.step(key).update(status=status, ended=time.time(), detail=detail, fraction=1)
        self.save()


def _mark_unsure(result: dict) -> dict:
    """Free mode text file: put (?) after each word Whisper was unsure about."""
    marked = json.loads(json.dumps(result))
    for ln in marked["lines"]:
        words = ln["text"].split(" ")
        for f in ln.get("flags", []):
            for i, w in enumerate(words):
                if w == f["w"]:
                    words[i] = w + "(?)"
                    break
        ln["text"] = " ".join(words)
    return marked


def _audio_and_whisper(job: Job) -> dict:
    """Steps 1-3. Returns the Whisper runs."""
    d, st = job.dir, job.state
    opts = st["options"]

    job.start("upload")
    src = d / st["input"]
    job.finish("upload", f"{st['filename']}, {st['size'] / 1e6:.1f} MB")

    job.start("convert", "ffmpeg → 16 kHz mono WAV")
    st["duration"] = audio.duration_seconds(src)
    audio.to_wav(src, d / "plain.wav")
    if opts["runs"] >= 3:
        audio.to_wav(src, d / "clean.wav", clean=True)
    if not st["duration"]:
        st["duration"] = (d / "plain.wav").stat().st_size / 32000
    job.finish("convert", f"16 kHz mono WAV, {outputs.clock(st['duration'])}")

    from . import whisper_runs  # imported here so a missing model doesn't break the other steps' imports
    job.start("whisper", "Checking the Whisper model")
    mpath = config.whisper_model_path()
    if not model_download.is_ready(mpath):
        model_download.ensure_model(mpath, lambda f: job.progress(
            "whisper", f * 0.05, f"Downloading the Whisper model, {f * 100:.0f}%"))
    chosen = whisper_runs.RUNS[:opts["runs"]]
    runs = {}
    (d / "runs").mkdir(exist_ok=True)
    for k, r in enumerate(chosen):
        label = f"Run {r['id']} of {len(chosen)}: {r['label'].lower()}"
        job.progress("whisper", 0.05 + 0.95 * k / len(chosen), label)
        t0 = time.time()
        # A fresh model per run: whisper.cpp otherwise feeds the previous run's text in as context,
        # and the runs would copy each other instead of disagreeing where Whisper is unsure.
        model = whisper_runs.load_model(mpath, config.whisper_threads())
        res = whisper_runs.transcribe(
            model, d / f"{r['audio']}.wav", r["prompt"], st["duration"],
            lambda f, k=k, label=label: job.progress(
                "whisper", 0.05 + 0.95 * (k + f) / len(chosen), f"{label}, {f * 100:.0f}%"))
        del model
        res.update(id=r["id"], label=r["label"], seconds=round(time.time() - t0, 1))
        (d / "runs" / f"{r['id']}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        runs[r["id"]] = res
    secs = sum(r["seconds"] for r in runs.values())
    job.finish("whisper", f"{len(runs)} run{'s' if len(runs) > 1 else ''} of Whisper large-v3, {outputs.clock(secs)}")
    return runs


def run(d: Path) -> int:
    job = Job(d)
    st = job.state
    opts = st["options"]
    st["status"] = "running"
    st["pid"] = os.getpid()
    job.save()
    try:
        if st.get("redo"):
            runs = saved_runs(d)
            if not runs:
                raise RuntimeError("This job has no saved Whisper runs to correct again.")
        else:
            runs = _audio_and_whisper(job)
        run_labels = {k: v["label"] for k, v in runs.items()}

        # 4. Compare
        job.start("compare")
        compared = compare.build_lines(runs)
        if not compared["lines"]:
            raise RuntimeError("Whisper found no speech in this file.")
        n_flags = sum(len(ln["low_conf"]) for ln in compared["lines"])
        n_diffs = sum(len(ln["diffs"]) for ln in compared["lines"])
        job.finish("compare", f"{len(compared['lines'])} lines, {n_flags} unsure words, {n_diffs} disagreements")

        # 5. Correct
        result = None
        job.current = "correct"
        if opts["model"] == config.FREE_MODE:
            job.finish("correct", "Skipped: free mode, check the flagged words yourself", status="skipped")
        elif not config.api_key():
            job.finish("correct", "Skipped: no API key in config.env", status="skipped")
            st["warning"] = "No API key in config.env, so this is Whisper's text with unsure words flagged."
        else:
            label = config.CLAUDE_MODELS[opts["model"]]["label"]
            job.start("correct", f"{label} is reading the runs")
            try:
                result = correct(compared, run_labels, opts["model"], opts["effort"],
                                 lambda f, part: job.progress("correct", f, f"{label} is writing{', ' + part if part else ''}"))
                u = result["usage"]
                served = ", ".join(u["served_by"])
                job.finish("correct", f"{label}: {len(result['corrections'])} fixes, ${u['cost_usd']:.4f} "
                                      f"({u['input_tokens']:,} in / {u['output_tokens']:,} out tokens"
                                      f"{', answered by ' + served if served != opts['model'] else ''})")
            except CorrectionError as e:
                job.finish("correct", str(e), status="error")
                st["warning"] = f"AI correction failed: {e} Meanwhile, here is Whisper's text with unsure words flagged."
        if result is None:
            result = compare.free_result(compared, list(runs))

        # 6. Save
        job.start("save")
        title = Path(st["filename"]).stem
        if result["mode"] == "corrected":
            how = (f"Corrected by {config.CLAUDE_MODELS[opts['model']]['label']} "
                   f"from {len(runs)} Whisper large-v3 run{'s' if len(runs) > 1 else ''}.")
            outputs.write_txt(result, title, st["duration"], how, d / "transcript.txt")
        else:
            how = "Whisper large-v3 only, not corrected. (?) marks words Whisper was unsure about."
            outputs.write_txt(_mark_unsure(result), title, st["duration"], how, d / "transcript.txt")
        outputs.write_srt(result, d / "transcript.srt")
        payload = {"result": result, "compared": compared,
                   "runs": {k: {"label": v["label"], "seconds": v["seconds"], "segments": v["segments"]}
                            for k, v in runs.items()}}
        (d / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        job.finish("save", "transcript.txt, transcript.srt, result.json")
        st["status"] = "done"
        st["redo"] = False
        job.save()
        return 0
    except Exception as e:  # report any failure on the page instead of dying silently
        traceback.print_exc()
        job.finish(job.current, str(e)[:300], status="error")
        st["status"] = "error"
        st["error"] = str(e)[:500]
        st["redo"] = False
        job.save()
        return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Transcribe an Arabic voice note.")
    ap.add_argument("job_dir", nargs="?", help="an existing jobs/<id> folder (used by the server)")
    ap.add_argument("--file", help="audio or video file to transcribe")
    ap.add_argument("--model", default=None, help="claude-opus-5-5, claude-sonnet-5-5, claude-haiku-5-5 or none")
    ap.add_argument("--runs", type=int, default=2, help="Whisper runs: 1, 2 or 3")
    ap.add_argument("--effort", default=None, help="low, medium or high")
    ap.add_argument("--redo", action="store_true", help="with a job folder: run the correction again on its saved runs")
    a = ap.parse_args(argv)
    if a.job_dir:
        d = Path(a.job_dir).resolve()
        if a.redo:
            queue_redo(d, a.model or config.default_claude_model(), a.effort or config.default_effort())
        return run(d)
    if not a.file:
        ap.error("give a job folder or --file")
    d = create_job(Path(a.file).name, src=Path(a.file), options={"model": a.model, "runs": a.runs, "effort": a.effort})
    code = run(d)
    st = json.loads((d / "state.json").read_text(encoding="utf-8"))
    for s in st["steps"]:
        print(f"{s['status']:>8}  {s['title']}: {s['detail']}")
    if st.get("warning"):
        print("Note:", st["warning"])
    print("Files:", d)
    return code


if __name__ == "__main__":
    sys.exit(main())
