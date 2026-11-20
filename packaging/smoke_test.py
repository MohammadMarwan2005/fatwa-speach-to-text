"""Checks run on each build machine before an installer is made.

1. The freshly built whisper-cli transcribes a sample, and our JSON reader turns it into words.
2. The packaged app runs a whole job by itself: convert, Whisper, compare, save (free mode).

usage: python packaging/smoke_test.py <whisper-dir> <model.bin> <sample.wav> [<packaged app executable>]
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["WHISPER_LANGUAGE"] = "en"  # the sample (whisper.cpp's jfk.wav) is English

from pipeline.engines import CliEngine  # noqa: E402


def main() -> int:
    wdir, model, wav = (Path(p).resolve() for p in sys.argv[1:4])
    app = Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else None
    exe = wdir / ("whisper-cli.exe" if os.name == "nt" else "whisper-cli")

    engine = CliEngine(exe, threads=2, gpu=True)
    res = engine.transcribe(model, wav, "", 11.0)
    words = [w["w"] for s in res["segments"] for w in s["words"]]
    print("whisper-cli device:", engine.device)
    print("whisper-cli words:", " ".join(words))
    assert len(words) >= 5, "whisper-cli produced too few words"
    assert all(0.0 <= w["p"] <= 1.0 for s in res["segments"] for w in s["words"]), "bad confidence values"

    if app:
        from pipeline.job import create_job
        d = create_job("sample.wav", src=wav, options={"model": "none", "runs": 2})
        env = {**os.environ, "WHISPER_MODEL_PATH": str(model), "WHISPER_LANGUAGE": "en"}
        proc = subprocess.run([str(app), "--run-job", str(d)], env=env, timeout=1200)
        st = json.loads((d / "state.json").read_text(encoding="utf-8"))
        for s in st["steps"]:
            print(f"  {s['status']:>8}  {s['title']}: {s['detail']}")
        if st["status"] != "done":
            print((d / "log.txt").read_text(errors="replace")[-3000:] if (d / "log.txt").exists() else "")
        assert proc.returncode == 0 and st["status"] == "done", f"packaged app job failed: {st.get('error')}"
        text = (d / "transcript.txt").read_text(encoding="utf-8")
        print("packaged app transcript:", text[:300].replace("\n", " "))
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
