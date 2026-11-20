"""How Whisper is run, picked automatically.

cli     The whisper.cpp program bundled with the installed app (whisper-cli). It uses the GPU when it can:
        Vulkan on Windows and Linux (NVIDIA, AMD and Intel graphics), Metal on Apple Silicon Macs. Without a
        usable GPU it loads the fastest CPU code the processor supports (AVX, AVX2, AVX-512, ARM NEON).
        Each run is its own process, so runs never share context.
python  The pywhispercpp binding (CPU only). Used when running from source without a bundled whisper-cli.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

from . import config
from .whisper_runs import merge_tokens

FIRST_SPECIAL_ID = 50257  # Whisper's end-of-text token; everything from here up is special or a timestamp
_SPECIAL_TEXT = re.compile(r"^(\[_.*\]|<\|.*\|>)$")
_PROGRESS = re.compile(r"progress\s*=\s*(\d+)%")
_DEVICE_PATTERNS = [
    ("Vulkan", re.compile(r"ggml_vulkan:\s*\d+\s*=\s*([^|(]+)")),
    ("Metal", re.compile(r"GPU name:\s*(.+)")),
    ("Metal", re.compile(r"ggml_metal\w*:\s*found device:\s*(.+)")),
    ("CUDA", re.compile(r"Device\s+\d+:\s*([^,]+),\s*compute capability")),
]
_CPU_VARIANT = re.compile(r"libggml-cpu-([\w]+)|ggml-cpu-([\w]+)\.dll")
_USING_BACKEND = re.compile(r"using\s+([A-Za-z]+)\d*\s+backend")


def cli_path() -> Optional[Path]:
    exe = "whisper-cli.exe" if sys.platform == "win32" else "whisper-cli"
    custom = config.setting("WHISPER_CLI")
    candidates = [Path(custom).expanduser()] if custom else []
    candidates += [config.RESOURCES / "whisper" / exe, config.RESOURCES / "whisper" / "bin" / exe]
    found = next((c for c in candidates if c.is_file()), None)
    if found and sys.platform != "win32" and not os.access(found, os.X_OK):
        try:  # packaging can drop the executable bit
            found.chmod(found.stat().st_mode | 0o111)
        except OSError:
            pass
    return found


class CliEngine:
    name = "whisper.cpp"

    def __init__(self, exe: Path, threads: int, gpu: bool):
        self.exe, self.threads, self.gpu = exe, threads, gpu
        self.device = "starting"

    def _args(self, model: Path, wav: Path, prompt: str, out_base: Path, gpu: bool) -> list:
        args = [str(self.exe), "-m", str(model), "-f", str(wav), "-l", config.whisper_language(), "-bs", "5", "-bo", "5",
                "-t", str(self.threads), "-ojf", "-of", str(out_base), "-pp"]
        if prompt:
            args += ["--prompt", prompt]
        if not gpu:
            args.append("-ng")
        return args

    def _run(self, args: list, progress) -> tuple:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                creationflags=flags, text=True, encoding="utf-8", errors="replace")
        log = []
        for line in proc.stderr:
            log.append(line.rstrip())
            m = _PROGRESS.search(line)
            if m and progress:
                progress(min(1.0, int(m.group(1)) / 100))
        return proc.wait(), log

    def _describe(self, log: list, gpu_requested: bool) -> str:
        text = "\n".join(log)
        backend = next((b for b in _USING_BACKEND.findall(text) if b.upper() != "CPU"), "")
        for kind, pat in _DEVICE_PATTERNS:
            m = pat.search(text)
            if m and gpu_requested and (not backend or backend.lower().startswith(kind.lower()[:3])):
                return f"GPU ({kind}): {m.group(1).strip()}"
        if backend and gpu_requested:
            return f"GPU ({backend})"
        variants = [a or b for a, b in _CPU_VARIANT.findall(text)]
        cpu = f"CPU ({variants[-1]})" if variants else "CPU"
        return f"{cpu}, {self.threads} threads"

    def transcribe(self, model: Path, wav: Path, prompt: str, duration: float,
                   progress: Optional[Callable[[float], None]] = None) -> dict:
        with tempfile.TemporaryDirectory(prefix="whisper-") as tmp:
            out_base = Path(tmp) / "out"
            code, log = self._run(self._args(model, wav, prompt, out_base, self.gpu), progress)
            used_gpu = self.gpu
            if code != 0 and self.gpu:  # a broken GPU driver shouldn't stop the job: retry on the CPU
                code, log = self._run(self._args(model, wav, prompt, out_base, False), progress)
                used_gpu = False
            if code != 0 or not out_base.with_suffix(".json").exists():
                tail = " | ".join(line for line in log[-6:] if line.strip())
                raise RuntimeError(f"whisper-cli failed (exit {code}): {tail[:300]}")
            self.device = self._describe(log, used_gpu) + ("" if used_gpu or not self.gpu else " (GPU failed)")
            data = json.loads(out_base.with_suffix(".json").read_text(encoding="utf-8", errors="replace"), strict=False)
        return parse_cli_json(data)


def parse_cli_json(data: dict) -> dict:
    """whisper-cli -ojf output -> our run format (segments with words, timing and confidence)."""
    segments = []
    for seg in data.get("transcription", []):
        t0, t1 = seg["offsets"]["from"] / 1000, seg["offsets"]["to"] / 1000
        tokens = []
        for tok in seg.get("tokens", []):
            text = tok.get("text", "")
            if tok.get("id", 0) >= FIRST_SPECIAL_ID or _SPECIAL_TEXT.match(text.strip() or "x"):
                continue
            off = tok.get("offsets") or {}
            tokens.append((text.encode("utf-8"), float(tok.get("p", 1.0)),
                           off.get("from", -1000) / 1000, off.get("to", -1000) / 1000))
        words = merge_tokens(tokens, t0, t1)
        if words:
            segments.append({"start": round(t0, 2), "end": round(t1, 2), "text": seg.get("text", "").strip(),
                             "words": words})
    return {"segments": segments}


class PythonEngine:
    name = "pywhispercpp"

    def __init__(self, threads: int):
        self.threads = threads
        self.device = f"CPU, {threads} threads"

    def transcribe(self, model: Path, wav: Path, prompt: str, duration: float,
                   progress: Optional[Callable[[float], None]] = None) -> dict:
        from . import whisper_runs
        # A fresh model per run: whisper.cpp otherwise feeds the previous run's text in as context.
        m = whisper_runs.load_model(model, self.threads)
        try:
            return whisper_runs.transcribe(m, wav, prompt, duration, progress)
        finally:
            del m


def get_engine():
    mode = config.whisper_engine()
    exe = cli_path()
    threads = config.whisper_threads()
    if exe and mode in ("auto", "cli"):
        return CliEngine(exe, threads, config.use_gpu())
    if mode == "cli":
        raise RuntimeError("WHISPER_ENGINE=cli but no whisper-cli program was found.")
    return PythonEngine(threads)


def describe_available() -> str:
    """Short text for the page: which engine will run."""
    exe = cli_path()
    if exe and config.whisper_engine() in ("auto", "cli"):
        return "whisper.cpp, GPU when available" if config.use_gpu() else "whisper.cpp, CPU only"
    return f"Python binding, CPU, {config.whisper_threads()} threads"
