"""Paths, models, prices and the settings read from config.env.

Running from source, everything lives in the project folder. In the installed app (PyInstaller,
"frozen") the program files are read-only, so jobs, config.env and models go to the user's
data folder: %LOCALAPPDATA%\\FatwaTranscript on Windows, ~/Library/Application Support/FatwaTranscript
on macOS, ~/.local/share/fatwa-transcript on Linux.
"""
import os
import shutil
import sys
from pathlib import Path

APP_NAME = "Fatwa Transcript"
FROZEN = bool(getattr(sys, "frozen", False))

# Read-only files that ship with the app (web page, example config, whisper.cpp program).
RESOURCES = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
ROOT = RESOURCES  # kept for older imports


def _user_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "FatwaTranscript"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "FatwaTranscript"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "fatwa-transcript"


DATA = _user_data_dir() if FROZEN else RESOURCES
JOBS_DIR = DATA / "jobs"
WEB_DIR = RESOURCES / "web"
CONFIG_FILE = DATA / "config.env"
MODELS_DIR = (DATA / "models") if FROZEN else Path.home() / ".local/share/pywhispercpp/models"

# Whisper models for whisper.cpp (5-bit quantized). Size and SHA-256 as published on Hugging Face.
WHISPER_MODELS = {
    "large-v3": {
        "label": "Best accuracy (large-v3)",
        "file": "ggml-large-v3-q5_0.bin",
        "size": 1_081_140_203,
        "sha256": "d75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1",
    },
    "large-v3-turbo": {
        "label": "Faster (large-v3-turbo)",
        "file": "ggml-large-v3-turbo-q5_0.bin",
        "size": 574_041_195,
        "sha256": "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
    },
}
DEFAULT_WHISPER_MODEL = "large-v3"
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{file}"

# Claude models offered for the correction step. Prices are USD per million tokens
# (Anthropic API list prices, checked 2026-11). Haiku 5.5 costs more above 100K input tokens.
CLAUDE_MODELS = {
    "claude-opus-5-5": {"label": "Claude Opus 5.5", "in": 4.00, "out": 20.00, "fallbacks": True},
    "claude-sonnet-5-5": {"label": "Claude Sonnet 5.5", "in": 2.00, "out": 10.00, "fallbacks": True},
    "claude-haiku-5-5": {"label": "Claude Haiku 5.5", "in": 0.10, "out": 0.50, "fallbacks": False,
                         "long_threshold": 100_000, "in_long": 0.50, "out_long": 2.50},
}
# Models a server-side fallback may answer with; used only to price the call correctly.
FALLBACK_PRICES = {"claude-opus-5": (5.00, 25.00), "claude-opus-4-8": (5.00, 25.00), "claude-sonnet-5": (2.00, 10.00)}
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high")
DEFAULT_EFFORT = "medium"
FREE_MODE = "none"  # model value meaning "skip the AI step"

LOW_CONFIDENCE = 0.75  # words below this Whisper confidence are flagged


def ensure_data_dir() -> None:
    """Create the data folder and a config.env (from the example) on first start."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        example = RESOURCES / "config.example.env"
        if example.exists():
            shutil.copyfile(example, CONFIG_FILE)


def read_config_file() -> dict:
    """Parse config.env (KEY=VALUE lines, # comments). Read fresh every time so edits apply without a restart."""
    values = {}
    try:
        text = CONFIG_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def setting(name: str, default: str = "") -> str:
    """config.env wins; the process environment is the fallback."""
    value = read_config_file().get(name, "")
    return value or os.environ.get(name, "") or default


def api_key() -> str:
    key = setting("ANTHROPIC_API_KEY")
    return "" if key.startswith("put-your") else key


def whisper_model_name(requested: str = "") -> str:
    name = requested or setting("WHISPER_MODEL", DEFAULT_WHISPER_MODEL)
    return name if name in WHISPER_MODELS else DEFAULT_WHISPER_MODEL


def whisper_model_path(name: str = "") -> Path:
    """WHISPER_MODEL_PATH overrides everything (any ggml model file); otherwise the named model."""
    custom = setting("WHISPER_MODEL_PATH")
    if custom:
        return Path(custom).expanduser()
    return MODELS_DIR / WHISPER_MODELS[whisper_model_name(name)]["file"]


def physical_cores() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def whisper_threads() -> int:
    """Default: one thread per physical core. On a 4-core laptop 4 threads beat 8 (measured)."""
    try:
        return max(1, int(setting("WHISPER_THREADS", "0"))) if setting("WHISPER_THREADS") else physical_cores()
    except ValueError:
        return physical_cores()


def whisper_engine() -> str:
    """auto (bundled whisper-cli if present, else the Python binding), cli, or python."""
    value = setting("WHISPER_ENGINE", "auto").lower()
    return value if value in ("auto", "cli", "python") else "auto"


def whisper_language() -> str:
    """Spoken language for Whisper (ISO code). Arabic unless config.env says otherwise."""
    return setting("WHISPER_LANGUAGE", "ar").lower()


def use_gpu() -> bool:
    return setting("WHISPER_GPU", "on").lower() not in ("off", "0", "no", "false")


def default_claude_model() -> str:
    model = setting("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
    return model if model in CLAUDE_MODELS or model == FREE_MODE else DEFAULT_CLAUDE_MODEL


def default_effort() -> str:
    effort = setting("CLAUDE_EFFORT", DEFAULT_EFFORT)
    return effort if effort in EFFORTS else DEFAULT_EFFORT


def port() -> int:
    try:
        return int(setting("PORT", "8765"))
    except ValueError:
        return 8765
