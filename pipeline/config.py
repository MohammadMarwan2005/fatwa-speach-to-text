"""Paths, model list, prices and the settings read from config.env."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "jobs"
WEB_DIR = ROOT / "web"
CONFIG_FILE = ROOT / "config.env"

# Whisper large-v3, 5-bit quantized, for whisper.cpp. Shared with other tools that use pywhispercpp.
WHISPER_MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_0.bin"
WHISPER_MODEL_SIZE = 1_081_140_203
WHISPER_MODEL_SHA256 = "d75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1"
DEFAULT_WHISPER_MODEL_PATH = Path.home() / ".local/share/pywhispercpp/models/ggml-large-v3-q5_0.bin"

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
        value = value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def setting(name: str, default: str = "") -> str:
    """config.env wins; the process environment is the fallback."""
    value = read_config_file().get(name, "")
    return value or os.environ.get(name, "") or default


def api_key() -> str:
    key = setting("ANTHROPIC_API_KEY")
    return "" if key.startswith("put-your") else key


def whisper_model_path() -> Path:
    return Path(setting("WHISPER_MODEL_PATH", str(DEFAULT_WHISPER_MODEL_PATH))).expanduser()


def whisper_threads() -> int:
    try:
        return max(1, int(setting("WHISPER_THREADS", str(os.cpu_count() or 4))))
    except ValueError:
        return os.cpu_count() or 4


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
