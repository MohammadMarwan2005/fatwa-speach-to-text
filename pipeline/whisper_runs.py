"""Step 3: the Whisper runs, the prompt, and turning Whisper's tokens into words with timing and confidence.

How Whisper is executed (bundled whisper-cli with GPU support, or the Python binding) lives in engines.py.
"""
import re
from pathlib import Path
from typing import Callable, Optional

# Opening formulas of a sermon or fatwa. Steers Whisper towards classical vocabulary and punctuation.
RELIGIOUS_PROMPT = (
    "بسم الله الرحمن الرحيم، الحمد لله رب العالمين، والصلاة والسلام على رسول الله، "
    "وعلى آله وصحبه أجمعين. قال الله تعالى في كتابه الكريم، وقال النبي صلى الله عليه وسلم."
)

# The three runs. Their disagreements show where Whisper is likely wrong.
RUNS = [
    {"id": "A", "label": "With religious prompt", "audio": "plain", "prompt": RELIGIOUS_PROMPT},
    {"id": "B", "label": "No prompt", "audio": "plain", "prompt": ""},
    {"id": "C", "label": "Cleaned audio, with prompt", "audio": "clean", "prompt": RELIGIOUS_PROMPT},
]

_PUNCT_ONLY = re.compile(r"^[\s،؛؟۔.,!?:;\"'«»()\-…]+$")


def merge_tokens(tokens: list, seg_t0: float, seg_t1: float) -> list:
    """tokens: [(bytes, p, t0_seconds, t1_seconds)] without special tokens.
    A token starting with a space starts a new word. A word's confidence is its least sure token,
    not counting punctuation tokens (an unsure comma shouldn't flag the word before it)."""
    words, cur = [], None
    for b, p, t0, t1 in tokens:
        punct = bool(_PUNCT_ONLY.match(b.decode("utf-8", "ignore") or " "))
        if cur is None or b.startswith(b" "):
            if cur:
                words.append(cur)
            cur = {"b": bytearray(b), "p": [], "s": t0, "e": t1}
        else:
            cur["b"] += b
            cur["e"] = t1
        if not punct:
            cur["p"].append(p)
    if cur:
        words.append(cur)
    out = []
    for x in words:
        text = bytes(x["b"]).decode("utf-8", "replace").strip()
        if text:
            out.append({"w": text, "p": round(min(x["p"]) if x["p"] else 1.0, 3),
                        "s": round(x["s"], 2), "e": round(x["e"], 2)})
    _fix_word_times(out, seg_t0, seg_t1)
    return out


def _fix_word_times(words: list, seg_t0: float, seg_t1: float) -> None:
    """If token timestamps are missing or out of the segment, spread words over the segment by length."""
    ok = all(seg_t0 - 0.5 <= w["s"] <= w["e"] <= seg_t1 + 0.5 and w["e"] > 0 for w in words)
    if ok or not words:
        return
    total = sum(len(w["w"]) for w in words) or 1
    pos = 0
    for w in words:
        w["s"] = round(seg_t0 + (seg_t1 - seg_t0) * pos / total, 2)
        pos += len(w["w"])
        w["e"] = round(seg_t0 + (seg_t1 - seg_t0) * pos / total, 2)


# --- Python binding (pywhispercpp, CPU). Used when running from source without a bundled whisper-cli. ---

def load_model(model_path: Path, threads: int):
    from pywhispercpp.model import Model
    return Model(str(model_path), params_sampling_strategy=1, n_threads=threads,
                 print_progress=False, print_realtime=False)


def transcribe(model, wav: Path, prompt: str, duration: float,
               progress: Optional[Callable[[float], None]] = None) -> dict:
    import _pywhispercpp as pw
    from .audio import load_wav
    from .config import whisper_language

    def on_segment(seg):
        if progress and duration > 0:
            progress(min(1.0, (seg.t1 / 100) / duration))

    segs = model.transcribe(
        load_wav(wav),
        language=whisper_language(),
        initial_prompt=prompt,  # always set: pywhispercpp keeps the previous call's value otherwise
        beam_search={"beam_size": 5, "patience": -1.0},
        greedy={"best_of": 5},
        token_timestamps=True,
        no_context=False,
        new_segment_callback=on_segment,
    )
    ctx = model._ctx
    eot = pw.whisper_token_eot(ctx)
    segments = []
    for i, s in enumerate(segs):
        t0, t1 = s.t0 / 100, s.t1 / 100
        tokens = []
        for j in range(pw.whisper_full_n_tokens(ctx, i)):
            tid = pw.whisper_full_get_token_id(ctx, i, j)
            if tid >= eot:  # special and timestamp tokens
                continue
            data = pw.whisper_full_get_token_data(ctx, i, j)
            tokens.append((pw.whisper_token_to_bytes(ctx, tid), float(pw.whisper_full_get_token_p(ctx, i, j)),
                           data.t0 / 100, data.t1 / 100))
        words = merge_tokens(tokens, t0, t1)
        if words:
            segments.append({"start": round(t0, 2), "end": round(t1, 2), "text": s.text.strip(), "words": words})
    return {"segments": segments}
