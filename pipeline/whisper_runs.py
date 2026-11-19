"""Step 3: run Whisper large-v3 (whisper.cpp) and keep every word with its timing and confidence."""
import re
from pathlib import Path
from typing import Callable, Optional

import _pywhispercpp as pw
from pywhispercpp.model import Model

from .audio import load_wav

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


def load_model(model_path: Path, threads: int) -> Model:
    return Model(str(model_path), params_sampling_strategy=1, n_threads=threads,
                 print_progress=False, print_realtime=False)


_PUNCT_ONLY = re.compile(r"^[\s،؛؟۔.,!?:;\"'«»()\-…]+$")


def _words_of_segment(ctx, i: int, seg_t0: float, seg_t1: float, eot: int) -> list:
    """Join Whisper's sub-word tokens into words; a token starting with a space starts a new word.
    A word's confidence is its least sure token, not counting punctuation tokens (an unsure comma
    shouldn't flag the word before it)."""
    words, cur = [], None
    for j in range(pw.whisper_full_n_tokens(ctx, i)):
        tid = pw.whisper_full_get_token_id(ctx, i, j)
        if tid >= eot:  # special and timestamp tokens
            continue
        data = pw.whisper_full_get_token_data(ctx, i, j)
        b = pw.whisper_token_to_bytes(ctx, tid)
        p = float(pw.whisper_full_get_token_p(ctx, i, j))
        punct = bool(_PUNCT_ONLY.match(b.decode("utf-8", "ignore") or " "))
        t0, t1 = data.t0 / 100, data.t1 / 100
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


def transcribe(model: Model, wav: Path, prompt: str, duration: float,
               progress: Optional[Callable[[float], None]] = None) -> dict:
    audio = load_wav(wav)

    def on_segment(seg):
        if progress and duration > 0:
            progress(min(1.0, (seg.t1 / 100) / duration))

    segs = model.transcribe(
        audio,
        language="ar",
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
        words = _words_of_segment(ctx, i, t0, t1, eot)
        if words:
            segments.append({"start": round(t0, 2), "end": round(t1, 2), "text": s.text.strip(), "words": words})
    return {"segments": segments}
