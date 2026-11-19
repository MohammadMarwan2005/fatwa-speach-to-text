"""Step 4: line the runs up by time, flag unsure words, and list where the runs disagree."""
import bisect
import difflib
import re

from .config import LOW_CONFIDENCE

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_PUNCT = re.compile(r"[،؛؟۔.,!?:;\"'«»()\[\]﴾﴿…\-–—]")
_ALEF = re.compile(r"[أإآٱ]")


def normalize(word: str) -> str:
    """Compare words without vowel marks, punctuation or alef variants."""
    return _ALEF.sub("ا", _PUNCT.sub("", _DIACRITICS.sub("", word)))


def base_run_id(run_ids: list) -> str:
    # The no-prompt run splits into shorter segments, which make better subtitle lines.
    return "B" if "B" in run_ids else run_ids[0]


def build_lines(runs: dict) -> dict:
    """runs: {run_id: {"segments": [...]}} in run order. Returns {"base_run", "lines": [...]}."""
    run_ids = list(runs)
    base = base_run_id(run_ids)
    lines = []
    for seg in runs[base]["segments"]:
        lines.append({"id": len(lines) + 1, "start": seg["start"], "end": seg["end"],
                      "base": seg["text"], "words": {r: [] for r in run_ids}})
    if not lines:
        return {"base_run": base, "lines": []}
    starts = [ln["start"] for ln in lines]

    def line_for(t: float) -> dict:
        i = max(0, bisect.bisect_right(starts, t) - 1)
        if i + 1 < len(lines) and t >= lines[i]["end"]:
            # In a gap between lines: pick the nearer one.
            if lines[i + 1]["start"] - t < t - lines[i]["end"]:
                i += 1
        return lines[i]

    for r in run_ids:
        for seg in runs[r]["segments"]:
            for w in seg["words"]:
                if r == base:
                    continue
                line_for((w["s"] + w["e"]) / 2)["words"][r].append(w)
    for ln, seg in zip(lines, runs[base]["segments"]):
        ln["words"][base] = seg["words"]

    for ln in lines:
        ln["alts"] = {r: " ".join(w["w"] for w in ln["words"][r]) for r in run_ids if r != base}
        ln["low_conf"] = [{"run": r, "w": w["w"], "p": w["p"]}
                          for r in run_ids for w in ln["words"][r] if w["p"] < LOW_CONFIDENCE]
        ln["diffs"] = []
        base_tokens = [w["w"] for w in ln["words"][base]]
        for r in run_ids:
            if r != base:
                ln["diffs"] += _diff(base_tokens, [w["w"] for w in ln["words"][r]], r)
        del ln["words"]
    return {"base_run": base, "lines": lines}


def _diff(base_tokens: list, alt_tokens: list, run_id: str) -> list:
    b = [(t, normalize(t)) for t in base_tokens if normalize(t)]
    a = [(t, normalize(t)) for t in alt_tokens if normalize(t)]
    sm = difflib.SequenceMatcher(a=[x[1] for x in b], b=[x[1] for x in a], autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            out.append({"run": run_id, "base": " ".join(x[0] for x in b[i1:i2]),
                        "alt": " ".join(x[0] for x in a[j1:j2])})
    return out


def free_result(compared: dict, run_ids: list) -> dict:
    """Result without the AI step: the prompted run's text (it has punctuation) on the base run's timing,
    with its unsure words flagged for a person to check."""
    base = compared["base_run"]
    text_run = "A" if "A" in run_ids else base
    lines = []
    for ln in compared["lines"]:
        text = ln["base"] if text_run == base else (ln["alts"].get(text_run) or ln["base"])
        source = text_run if text == ln["alts"].get(text_run) else base
        flags = [f for f in ln["low_conf"] if f["run"] == source]
        lines.append({"id": ln["id"], "start": ln["start"], "end": ln["end"], "section": "",
                      "text": text, "text_run": source, "flags": flags})
    return {"mode": "free", "lines": lines, "corrections": [], "notes": [], "summary": ""}
