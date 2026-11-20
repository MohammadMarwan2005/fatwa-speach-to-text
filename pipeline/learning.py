"""Learning from the corrections people make.

Every saved correction (jobs/<id>/edits.json) is compared with the line it replaced. Each word or short
phrase that was changed becomes a learned fix: "heard" -> "wrote". Changes of punctuation or vowel marks
only are ignored, and so are lines that were rewritten almost completely.

The learned fixes are used three ways:
  1. suggestions in new transcripts (one click to apply; nothing changes by itself),
  2. extra words in Whisper's prompt for the prompted run (A), so Whisper hears them right more often,
  3. a list sent to Claude with the correction request, when an API key is used.

Fixes are counted by the number of transcripts they were made in, so saving the same job again doesn't
inflate them. Fixes can be removed (blocked) and shared between users by export/import.
"""
import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

from . import config
from .compare import normalize

MAX_PHRASE = 3          # learn replacements of up to 3 words
MIN_SIMILARITY = 0.4    # below this the line was rewritten, not corrected: don't learn from it
_PUNCT_CLASS = r"[،؛؟۔.,!?:;\"'«»()\[\]﴾﴿…\-–—]"
_EDGE_PUNCT = re.compile(r"^[\s" + _PUNCT_CLASS[1:-1] + r"]+|[\s" + _PUNCT_CLASS[1:-1] + r"]+$")

IMPORTED_FILE = "learned_imported.json"
BLOCKED_FILE = "learned_blocked.json"


def _strip(s: str) -> str:
    return _EDGE_PUNCT.sub("", s)


def _tokens(text: str) -> list:
    return [t for t in text.split() if normalize(t)]


def key_of(heard: str) -> str:
    return " ".join(normalize(t) for t in heard.split() if normalize(t))


def pairs_from_edit(original: str, corrected: str) -> list:
    """[(heard, wrote, context)] for the words a person changed in one line.
    context is the (normalized) word before the change, used to tell apart single words that are
    usually right (عليكم in السلام عليكم) from the places where they were misheard (الله عليكم)."""
    a, b = _tokens(original), _tokens(corrected)
    if not a or not b:
        return []
    sm = SequenceMatcher(a=[normalize(x) for x in a], b=[normalize(x) for x in b], autojunk=False)
    if sm.ratio() < MIN_SIMILARITY:
        return []
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "replace" or i2 - i1 > MAX_PHRASE or j2 - j1 > MAX_PHRASE:
            continue
        heard, wrote = _strip(" ".join(a[i1:i2])), _strip(" ".join(b[j1:j2]))
        if heard and wrote and key_of(heard) != key_of(wrote):
            out.append((heard, wrote, normalize(a[i1 - 1]) if i1 > 0 else ""))
    return out


def _windows(text: str):
    """(key, previous word) for every 1-3 word window of a line."""
    toks = [t for t in text.split()]
    norm = [normalize(t) for t in toks]
    for i in range(len(toks)):
        for n in range(1, MAX_PHRASE + 1):
            window = [x for x in norm[i:i + n] if x]
            if len(window) == n:
                yield " ".join(window), (norm[i - 1] if i > 0 else "")


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def blocked() -> set:
    return set(_read(config.DATA / BLOCKED_FILE, {"keys": []}).get("keys", []))


def block(key: str) -> None:
    keys = blocked() | {key}
    _write(config.DATA / BLOCKED_FILE, {"keys": sorted(keys)})
    imported = _read(config.DATA / IMPORTED_FILE, {"fixes": []})
    imported["fixes"] = [f for f in imported.get("fixes", []) if key_of(f.get("heard", "")) != key]
    _write(config.DATA / IMPORTED_FILE, imported)


def unblock(key: str) -> None:
    _write(config.DATA / BLOCKED_FILE, {"keys": sorted(blocked() - {key})})


def collect() -> dict:
    """All learned fixes: {key: {"heard", "wrote": {form: transcripts}, "jobs": set, "imported": n,
    "contexts": set, "kept": n, "updated"}}."""
    agg = {}
    reviewed = []  # final texts of saved transcripts, to count where a "heard" form was kept

    def add(heard, wrote, source, n=1, when=0.0, context=""):
        k = key_of(heard)
        if not k:
            return
        e = agg.setdefault(k, {"heard": heard, "wrote": {}, "jobs": set(), "imported": 0, "updated": 0.0,
                               "contexts": set(), "kept": 0})
        if context:
            e["contexts"].add(context)
        if source is None:
            e["imported"] += n
            e["wrote"][wrote] = e["wrote"].get(wrote, 0) + n
        elif source not in e["jobs"] or wrote not in e.setdefault("_seen", {}).get(source, set()):
            e["jobs"].add(source)
            e.setdefault("_seen", {}).setdefault(source, set()).add(wrote)
            e["wrote"][wrote] = e["wrote"].get(wrote, 0) + 1
        e["updated"] = max(e["updated"], when or 0.0)

    if config.JOBS_DIR.exists():
        for d in sorted(config.JOBS_DIR.iterdir()):
            edits = _read(d / "edits.json", None) if d.is_dir() else None
            if not edits or not edits.get("lines"):
                continue
            payload = _read(d / "result.json", None)
            if not payload:
                continue
            original = {str(ln["id"]): ln["text"] for ln in payload["result"]["lines"]}
            for line_id, text in edits["lines"].items():
                for heard, wrote, context in pairs_from_edit(original.get(line_id, ""), text):
                    add(heard, wrote, d.name, when=edits.get("saved_at") or 0.0, context=context)
            reviewed += [edits["lines"].get(lid, text) for lid, text in original.items()]
    for f in _read(config.DATA / IMPORTED_FILE, {"fixes": []}).get("fixes", []):
        if f.get("heard") and f.get("wrote"):
            add(f["heard"], f["wrote"], None, n=max(1, int(f.get("count", 1))), when=f.get("updated", 0.0))
            for c in f.get("contexts", []):
                agg[key_of(f["heard"])]["contexts"].add(c)
    for k in blocked():
        agg.pop(k, None)
    # A "heard" form still present in a saved transcript was kept by the person: evidence against the fix.
    for text in reviewed:
        for k, _prev in _windows(text):
            if k in agg:
                agg[k]["kept"] += 1
    for e in agg.values():
        e.pop("_seen", None)
    return agg


def fixes() -> list:
    """Learned fixes for the page, most frequent first."""
    out = []
    for k, e in collect().items():
        wrote, n = max(e["wrote"].items(), key=lambda kv: kv[1])
        count = len(e["jobs"]) + e["imported"]
        out.append({"key": k, "heard": e["heard"], "wrote": wrote, "count": count,
                    "transcripts": len(e["jobs"]), "imported": e["imported"], "updated": e["updated"],
                    "kept": e["kept"], "confidence": round(count / (count + e["kept"]), 2),
                    "contexts": sorted(e["contexts"]),
                    "alternatives": {w: c for w, c in e["wrote"].items() if w != wrote}})
    out.sort(key=lambda f: (-f["count"], f["heard"]))
    return out


def suggest(lines: list, learned: list = None) -> dict:
    """{line_id: [{"heard", "replacement", "wrote", "count", "sure", "same_context", ...}]} where a line still
    has a learned "heard". "sure" ones may be applied all at once; the others are only offered one by one."""
    learned = learned if learned is not None else fixes()
    by_key = {f["key"]: f for f in learned}
    out = {}
    for ln in lines:
        toks = ln["text"].split()
        norm = [normalize(t) for t in toks]
        i, found = 0, []
        while i < len(toks):
            hit = None
            for n in range(MAX_PHRASE, 0, -1):
                window = [x for x in norm[i:i + n] if x]
                if len(window) != n:
                    continue
                f = by_key.get(" ".join(window))
                if f and key_of(f["wrote"]) != " ".join(window):
                    hit = (n, f)
                    break
            if hit:
                n, f = hit
                exact = " ".join(toks[i:i + n])
                lead_m = re.match(_PUNCT_CLASS + r"+", toks[i])
                tail_m = re.search(_PUNCT_CLASS + r"+$", toks[i + n - 1])
                lead, tail = (lead_m.group(0) if lead_m else ""), (tail_m.group(0) if tail_m else "")
                prev = norm[i - 1] if i > 0 else ""
                same_context = prev in f["contexts"]
                # Sure: mostly changed, never mostly kept, and for single words seen in the same context.
                sure = f["confidence"] >= 0.8 and (n > 1 or same_context) and (f["count"] >= 2 or same_context)
                found.append({"heard": exact, "replacement": f"{lead}{f['wrote']}{tail}", "wrote": f["wrote"],
                              "count": f["count"], "kept": f["kept"], "key": f["key"],
                              "same_context": same_context, "sure": sure})
                i += n
            else:
                i += 1
        if found:
            out[str(ln["id"])] = found
    return out


def prompt_terms(limit: int = 25, max_chars: int = 350) -> list:
    """The most frequently corrected-to words, for Whisper's prompt."""
    terms, total = [], 0
    for f in fixes():
        w = f["wrote"]
        if w in terms or total + len(w) + 2 > max_chars:
            continue
        terms.append(w)
        total += len(w) + 2
        if len(terms) >= limit:
            break
    return terms


def export_data() -> dict:
    return {"app": config.APP_NAME, "kind": "learned-fixes", "version": 1, "exported": time.time(),
            "fixes": [{"heard": f["heard"], "wrote": f["wrote"], "count": f["count"], "updated": f["updated"],
                       "contexts": f["contexts"]} for f in fixes()]}


def import_data(data: dict) -> int:
    """Merge fixes exported by another user. Returns how many were added or updated."""
    incoming = [f for f in data.get("fixes", []) if isinstance(f, dict) and f.get("heard") and f.get("wrote")]
    path = config.DATA / IMPORTED_FILE
    current = _read(path, {"fixes": []}).get("fixes", [])
    index = {(key_of(f["heard"]), f["wrote"]): f for f in current}
    n = 0
    for f in incoming:
        k = (key_of(str(f["heard"])), str(f["wrote"]))
        if not k[0]:
            continue
        entry = index.setdefault(k, {"heard": str(f["heard"]), "wrote": str(f["wrote"]), "count": 0, "updated": 0,
                                     "contexts": []})
        entry["count"] = max(int(entry.get("count", 0)), int(f.get("count", 1) or 1))
        entry["updated"] = max(float(entry.get("updated", 0) or 0), float(f.get("updated", 0) or 0))
        entry["contexts"] = sorted(set(entry.get("contexts", [])) | {str(c) for c in f.get("contexts", [])})
        n += 1
    _write(path, {"fixes": list(index.values())})
    return n
