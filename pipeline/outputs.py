"""Step 6: write the transcript as plain text and as subtitles, and apply the user's hand edits."""
import json
from pathlib import Path

PARAGRAPH_GAP = 1.5  # seconds of silence that start a new paragraph


def clock(t: float) -> str:
    t = max(0, int(t))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def srt_time(t: float) -> str:
    ms = max(0, int(round(t * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def apply_edits(result: dict, edits: dict) -> dict:
    """A copy of the result with the user's corrected lines ({"<line id>": "text"}) put in."""
    out = json.loads(json.dumps(result))
    for ln in out["lines"]:
        text = edits.get(str(ln["id"]))
        if text is not None:
            ln["text"] = text.strip()
            ln["flags"] = []  # the person checked this line
            ln["edited"] = True
    return out


def _blocks(result: dict, with_notes: bool):
    """Yield (kind, text) blocks: section headings and paragraphs (a pause starts a new paragraph)."""
    notes_by_line = {}
    if with_notes:
        for n in result.get("notes", []):
            notes_by_line.setdefault(n["line_id"], []).append(n["note"])
    para, prev_end, footnotes = [], None, []
    for ln in result["lines"]:
        if ln.get("section") or (prev_end is not None and ln["start"] - prev_end >= PARAGRAPH_GAP):
            if para:
                yield "para", " ".join(para)
                para = []
        if ln.get("section"):
            yield "section", f"{ln['section']} [{clock(ln['start'])}]" if with_notes else ln["section"]
        text = ln["text"]
        for note in notes_by_line.get(ln["id"], []):
            footnotes.append(f"({len(footnotes) + 1}) [{clock(ln['start'])}] {note}")
            text += f" ({len(footnotes)})"
        if text:
            para.append(text)
        prev_end = ln["end"]
    if para:
        yield "para", " ".join(para)
    if footnotes:
        yield "notes", "Notes:\n" + "\n".join(footnotes)


def plain_text(result: dict) -> str:
    """Just the transcript: headings and paragraphs, no title, notes or marks. Used for "Save .txt"."""
    return "\n\n".join(text for _, text in _blocks(result, with_notes=False)) + "\n"


def write_txt(result: dict, title: str, duration: float, how: str, path: Path) -> None:
    head = f"{title} ({clock(duration)})\n{how}\n\n"
    body = "\n\n".join(text for _, text in _blocks(result, with_notes=True))
    path.write_text(head + body + "\n", encoding="utf-8")


def write_srt(result: dict, path: Path) -> None:
    cues = []
    for i, ln in enumerate(result["lines"], 1):
        cues.append(f"{i}\n{srt_time(ln['start'])} --> {srt_time(ln['end'])}\n{ln['text']}\n")
    path.write_text("\n".join(cues), encoding="utf-8")
