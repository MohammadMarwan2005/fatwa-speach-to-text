"""Step 6: write the transcript as plain text and as subtitles."""
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


def write_txt(result: dict, title: str, duration: float, how: str, path: Path) -> None:
    out = [f"{title} ({clock(duration)})", how, ""]
    para, prev_end = [], None
    notes_by_line = {}
    for n in result.get("notes", []):
        notes_by_line.setdefault(n["line_id"], []).append(n["note"])
    note_no = 0
    footnotes = []

    def flush():
        if para:
            out.append(" ".join(para))
            out.append("")
            para.clear()

    for ln in result["lines"]:
        if ln.get("section"):
            flush()
            out.append(f"{ln['section']} [{clock(ln['start'])}]")
        elif prev_end is not None and ln["start"] - prev_end >= PARAGRAPH_GAP:
            flush()
        text = ln["text"]
        for note in notes_by_line.get(ln["id"], []):
            note_no += 1
            text += f" ({note_no})"
            footnotes.append(f"({note_no}) [{clock(ln['start'])}] {note}")
        para.append(text)
        prev_end = ln["end"]
    flush()
    if footnotes:
        out += ["Notes:"] + footnotes + [""]
    path.write_text("\n".join(out), encoding="utf-8")


def write_srt(result: dict, path: Path) -> None:
    cues = []
    for i, ln in enumerate(result["lines"], 1):
        cues.append(f"{i}\n{srt_time(ln['start'])} --> {srt_time(ln['end'])}\n{ln['text']}\n")
    path.write_text("\n".join(cues), encoding="utf-8")
