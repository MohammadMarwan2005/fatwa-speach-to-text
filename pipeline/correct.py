"""Step 5: Claude reads the lined-up Whisper runs and writes the corrected transcript.

The API key is read from config.env on the server side only; it is never sent to the browser.
"""
import json
from typing import Callable, Optional

import anthropic

from . import config
from .outputs import clock

CHUNK_LINES = 120       # lines per request (about 8-10 minutes of speech), keeps each answer well under the cap
CONTEXT_LINES = 3       # corrected lines from the previous chunk, shown as read-only context
MAX_TOKENS = 64000      # streaming, so a large cap doesn't hit HTTP timeouts
FALLBACK_BETA = "server-side-fallback-2026-07-01"

KINDS = ["spelling", "word_boundary", "runs_disagreed", "fixed_phrase", "low_confidence_outlier",
         "quran", "hadith", "orthography", "other"]

SYSTEM_PROMPT = """You correct Arabic speech-to-text output. The recordings are mostly Islamic lessons, sermons, questions and fatwas, in Modern Standard Arabic or dialect.

You receive several machine transcriptions ("runs") of the same recording made by Whisper large-v3, already lined up line by line, with the words Whisper was unsure about. Write the most faithful transcript of what was actually said.

The runs:
- The base run gives each line's timing and its words are the reference you compare against. When it is run B, it was made without a prompt, so it has no punctuation.
- Run A was made with a prompt of religious opening formulas. It has punctuation, but the prompt can push in formula words that were not said.
- Run C, if present, used noise-reduced audio and is often a little worse. Treat it as a weak extra opinion.

Rules:
1. Faithful, not edited. Write what the speaker said. Don't paraphrase, summarize, reorder, or correct the speaker's grammar or dialect. Don't add words that weren't said.
2. Fix Whisper's spelling errors. Common ones: ذ written as د (معودات → معوِّذات), ث as ت or س, ظ as ض or ز, a missing or wrong hamza.
3. Fix wrong word boundaries. Whisper often splits or joins words differently from what was said (شري عليك or شريع لك → شُرِع لك; يمنع أنك or يمنع عنك → يمنعُك). Read the runs as sounds and choose the real words those sounds make.
4. When the runs disagree, choose the reading that best explains all of them and fits the meaning. It may be a third form that none of them wrote.
5. A word that only one run has, with low confidence, is probably wrong.
6. Restore fixed phrases to their standard form when the sounds fit: أحسن الله إليكم، صلى الله عليه وسلم، رضي الله عنه، جزاك الله خيرًا، بارك الله فيكم.
7. Quran: when a verse is recited, write it exactly as in the Mushaf, between ﴿ ﴾. If the speaker clearly recited it differently, keep what was said and add a note.
8. Hadith and other quotations: put the quoted words between « ». Don't replace them with a version from memory unless the sounds match it.
9. Possible slips by the speaker: if all runs agree on a word that looks like a slip, keep it as said and add a note with the likely intended word.
10. Use Arabic punctuation (، ؛ ؟ . :). Add vowel marks only where a word would be ambiguous without them (شُرِع، يُفعَل), and full vowel marks inside Quran verses.
11. Parts: if the recording has clear parts, such as a question read out and then the answer, set `section` on the first line of each part (for example السؤال or الجواب). Leave it empty everywhere else.
12. If no reading is reasonably supported, write your best guess and add a note that starts with "Unclear:".
13. List in `corrections` every place where your words differ from the base run's words. Changes of punctuation or vowel marks alone don't need an entry. `heard` is the base run's words there, `wrote` is your words exactly as they appear in your line text, `runs_supporting` lists the runs that had your reading (empty if no run had it), and `reason` is one short English sentence.
14. Return every line id you were asked for exactly once, in order. Keep each line to what was said in its time span; move a word to a neighbouring line only when the runs show it belongs there.
15. `summary`: one short English sentence on what this part of the recording is about."""

SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "section": {"type": "string"}, "text": {"type": "string"}},
            "required": ["id", "section", "text"], "additionalProperties": False}},
        "corrections": {"type": "array", "items": {
            "type": "object",
            "properties": {"line_id": {"type": "integer"}, "heard": {"type": "string"}, "wrote": {"type": "string"},
                           "kind": {"type": "string", "enum": KINDS},
                           "runs_supporting": {"type": "array", "items": {"type": "string"}},
                           "reason": {"type": "string"}},
            "required": ["line_id", "heard", "wrote", "kind", "runs_supporting", "reason"],
            "additionalProperties": False}},
        "notes": {"type": "array", "items": {
            "type": "object",
            "properties": {"line_id": {"type": "integer"}, "note": {"type": "string"}},
            "required": ["line_id", "note"], "additionalProperties": False}},
        "summary": {"type": "string"},
    },
    "required": ["lines", "corrections", "notes", "summary"],
    "additionalProperties": False,
}


class CorrectionError(Exception):
    """A readable reason the AI step could not finish."""


def _api_message(e: anthropic.APIStatusError) -> str:
    """The API's own one-line message, without the status code and JSON around it."""
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict) and inner.get("message"):
            return str(inner["message"])
    return str(getattr(e, "message", e))


def price_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    info = config.CLAUDE_MODELS.get(model)
    if info:
        long = tokens_in > info.get("long_threshold", float("inf"))
        p_in, p_out = (info["in_long"], info["out_long"]) if long else (info["in"], info["out"])
    else:
        p_in, p_out = config.FALLBACK_PRICES.get(model, (0.0, 0.0))
    return tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out


def _payload(chunk: list, run_labels: dict, base: str, context: list) -> str:
    lines = []
    for ln in chunk:
        item = {"id": ln["id"], "time": f"{clock(ln['start'])}-{clock(ln['end'])}",
                "base": ln["base"], "other_runs": ln["alts"]}
        if ln["low_conf"]:
            item["unsure_words"] = [f"{f['w']} (run {f['run']}, {f['p']:.2f})" for f in ln["low_conf"]]
        if ln["diffs"]:
            item["disagreements"] = [f"base: {d['base'] or '—'} | run {d['run']}: {d['alt'] or '—'}" for d in ln["diffs"]]
        lines.append(item)
    data = {"runs": run_labels, "base_run": base, "lines": lines}
    if context:
        data["already_corrected_previous_lines"] = context
    ids = f"{chunk[0]['id']}-{chunk[-1]['id']}" if len(chunk) > 1 else str(chunk[0]["id"])
    return (f"Correct lines {ids}. Return each of these ids exactly once.\n\n"
            + json.dumps(data, ensure_ascii=False, indent=1))


def _call(client, model: str, effort: str, text: str, expected_chars: int,
          progress: Optional[Callable[[float], None]]):
    params = dict(
        model=model, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
    )
    if config.CLAUDE_MODELS[model]["fallbacks"]:
        # If the model declines, the API re-runs the request on a fallback model in the same call.
        stream_ctx = client.beta.messages.stream(**params, betas=[FALLBACK_BETA], fallbacks="default")
    else:
        stream_ctx = client.messages.stream(**params)
    got = 0
    with stream_ctx as stream:
        for piece in stream.text_stream:
            got += len(piece)
            if progress and expected_chars:
                progress(min(0.99, got / expected_chars))
        return stream.get_final_message()


def correct(compared: dict, run_labels: dict, model: str, effort: str,
            progress: Optional[Callable[[float, str], None]] = None) -> dict:
    key = config.api_key()
    if not key:
        raise CorrectionError("No API key. Put it in config.env as ANTHROPIC_API_KEY=...")
    if model not in config.CLAUDE_MODELS:
        raise CorrectionError(f"Unknown model {model}.")
    client = anthropic.Anthropic(api_key=key, max_retries=3, timeout=900)
    base = compared["base_run"]
    all_lines = compared["lines"]
    chunks = [all_lines[i:i + CHUNK_LINES] for i in range(0, len(all_lines), CHUNK_LINES)]
    out_lines, corrections, notes, summaries = [], [], [], []
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "requests": 0, "served_by": []}

    for n, chunk in enumerate(chunks):
        context = [{"id": ln["id"], "text": ln["text"]} for ln in out_lines[-CONTEXT_LINES:]]
        text = _payload(chunk, run_labels, base, context)
        expected = int(sum(len(ln["base"]) for ln in chunk) * 2.2) + 400

        def chunk_progress(f, n=n):
            if progress:
                progress((n + f) / len(chunks), f"part {n + 1} of {len(chunks)}" if len(chunks) > 1 else "")

        try:
            msg = _call(client, model, effort, text, expected, chunk_progress)
        except anthropic.AuthenticationError:
            raise CorrectionError("The API key was rejected. Check ANTHROPIC_API_KEY in config.env.")
        except anthropic.PermissionDeniedError:
            raise CorrectionError(f"This API key isn't allowed to use {model}.")
        except anthropic.NotFoundError:
            raise CorrectionError(f"The model {model} wasn't found for this API key.")
        except anthropic.RateLimitError:
            raise CorrectionError("The API is rate limiting this key. Wait a minute and try again.")
        except anthropic.BadRequestError as e:
            msg = _api_message(e)
            if "credit balance" in msg.lower():
                raise CorrectionError("Your API account has no credit. Add credit at platform.claude.com "
                                      "(Billing), then use “Run correction again”.")
            raise CorrectionError(f"The API rejected the request: {msg}")
        except anthropic.APIStatusError as e:
            raise CorrectionError(f"The API returned an error ({e.status_code}): {_api_message(e)} Try again later.")
        except anthropic.APIConnectionError:
            raise CorrectionError("Couldn't reach the Anthropic API. Check the internet connection.")

        if msg.stop_reason == "refusal":
            cat = getattr(getattr(msg, "stop_details", None), "category", None)
            raise CorrectionError(f"Claude declined to process this recording{f' ({cat})' if cat else ''}.")
        if msg.stop_reason == "max_tokens":
            raise CorrectionError("Claude's answer was cut off before it finished.")
        text_block = next((b for b in msg.content if getattr(b, "type", "") == "text"), None)
        if text_block is None:
            raise CorrectionError("Claude returned no transcript text.")
        try:
            data = json.loads(text_block.text)
        except json.JSONDecodeError:
            raise CorrectionError("Claude's answer wasn't valid JSON.")

        u = msg.usage
        tin = (u.input_tokens or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0) \
            + (getattr(u, "cache_read_input_tokens", 0) or 0)
        tout = u.output_tokens or 0
        usage["input_tokens"] += tin
        usage["output_tokens"] += tout
        usage["cost_usd"] += price_usd(msg.model, tin, tout)
        usage["requests"] += 1
        if msg.model not in usage["served_by"]:
            usage["served_by"].append(msg.model)

        got = {ln["id"]: ln for ln in data["lines"]}
        ids = {ln["id"] for ln in chunk}
        for ln in chunk:
            g = got.get(ln["id"])
            if g is None:
                notes.append({"line_id": ln["id"], "note": "Claude didn't return this line; Whisper's text is shown."})
            out_lines.append({"id": ln["id"], "start": ln["start"], "end": ln["end"],
                              "section": (g or {}).get("section", "").strip(),
                              "text": ((g or {}).get("text") or ln["base"]).strip()})
        corrections += [c for c in data["corrections"] if c["line_id"] in ids]
        notes += [x for x in data["notes"] if x["line_id"] in ids]
        if data["summary"].strip():
            summaries.append(data["summary"].strip())

    usage["cost_usd"] = round(usage["cost_usd"], 4)
    return {"mode": "corrected", "lines": out_lines, "corrections": corrections, "notes": notes,
            "summary": " ".join(summaries), "usage": usage, "model": model, "effort": effort}
