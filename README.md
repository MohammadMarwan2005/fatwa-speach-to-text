# Fatwa transcript

Turns Arabic voice notes (lessons, sermons, questions and fatwas) into a corrected transcript.
Runs on your own computer: a page in your browser, plus a small local server.

## Start

```bash
./run.sh
```

The first start installs the packages into `.venv` (about a minute). The page opens at <http://127.0.0.1:8765>.
The first transcription downloads the Whisper model (1.1 GB, once) unless it is already in
`~/.local/share/pywhispercpp/models/`.

## Your API key

Put it in **`config.env`** in this folder:

```
ANTHROPIC_API_KEY=sk-ant-...
```

It is read again for every transcription, so you don't need to restart. The key never leaves the server:
the page only learns whether a key is set. `config.env` is listed in `.gitignore`.
Without a key the app still works in free mode (step 5 is skipped).

## What happens to a file

| Step | What | Where it runs | Cost |
|---|---|---|---|
| 1. Get the audio in | The file is copied to `jobs/<id>/` | this computer | free |
| 2. Convert | ffmpeg makes a 16 kHz mono WAV | this computer | free |
| 3. Speech to text | Whisper large-v3 (whisper.cpp), 2 runs: with a religious prompt and without; optional 3rd on cleaned audio. Every word keeps its confidence | this computer, CPU | free |
| 4. Score and compare | Runs are lined up by time; words under 0.75 confidence and disagreements are listed | this computer | free |
| 5. Correct and polish | Claude reads the runs and the flags and writes the corrected text, listing every fix with its reason | Anthropic API | paid, about $1 per audio hour on Opus 5.5 (estimate) |
| 6. Save | `transcript.txt`, `transcript.srt`, `result.json` | this computer | free |

In free mode (no key, or "No AI" chosen), you get the prompted run's text with unsure words marked,
and the page shows what the other runs heard so you can check them by ear: click a line to play it.

## Choosing the model

| Model | Price per million tokens (in / out) | Rough cost per audio hour |
|---|---|---|
| Claude Opus 5.5 (default) | $4 / $20 | about $1 |
| Claude Sonnet 5.5 | $2 / $10 | about $0.50 |
| Claude Haiku 5.5 | $0.10 / $0.50 | about 2 cents |

The exact tokens and cost of each job are shown on the page, from the API's own usage numbers.
Opus and Sonnet requests use the API's server-side fallback: if the model declines a request,
the API retries it on another model in the same call.

## Speed

Whisper runs on the CPU at roughly 2–3× the audio length per run. A 1-minute voice note takes about
5 minutes with 2 runs; a 1-hour lecture would take hours. Long recordings are sent to Claude in parts
of about 120 lines.

## From the terminal

```bash
.venv/bin/python -m pipeline.job --file voice.m4a                # default model from config.env
.venv/bin/python -m pipeline.job --file voice.m4a --model none   # free mode
.venv/bin/python -m pipeline.job --file voice.m4a --model claude-sonnet-5-5 --runs 3
```

Results go to `jobs/<id>/` and show up on the page too.

## Files

```
config.env            your API key and optional settings (not shared)
config.example.env    the same, without a key, to share
run.sh                set up and start
server.py             local server, 127.0.0.1 only, one job at a time
pipeline/
  audio.py            step 2
  whisper_runs.py     step 3, and the religious prompt
  compare.py          step 4
  correct.py          step 5, and the correction rules sent to Claude
  outputs.py          step 6
  job.py              runs the steps and records them in jobs/<id>/state.json
  config.py           paths, models, prices, settings
  model_download.py   downloads and checks the Whisper model
web/index.html        the page
jobs/                 one folder per transcription (input, WAV, runs, results, log)
```

To change how Claude corrects, edit `SYSTEM_PROMPT` in `pipeline/correct.py`.
To change the Whisper prompt, edit `RELIGIOUS_PROMPT` in `pipeline/whisper_runs.py`.
