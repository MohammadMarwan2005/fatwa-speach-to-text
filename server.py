"""Local web server for the transcriber page. Listens on 127.0.0.1 only and runs one job at a time.

Start it with ./run.sh (or: .venv/bin/python server.py). The API key stays in config.env on this computer.
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from pipeline import config, model_download
from pipeline.job import create_job, queue_redo, saved_runs

MAX_UPLOAD = 1024 ** 3  # 1 GB
JOB_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
DOWNLOADS = {
    "transcript.txt": "text/plain; charset=utf-8",
    "transcript.srt": "application/x-subrip; charset=utf-8",
    "result.json": "application/json; charset=utf-8",
}


class Worker(threading.Thread):
    """Runs queued jobs one after another, each in its own process (Whisper uses all CPU cores)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.jobs = queue.Queue()
        self.current = None

    def run(self):
        while True:
            d = self.jobs.get()
            self.current = d.name
            with open(d / "log.txt", "ab") as log:
                subprocess.run([sys.executable, "-m", "pipeline.job", str(d)], cwd=config.ROOT,
                               stdout=log, stderr=subprocess.STDOUT)
            self.current = None


WORKER = Worker()


def read_state(d: Path):
    try:
        return json.loads((d / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def recover_jobs() -> None:
    """After a restart: jobs that were running are marked as stopped; queued ones run again."""
    config.JOBS_DIR.mkdir(exist_ok=True)
    queued = []
    for d in sorted(config.JOBS_DIR.iterdir()):
        st = read_state(d) if d.is_dir() else None
        if not st:
            continue
        if st["status"] == "running" and not _alive(st.get("pid")):
            st["status"], st["error"] = "error", "Stopped: the job ended before finishing (server or computer restarted)."
            for s in st["steps"]:
                if s["status"] == "running":
                    s["status"], s["detail"] = "error", "Stopped"
            (d / "state.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        elif st["status"] == "queued":
            queued.append(d)
    for d in queued:
        WORKER.jobs.put(d)


class Handler(BaseHTTPRequestHandler):
    server_version = "fatwa-transcript"

    # --- helpers -------------------------------------------------------------
    def _host_ok(self) -> bool:
        # Blocks DNS-rebinding pages from talking to this server.
        port = self.server.server_address[1]
        return self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _send(self, code: int, body: bytes, ctype: str, extra: dict = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data, code: int = 200) -> None:
        self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, code: int, message: str) -> None:
        self._json({"error": message}, code)

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def _job_dir(self, job_id: str):
        if not JOB_ID.match(job_id):
            return None
        d = config.JOBS_DIR / job_id
        return d if (d / "state.json").exists() else None

    # --- routes --------------------------------------------------------------
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._host_ok():
            return self._error(403, "Open the page at http://127.0.0.1")
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, (config.WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/status":
            return self._json(self._status())
        if path == "/api/jobs":
            return self._json(self._list_jobs())
        m = re.match(r"^/api/jobs/([^/]+)(?:/(files/[^/]+|audio))?$", path)
        if not m:
            return self._error(404, "Not found")
        d = self._job_dir(m.group(1))
        if d is None:
            return self._error(404, "No such job")
        if m.group(2) is None:
            return self._json(read_state(d))
        if m.group(2) == "audio":
            return self._audio(d / "plain.wav")
        name = unquote(m.group(2)[len("files/"):])
        if name not in DOWNLOADS or not (d / name).exists():
            return self._error(404, "File not ready")
        st = read_state(d)
        stem = Path(st["filename"]).stem
        dl = f"{stem}.{name.split('.', 1)[1]}" if name.startswith("transcript") else f"{stem}.result.json"
        disposition = "attachment" if parse_qs(urlparse(self.path).query).get("download") else "inline"
        return self._send(200, (d / name).read_bytes(), DOWNLOADS[name],
                          {"Content-Disposition": f"{disposition}; filename*=UTF-8''{_quote(dl)}"})

    def do_POST(self):
        if not self._host_ok():
            return self._error(403, "Open the page at http://127.0.0.1")
        # A custom header that other websites can't send without permission (blocks cross-site uploads).
        if self.headers.get("X-Transcriber") != "1":
            return self._error(403, "Missing X-Transcriber header")
        path = urlparse(self.path).path
        m = re.match(r"^/api/jobs/([^/]+)/redo$", path)
        if m:
            return self._redo(m.group(1))
        if path != "/api/jobs":
            return self._error(404, "Not found")
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._error(400, "No file received")
        if length > MAX_UPLOAD:
            return self._error(413, "File is larger than 1 GB")
        data = self.rfile.read(length)
        q = parse_qs(urlparse(self.path).query)
        opts = {"model": q.get("model", [None])[0], "runs": q.get("runs", [2])[0], "effort": q.get("effort", [None])[0]}
        d = create_job(q.get("filename", ["audio"])[0], data=data, options=opts)
        WORKER.jobs.put(d)
        return self._json(read_state(d), 201)

    # --- route bodies --------------------------------------------------------
    def _redo(self, job_id: str) -> None:
        """Run steps 4-6 again with the chosen model, reusing the job's Whisper runs."""
        d = self._job_dir(job_id)
        if d is None:
            return self._error(404, "No such job")
        st = read_state(d)
        if st["status"] in ("queued", "running"):
            return self._error(409, "This job is still running")
        if not saved_runs(d):
            return self._error(409, "This job has no Whisper runs to reuse")
        q = parse_qs(urlparse(self.path).query)
        model = q.get("model", [""])[0]
        if model not in config.CLAUDE_MODELS:
            return self._error(400, "Choose a Claude model")
        if not config.api_key():
            return self._error(400, "Add your API key to config.env first")
        st = queue_redo(d, model, q.get("effort", [None])[0])
        WORKER.jobs.put(d)
        return self._json(st)

    def _status(self) -> dict:
        mpath = config.whisper_model_path()
        return {
            "api_key_set": bool(config.api_key()),
            "config_file": str(config.CONFIG_FILE),
            "default_model": config.default_claude_model(),
            "default_effort": config.default_effort(),
            "models": [{"id": k, "label": v["label"], "in": v["in"], "out": v["out"]}
                       for k, v in config.CLAUDE_MODELS.items()],
            "whisper": {"ready": model_download.is_ready(mpath), "path": str(mpath),
                        "threads": config.whisper_threads()},
            "busy": WORKER.current,
            "queued": WORKER.jobs.qsize(),
        }

    def _list_jobs(self) -> list:
        out = []
        if config.JOBS_DIR.exists():
            for d in sorted(config.JOBS_DIR.iterdir(), reverse=True)[:100]:
                st = read_state(d) if d.is_dir() else None
                if st:
                    out.append({k: st.get(k) for k in ("id", "filename", "created", "status", "duration", "options", "warning")})
        return out

    def _audio(self, wav: Path) -> None:
        """Serve the converted WAV with Range support, so the page can jump to any line."""
        if not wav.exists():
            return self._error(404, "Audio not ready")
        size = wav.stat().st_size
        rng = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if not rng:
            return self._send(200, wav.read_bytes(), "audio/wav", {"Accept-Ranges": "bytes"})
        start = int(rng.group(1)) if rng.group(1) else max(0, size - int(rng.group(2) or 0))
        end = min(int(rng.group(2)), size - 1) if rng.group(1) and rng.group(2) else size - 1
        if start >= size or start > end:
            return self._send(416, b"", "audio/wav", {"Content-Range": f"bytes */{size}"})
        with open(wav, "rb") as f:
            f.seek(start)
            body = f.read(end - start + 1)
        self._send(206, body, "audio/wav", {"Accept-Ranges": "bytes", "Content-Range": f"bytes {start}-{end}/{size}"})


def _alive(pid) -> bool:
    """True if a job process with this id is still running (e.g. one started from the terminal)."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _quote(s: str) -> str:
    from urllib.parse import quote
    return quote(s, safe="")


def main() -> None:
    recover_jobs()
    WORKER.start()
    port = config.port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Transcriber running at {url}  (Ctrl+C to stop)")
    if not config.api_key():
        print(f"No API key yet: put ANTHROPIC_API_KEY=... in {config.CONFIG_FILE} (free mode works without it).")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
