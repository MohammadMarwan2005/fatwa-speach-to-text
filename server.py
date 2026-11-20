"""Local web server for the transcriber page. Listens on 127.0.0.1 only and runs one job at a time.

From source: ./run.sh (or .venv/bin/python server.py). The installed app starts it through app.py.
The API key stays in config.env on this computer; the page only learns whether one is set.
"""
import json
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from pipeline import config, engines, model_download, outputs
from pipeline.job import create_job, queue_redo, saved_runs

APP_VERSION = "0.3.0"
MAX_UPLOAD = 2 * 1024 ** 3  # 2 GB
JOB_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
DOWNLOADS = {
    "transcript.txt": "text/plain; charset=utf-8",
    "transcript.srt": "application/x-subrip; charset=utf-8",
    "result.json": "application/json; charset=utf-8",
}
HTTPD = None


def job_command(d: Path) -> list:
    """The installed app has no separate Python: it runs jobs by starting itself in job mode."""
    if config.FROZEN:
        return [sys.executable, "--run-job", str(d)]
    return [sys.executable, "-m", "pipeline.job", str(d)]


class Worker(threading.Thread):
    """Runs queued jobs one after another, each in its own process (Whisper uses the whole CPU or GPU)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.jobs = queue.Queue()
        self.current = None

    def run(self):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        while True:
            d = self.jobs.get()
            self.current = d.name
            with open(d / "log.txt", "ab") as log:
                subprocess.run(job_command(d), cwd=config.RESOURCES, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=flags)
            self.current = None


WORKER = Worker()


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_state(d: Path):
    return _read_json(d / "state.json")


def _alive(pid) -> bool:
    """True if a job process with this id is still running (e.g. one started from the terminal).
    psutil works on every OS; os.kill(pid, 0) would terminate the process on Windows."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        return False


def recover_jobs() -> None:
    """After a restart: jobs that were running are marked as stopped; queued ones run again."""
    queued = []
    for d in sorted(config.JOBS_DIR.iterdir()):
        st = read_state(d) if d.is_dir() else None
        if not st:
            continue
        if st["status"] == "running" and not _alive(st.get("pid")):
            st["status"], st["error"] = "error", "Stopped: the job ended before finishing (app or computer restarted)."
            for s in st["steps"]:
                if s["status"] == "running":
                    s["status"], s["detail"] = "error", "Stopped"
            (d / "state.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        elif st["status"] == "queued":
            queued.append(d)
    for d in queued:
        WORKER.jobs.put(d)


def last_device() -> str:
    """Where Whisper ran last time (shown on the page), e.g. "GPU (Vulkan): NVIDIA GeForce GTX 1650"."""
    if config.JOBS_DIR.exists():
        for d in sorted(config.JOBS_DIR.iterdir(), reverse=True):
            st = read_state(d) if d.is_dir() else None
            if st and st.get("device"):
                return st["device"]
    return ""


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
        m = re.match(r"^/api/jobs/([^/]+)(?:/(files/[^/]+|audio|edits))?$", path)
        if not m:
            return self._error(404, "Not found")
        d = self._job_dir(m.group(1))
        if d is None:
            return self._error(404, "No such job")
        if m.group(2) is None:
            return self._json(read_state(d))
        if m.group(2) == "audio":
            return self._audio(d / "plain.wav")
        if m.group(2) == "edits":
            return self._json(_read_json(d / "edits.json") or {"lines": {}, "saved_at": None})
        name = unquote(m.group(2)[len("files/"):])
        if name not in DOWNLOADS or not (d / name).exists():
            return self._error(404, "File not ready")
        st = read_state(d)
        stem = Path(st["filename"]).stem
        dl = f"{stem}.{name.split('.', 1)[1]}" if name.startswith("transcript") else f"{stem}.result.json"
        disposition = "attachment" if parse_qs(urlparse(self.path).query).get("download") else "inline"
        return self._send(200, (d / name).read_bytes(), DOWNLOADS[name],
                          {"Content-Disposition": f"{disposition}; filename*=UTF-8''{quote(dl, safe='')}"})

    def do_POST(self):
        if not self._host_ok():
            return self._error(403, "Open the page at http://127.0.0.1")
        # A custom header that other websites can't send without permission (blocks cross-site requests).
        if self.headers.get("X-Transcriber") != "1":
            return self._error(403, "Missing X-Transcriber header")
        path = urlparse(self.path).path
        if path == "/api/quit":
            self._json({"ok": True})
            threading.Timer(0.3, lambda: HTTPD and HTTPD.shutdown()).start()
            return None
        m = re.match(r"^/api/jobs/([^/]+)/redo$", path)
        if m:
            return self._redo(m.group(1))
        m = re.match(r"^/api/jobs/([^/]+)/edits$", path)
        if m:
            return self._save_edits(m.group(1))
        if path != "/api/jobs":
            return self._error(404, "Not found")
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._error(400, "No file received")
        if length > MAX_UPLOAD:
            return self._error(413, "File is larger than 2 GB")
        data = self.rfile.read(length)
        q = parse_qs(urlparse(self.path).query)
        opts = {k: q.get(k, [None])[0] for k in ("model", "runs", "effort", "whisper")}
        d = create_job(q.get("filename", ["audio"])[0], data=data, options=opts)
        WORKER.jobs.put(d)
        return self._json(read_state(d), 201)

    # --- route bodies --------------------------------------------------------
    def _save_edits(self, job_id: str) -> None:
        """Store the person's corrected lines, rewrite transcript.txt/.srt with them,
        and return the clean text for the "Save .txt" file."""
        d = self._job_dir(job_id)
        if d is None:
            return self._error(404, "No such job")
        payload = _read_json(d / "result.json")
        if payload is None:
            return self._error(409, "This job has no transcript yet")
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= 5 * 1024 * 1024:
            return self._error(400, "Nothing to save")
        try:
            sent = json.loads(self.rfile.read(length).decode("utf-8"))["lines"]
            assert isinstance(sent, dict)
        except (ValueError, KeyError, AssertionError, UnicodeDecodeError):
            return self._error(400, "Expected {\"lines\": {\"<line id>\": \"text\"}}")
        result = payload["result"]
        original = {str(ln["id"]): ln["text"] for ln in result["lines"]}
        edits = {}
        for line_id, text in sent.items():
            if line_id in original and isinstance(text, str):
                text = " ".join(text.split())[:20000]
                if text != " ".join(original[line_id].split()):
                    edits[line_id] = text
        (d / "edits.json").write_text(json.dumps({"lines": edits, "saved_at": time.time()},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
        edited = outputs.apply_edits(result, edits)
        st = read_state(d)
        how = payload.get("how") or "Whisper large-v3."
        if edits:
            how += f" {len(edits)} line{'s' if len(edits) != 1 else ''} corrected by hand."
        outputs.write_txt(edited, Path(st["filename"]).stem, st.get("duration", 0), how, d / "transcript.txt")
        outputs.write_srt(edited, d / "transcript.srt")
        return self._json({"ok": True, "edited": len(edits), "text": outputs.plain_text(edited)})

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
        wname = config.whisper_model_name()
        return {
            "version": APP_VERSION,
            "installed": config.FROZEN,
            "api_key_set": bool(config.api_key()),
            "config_file": str(config.CONFIG_FILE),
            "default_model": config.default_claude_model(),
            "default_effort": config.default_effort(),
            "models": [{"id": k, "label": v["label"], "in": v["in"], "out": v["out"]}
                       for k, v in config.CLAUDE_MODELS.items()],
            "whisper": {
                "default": wname,
                "ready": model_download.is_ready(config.whisper_model_path(wname)),
                "models": [{"id": k, "label": v["label"], "size": v["size"],
                            "ready": model_download.is_ready(config.whisper_model_path(k))}
                           for k, v in config.WHISPER_MODELS.items()],
                "engine": engines.describe_available(),
                "last_device": last_device(),
                "threads": config.whisper_threads(),
            },
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


def _ours_running(port: int) -> bool:
    """Is this app already running on the port? Then we just open the page instead of starting twice."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=1.5) as r:
            return "api_key_set" in json.loads(r.read().decode("utf-8"))
    except (OSError, ValueError):
        return False


def _free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def main() -> None:
    global HTTPD
    config.ensure_data_dir()
    open_browser = "--no-browser" not in sys.argv
    port = config.port()
    for candidate in range(port, port + 20):
        if _free(candidate):
            port = candidate
            break
        if _ours_running(candidate):
            print(f"Already running at http://127.0.0.1:{candidate}")
            if open_browser:
                webbrowser.open(f"http://127.0.0.1:{candidate}")
            return
    recover_jobs()
    WORKER.start()
    HTTPD = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"{config.APP_NAME} {APP_VERSION} running at {url}  (Ctrl+C or the Quit button to stop)")
    print(f"Data folder: {config.DATA}")
    if not config.api_key():
        print(f"No API key: free mode. To add one, put ANTHROPIC_API_KEY=... in {config.CONFIG_FILE}")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        HTTPD.serve_forever()
    except KeyboardInterrupt:
        pass
    print("Stopped.")


if __name__ == "__main__":
    main()
