"""Entry point of the installed app (and of `python app.py` from source).

Double-clicking the app starts the local server and opens the page in the default browser.
The same program runs one transcription job when started with --run-job <folder> (used by the server,
because an installed app has no separate Python interpreter).
"""
import multiprocessing
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--run-job":
        from pipeline.job import run
        return run(Path(sys.argv[2]).resolve())
    if len(sys.argv) >= 2 and sys.argv[1] in ("--version", "--self-test"):
        return self_test()
    import server
    server.main()
    return 0


def self_test() -> int:
    """Check used by the build machines: modules import, files are bundled, whisper-cli is found, and
    HTTPS downloads work (the model download failed on macOS once, for lack of a certificate list)."""
    import urllib.request
    import server
    from pipeline import config, engines, model_download
    print(f"{config.APP_NAME} {server.APP_VERSION}")
    web = (config.WEB_DIR / "index.html").is_file()
    print("web page:", web)
    print("whisper-cli:", engines.cli_path())
    print("engine:", engines.describe_available())
    print("data folder:", config.DATA)
    https = False
    try:
        req = urllib.request.Request(config.MODEL_URL.format(file="ggml-tiny.bin"),
                                     headers={"Range": "bytes=0-1023", "User-Agent": "fatwa-transcript"})
        with urllib.request.urlopen(req, timeout=60, context=model_download._ssl_context()) as r:
            https = r.status in (200, 206) and len(r.read()) > 0
    except Exception as e:
        print("https error:", e)
    print("https download:", https)
    return 0 if web and https else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
