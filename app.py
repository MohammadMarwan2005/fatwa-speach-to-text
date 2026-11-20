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
    """Quick check used by the build machines: modules import, files are bundled, whisper-cli is found."""
    import server
    from pipeline import config, engines
    print(f"{config.APP_NAME} {server.APP_VERSION}")
    print("web page:", (config.WEB_DIR / "index.html").is_file())
    print("whisper-cli:", engines.cli_path())
    print("engine:", engines.describe_available())
    print("data folder:", config.DATA)
    return 0 if (config.WEB_DIR / "index.html").is_file() else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
