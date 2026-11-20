# PyInstaller recipe for the installed app: one folder with Python, the page, ffmpeg and whisper.cpp.
# Build (after build/whisper/ holds whisper-cli and its libraries):  pyinstaller packaging/fatwa-transcript.spec
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

root = Path(SPECPATH).parent
whisper_dir = root / "build" / "whisper"

datas = [(str(root / "web"), "web"), (str(root / "config.example.env"), ".")]
datas += collect_data_files("imageio_ffmpeg")  # the ffmpeg program for this OS
datas += collect_data_files("anthropic")
datas += collect_data_files("certifi")  # HTTPS certificates: the macOS app's Python has none of its own
# whisper-cli and its GPU/CPU libraries are copied as-is (not analysed), so the system's Vulkan
# loader and GPU drivers are used at run time instead of copies from the build machine.
if whisper_dir.exists():
    datas += [(str(p), "whisper") for p in sorted(whisper_dir.iterdir()) if p.is_file()]

a = Analysis(
    [str(root / "app.py")],
    pathex=[str(root)],
    datas=datas,
    hiddenimports=["server", "pipeline.job", "pipeline.engines", "pipeline.correct", "pipeline.compare",
                   "pipeline.outputs", "pipeline.audio", "pipeline.model_download", "pipeline.whisper_runs",
                   "pipeline.learning", "pipeline.training", "psutil", "anthropic", "certifi"],
    excludes=["pywhispercpp", "_pywhispercpp", "numpy", "tkinter", "matplotlib", "PIL"],
)
pyz = PYZ(a.pure)
windowed = sys.platform in ("win32", "darwin")  # no terminal window on Windows and macOS
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="FatwaTranscript", console=not windowed,
          upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="FatwaTranscript", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Fatwa Transcript.app",
        bundle_identifier="io.github.mohammadmarwan2005.fatwatranscript",
        info_plist={"CFBundleShortVersionString": "0.4.0", "LSMinimumSystemVersion": "12.0",
                    "NSHighResolutionCapable": True},
    )
