# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the packaged (double-click) build of the web app.
#
#   pyinstaller --noconfirm windows/CreativeAutomationPipeline.spec
#
# Output: dist/CreativeAutomationPipeline/ -- the exe plus an _internal/
# folder holding Python, the libraries, and the read-only files the app
# ships with (Flask templates and the bundled fonts). windows/build_exe.ps1
# then copies the user-editable folders (default_templates, assets/brand,
# briefs, .env.example) beside the exe and zips the lot; that is the
# layout webapp.py expects when frozen (BASE_DIR = the exe's folder).
#
# One-folder rather than one-file on purpose: one-file unpacks ~300 MB of
# scikit-image/OpenCV into a temp dir on every launch (slow, and the
# thing antivirus heuristics dislike most), and the user's files need a
# stable folder to live in anyway.
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent

datas = [
    (str(ROOT / "templates"), "templates"),
    (str(ROOT / "fonts"), "fonts"),
]
# Wordlist shipped inside the package, not importable code.
datas += collect_data_files("better_profanity")

hiddenimports = (
    collect_submodules("src")
    + collect_submodules("skimage")
    + ["PIL._tkinter_finder", "psd_tools", "aggdraw", "cv2", "pytesseract", "deep_translator", "dotenv"]
)

a = Analysis(
    [str(ROOT / "webapp.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "unittest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CreativeAutomationPipeline",
    debug=False,
    strip=False,
    upx=False,
    # A console window on purpose: it is where the URL and any error is
    # printed, and closing it is how the server is stopped.
    console=True,
    icon=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="CreativeAutomationPipeline",
)
