#!/usr/bin/env bash
# One-shot local install for macOS and Linux.
#
#   ./install.sh          set up (safe to re-run; it only fills in what's missing)
#   ./run.sh              start the web app and open it in your browser
#
# What it does, in order: finds a Python 3.10+, makes a private virtual
# environment in .venv/, installs requirements.txt into it, creates .env
# from .env.example if you don't have one yet, and checks for the optional
# `tesseract` binary. Nothing is installed outside this folder except
# what pip caches, and nothing needs sudo.
set -euo pipefail

cd "$(dirname "$0")"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok \033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 1. A Python new enough (3.10+; the code uses match-free but
#         3.10-era typing, and psd-tools 1.10+ needs it).
say "Looking for Python 3.10 or newer"
PY=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      PY="$(command -v "$candidate")"
      break
    fi
  fi
done
[ -n "$PY" ] || die "No Python 3.10+ found. Install one from https://www.python.org/downloads/ (or 'brew install python@3.12' on a Mac) and re-run."
ok "$("$PY" --version) at $PY"

# ---- 2. A virtual environment of its own, so nothing here touches the
#         system Python or any other project.
if [ ! -x .venv/bin/python ]; then
  say "Creating virtual environment in .venv/"
  "$PY" -m venv .venv
else
  say "Using the existing .venv/"
fi
VPY=".venv/bin/python"
ok "$($VPY --version) in .venv/"

# ---- 3. Dependencies.
say "Installing dependencies (this can take a few minutes the first time)"
"$VPY" -m pip install --upgrade pip wheel >/dev/null
"$VPY" -m pip install -r requirements.txt
ok "requirements installed"

# The two that matter most for PSD work fail loudly here rather than
# silently at render time: without aggdraw every per-layer operation on
# a template with a vector shape returns nothing, and without
# scikit-image the saved-template preview refresh can't run.
"$VPY" - <<'PYCHECK'
import importlib, sys
missing = []
for mod, pkg in (("aggdraw", "aggdraw"), ("skimage", "scikit-image"), ("psd_tools", "psd-tools"), ("PIL", "Pillow"), ("flask", "Flask")):
    try:
        importlib.import_module(mod)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{pkg} ({exc.__class__.__name__}: {exc})")
if missing:
    print("These didn't import after install:\n  " + "\n  ".join(missing), file=sys.stderr)
    sys.exit(1)
PYCHECK
ok "aggdraw, scikit-image, psd-tools, Pillow and Flask all import"

# ---- 4. Secrets file. Never overwritten: a .env you've already filled
#         in is the one thing here that can't be regenerated.
if [ ! -f .env ]; then
  cp .env.example .env
  ok "created .env from .env.example -- add IDEOGRAM_API_KEY there to use Ideogram (Pollinations works without a key)"
else
  ok ".env already present, left as is"
fi

# ---- 5. Optional: the OCR binary behind the text-in-image compliance
#         check. The app runs without it; that one check just no-ops.
if command -v tesseract >/dev/null 2>&1; then
  ok "tesseract found ($(tesseract --version 2>&1 | head -1))"
else
  warn "tesseract not found -- optional. The text-in-image compliance check is skipped without it. 'brew install tesseract' (Mac) or 'apt install tesseract-ocr' (Debian/Ubuntu) adds it."
fi

# ---- 6. Folders the app writes to, so first run doesn't have to.
mkdir -p outputs downloads assets/generated_cache

chmod +x run.sh 2>/dev/null || true
echo
say "Done. Start it with:"
echo
echo "    ./run.sh"
echo
echo "Then open http://127.0.0.1:5000 (run.sh opens it for you). Ctrl-C stops it."
echo "Command-line pipeline: .venv/bin/python -m src.main --brief briefs/sample_campaign.yaml"
