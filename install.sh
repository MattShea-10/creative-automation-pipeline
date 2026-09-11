#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok \033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror\033[0m %s\n' "$*" >&2; exit 1; }

say "Looking for Python 3.10 or newer"
PY=""
PYTHONORG_PATH="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"

if [ "$(uname)" = "Darwin" ] && [ -x "$PYTHONORG_PATH" ]; then
  if "$PYTHONORG_PATH" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$PYTHONORG_PATH"
  fi
fi

if [ -z "$PY" ]; then
  for candidate in python3.12 python3.11 python3.10 python3.13 python3.14 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        PY="$(command -v "$candidate")"
        break
      fi
    fi
  done
fi

[ -n "$PY" ] || die "No Python 3.10+ found. Install one from https://www.python.org/downloads/ and re-run."
ok "$("$PY" --version) at $PY"

VPY=".venv/bin/python"
if [ ! -x "$VPY" ] || ! "$VPY" -m pip --version >/dev/null 2>&1; then
  say "Creating virtual environment in .venv/"
  rm -rf .venv
  "$PY" -m venv .venv || die "Failed to create .venv"
else
  say "Using the existing .venv/"
fi
ok "$($VPY --version) in .venv/, pip: $($VPY -m pip --version)"

say "Installing dependencies (this can take a few minutes the first time)"
"$VPY" -m pip install --upgrade pip wheel >/dev/null
"$VPY" -m pip install -r requirements.txt
ok "requirements installed"

"$VPY" - <<'PYCHECK'
import importlib, sys
missing = []
for mod, pkg in (("aggdraw", "aggdraw"), ("skimage", "scikit-image"), ("psd_tools", "psd-tools"), ("PIL", "Pillow"), ("flask", "Flask")):
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f"{pkg} ({exc.__class__.__name__}: {exc})")
if missing:
    print("These didn't import after install:\n  " + "\n  ".join(missing), file=sys.stderr)
    sys.exit(1)
PYCHECK
ok "aggdraw, scikit-image, psd-tools, Pillow and Flask all import"

if [ ! -f .env ]; then
  cp .env.example .env
  ok "created .env from .env.example"
else
  ok ".env already present, left as is"
fi

if command -v tesseract >/dev/null 2>&1; then
  ok "tesseract found ($(tesseract --version 2>&1 | head -1))"
else
  warn "tesseract not found -- optional. 'brew install tesseract' adds it."
fi

mkdir -p outputs downloads assets/generated_cache
chmod +x run.sh 2>/dev/null || true

echo
say "Done. Starting the app..."
echo

exec ./run.sh
