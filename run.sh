#!/usr/bin/env bash
# Start the web app from the install made by ./install.sh, and open it.
#   ./run.sh              on http://127.0.0.1:5000
#   PORT=8080 ./run.sh    on another port
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "No .venv/ yet -- run ./install.sh first." >&2; exit 1; }
PORT="${PORT:-5000}"
URL="http://127.0.0.1:$PORT"
# Open the browser once the server is answering, from the background, so
# the server itself stays in the foreground where Ctrl-C reaches it.
(
  for _ in $(seq 1 40); do
    sleep 0.5
    if curl -fs "$URL" >/dev/null 2>&1; then
      if command -v open >/dev/null 2>&1; then open "$URL"
      elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
      fi
      exit 0
    fi
  done
) &
echo "Creative Automation Pipeline -> $URL   (Ctrl-C to stop)"
PORT="$PORT" exec .venv/bin/python webapp.py
