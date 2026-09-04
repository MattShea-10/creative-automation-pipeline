#!/usr/bin/env bash
# Start the web app from the install made by ./install.sh, and open it.
#   ./run.sh              on http://127.0.0.1:5000
#   PORT=8080 ./run.sh    on another port
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "No .venv/ yet -- run ./install.sh first." >&2; exit 1; }
# 5000 unless taken. On macOS Monterey and later, AirPlay Receiver sits
# on 5000 and answers with a 403 that looks like the app failing -- so
# a busy port moves to the next free one and says so, rather than the
# browser opening on something that isn't this app.
PORT="${PORT:-5000}"
port_busy() {
  .venv/bin/python - "$1" <<'PYPORT'
import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(0)   # busy
finally:
    s.close()
sys.exit(1)       # free
PYPORT
}
if [ -z "${PORT_PINNED:-}" ]; then
  for candidate in "$PORT" 5050 8080 8000 8765; do
    if ! port_busy "$candidate"; then
      [ "$candidate" != "$PORT" ] && echo "Port $PORT is in use (on a Mac that is usually AirPlay Receiver) -- using $candidate instead."
      PORT="$candidate"; break
    fi
  done
fi
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
