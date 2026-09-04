#!/usr/bin/env bash
# Build CreativeAutomationPipeline.dmg -- run this ON A MAC, from the
# project folder:
#
#     ./macos/make_dmg.sh
#
# The image holds a clean copy of the project (what git tracks, plus the
# templates in default_templates/ even if uncommitted, minus .env and
# anything generated) and a double-clickable "Creative Automation
# Pipeline.app" that copies the project into the user's home folder on
# first launch, runs install.sh, and starts the web app in a Terminal
# window. Needs hdiutil, which every Mac has; nothing to install.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v hdiutil >/dev/null || { echo "hdiutil not found -- this has to run on macOS." >&2; exit 1; }

NAME="CreativeAutomationPipeline"
VOL="Creative Automation Pipeline"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "==> Staging a clean copy of the project"
mkdir -p "$STAGE/creative-automation-pipeline"
# What git tracks, without the repository itself...
git archive HEAD | tar -x -C "$STAGE/creative-automation-pipeline"
# ...plus the templates as they are on disk right now, since the ones
# in use are often edited or added without being committed yet.
if [ -d default_templates ]; then
  mkdir -p "$STAGE/creative-automation-pipeline/default_templates"
  cp default_templates/*.psd "$STAGE/creative-automation-pipeline/default_templates/" 2>/dev/null || true
fi
# Never the secrets file. The launcher's install step makes a blank one.
rm -f "$STAGE/creative-automation-pipeline/.env"
chmod +x "$STAGE/creative-automation-pipeline/install.sh" "$STAGE/creative-automation-pipeline/run.sh"

echo "==> Adding the launcher"
cp -R "macos/Creative Automation Pipeline.app" "$STAGE/"
chmod +x "$STAGE/Creative Automation Pipeline.app/Contents/MacOS/start"

cat > "$STAGE/READ ME FIRST.txt" <<'TXT'
Creative Automation Pipeline

1. Double-click "Creative Automation Pipeline.app".
   The first time, macOS may say it can't verify the developer:
   right-click (or Control-click) the app and choose Open, then Open again.

2. A Terminal window opens and installs everything into a folder called
   "Creative Automation Pipeline" in your home folder. This takes a few
   minutes the first time and needs Python 3.10 or newer (the installer
   says so if it can't find one -- get it from python.org).

3. The app opens in your browser when it's ready. Press Ctrl-C in the
   Terminal window to stop it. Double-click the app again to restart.

To use Ideogram, open ~/Creative Automation Pipeline/.env in a text
editor, paste the API key after IDEOGRAM_API_KEY=, save, and restart.
The default Pollinations provider works with no key.
TXT

echo "==> Building $NAME.dmg"
rm -f "$NAME.dmg"
hdiutil create -volname "$VOL" -srcfolder "$STAGE" -ov -format UDZO "$NAME.dmg" >/dev/null
echo "==> Done: $(pwd)/$NAME.dmg ($(du -h "$NAME.dmg" | cut -f1))"
