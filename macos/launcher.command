#!/bin/bash
cd "$(dirname "$0")/.."
DEST="$HOME/Creative Automation Pipeline"
mkdir -p "$DEST"
rsync -a --exclude '.venv' --exclude 'outputs' --exclude 'downloads' --exclude '__pycache__' \
      --exclude '.git' --exclude '_to_delete' --exclude '_template_backups' --exclude '.env' \
      --exclude 'default_templates/*/' \
      . "$DEST/"
cd "$DEST"
./install.sh && ./run.sh
