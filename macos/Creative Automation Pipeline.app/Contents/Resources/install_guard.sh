#!/bin/bash
# Which of two copies of the project is the newer code.
#
# The launcher copies the disk image's project over the installed one on
# every launch, so that a newer dmg updates an older install. Nothing
# stopped that running backwards: opening a two-day-old image quietly
# reverted a working folder to two-day-old source, and the only symptom
# was the app behaving as though a week of fixes had been undone.
#
# Sourced by the launcher and by tests/test_mac_launcher.py, so the
# comparison can be exercised without a disk image or a Terminal window.

# Newest modification time (epoch seconds) among a project's source
# files. Only the kinds of file the image ships: .venv, outputs and the
# rest belong to whoever is using the app and say nothing about which
# code is newer. Empty when there is no source there to date.
newest_source_mtime() {
  local root="$1"
  [ -d "$root" ] || return 0
  # BSD stat on macOS, GNU stat everywhere else (the tests run on both).
  local fmt
  if stat -f %m "$root" >/dev/null 2>&1; then fmt=(-f %m); else fmt=(-c %Y); fi
  find "$root" -type f \
    \( -name '*.py' -o -name '*.html' -o -name '*.css' -o -name '*.js' \
       -o -name '*.sh' -o -name '*.spec' -o -name '*.json' -o -name '*.yaml' \) \
    ! -path '*/.venv/*' ! -path '*/.git/*' ! -path '*/outputs/*' \
    ! -path '*/downloads/*' ! -path '*/node_modules/*' ! -path '*/__pycache__/*' \
    ! -path '*/_template_backups/*' ! -path '*/_to_delete/*' \
    -exec stat "${fmt[@]}" {} + 2>/dev/null | sort -n | tail -1
}

# True when copying SOURCE over DEST would take the installed code
# backwards.
#
# Anything undecidable answers false: no install yet, or neither side
# has datable source. A guard that blocks a first install would be worse
# than the problem it prevents.
image_is_older_than_install() {
  local source="$1" dest="$2"
  [ -d "$dest" ] || return 1
  local newest_image newest_install
  newest_image="$(newest_source_mtime "$source")"
  newest_install="$(newest_source_mtime "$dest")"
  [ -n "$newest_image" ] && [ -n "$newest_install" ] || return 1
  [ "$newest_image" -lt "$newest_install" ]
}
