#!/usr/bin/env bash
# Keep static/tailwind-*.css in step with styles/*.css, without anyone
# having to remember `npm run css`. Sourced by run.sh and macos/make_dmg.sh
# from the project root; call ensure_css after that.
#
# static/styles.sha256 is written by `npm run css` (scripts/stamp_css.js)
# and is in the exact format shasum/sha256sum -c expects, so checking it is
# the one-liner below. Hashes rather than timestamps because the Tailwind
# CLI leaves an output file untouched when a rebuild changes nothing --
# a date comparison would accuse the source forever after a no-op edit.
#
# Nothing here ever stops the caller: the worst case prints a warning and
# carries on with the stylesheet that is already built.

css_in_step() {
  [ -f static/styles.sha256 ] || return 0          # no stamp (older checkout)
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 --status -c static/styles.sha256 2>/dev/null
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum --status -c static/styles.sha256 2>/dev/null
  else
    return 0                                        # nothing to check with
  fi
}

ensure_css() {
  css_in_step && return 0
  printf '\033[1;36m==>\033[0m styles/ changed -- rebuilding the stylesheet\n'
  if ! command -v npm >/dev/null 2>&1; then
    printf '\033[1;33mwarn\033[0m Node is not installed, so the stylesheet cannot be rebuilt here.\n'
    printf '      Serving the one that is already built. Install Node, or push and\n'
    printf '      let the Windows workflow rebuild it.\n'
    return 0
  fi
  if [ ! -d node_modules ]; then
    printf '    (first time: installing the build tool with npm install)\n'
    npm install --no-audit --no-fund --silent >/dev/null 2>&1 || {
      printf '\033[1;33mwarn\033[0m npm install failed (offline?). Serving the stylesheet that is already built.\n'
      return 0
    }
  fi
  if npm run css --silent >/dev/null 2>&1; then
    printf '\033[1;32mok\033[0m   stylesheet rebuilt\n'
  else
    printf '\033[1;33mwarn\033[0m the rebuild failed -- run `npm run css` to see why.\n'
    printf '      Serving the stylesheet that is already built.\n'
  fi
  return 0
}
