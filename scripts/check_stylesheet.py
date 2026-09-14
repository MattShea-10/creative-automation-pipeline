"""Is the committed stylesheet the one styles/*.css compiles to?

Both the .exe and the .dmg serve static/tailwind-*.css as committed, so
if that file is behind its sources the two builds are styled from
different bytes -- a difference invisible in a screenshot and invisible
in a diff of the source. The release workflow recompiles on its runner
and calls this to compare.

    npm run css && python scripts/check_stylesheet.py

Line endings are ignored. Git on Windows checks text out as CRLF while
the Tailwind CLI writes LF, so a byte comparison failed on every line of
an otherwise identical file -- which is exactly the kind of false alarm
that teaches people to ignore a build check.
"""

from __future__ import annotations

import difflib
import subprocess
import sys

BUILT = ("static/tailwind-index.css", "static/tailwind-result.css", "static/styles.sha256")


def committed(path: str) -> bytes | None:
    """The file as HEAD has it, or None if HEAD doesn't have it."""
    result = subprocess.run(
        ["git", "show", f"HEAD:{path}"], capture_output=True, check=False
    )
    return result.stdout if result.returncode == 0 else None


def lines(data: bytes) -> list[str]:
    return data.replace(b"\r\n", b"\n").decode("utf-8", "replace").split("\n")


def _window(committed_line: str, rebuilt_line: str, span: int = 90) -> str:
    """The two lines around the first character that differs."""
    at = next(
        (i for i, (x, y) in enumerate(zip(committed_line, rebuilt_line)) if x != y),
        min(len(committed_line), len(rebuilt_line)),
    )
    start, end = max(0, at - span), at + span
    lead = "..." if start else ""
    return (
        f"  committed: {lead}{committed_line[start:end]}...\n"
        f"  rebuilt:   {lead}{rebuilt_line[start:end]}...\n"
        f"  (character {at} of that line)"
    )


def stale(path: str, on_disk: bytes) -> str:
    """"" when the working copy matches HEAD, else a short diff."""
    head = committed(path)
    if head is None:
        return f"{path} is not committed yet"
    before, after = lines(head), lines(on_disk)
    if before == after:
        return ""
    # Minified CSS is one enormous line, so neither a line diff nor the
    # first 400 characters says anything -- the change is usually 20,000
    # characters in. Show a window around the point where they part.
    for i, (a, b) in enumerate(zip(before, after), start=1):
        if a != b:
            return f"{path} differs at line {i}:\n{_window(a, b)}"
    return f"{path} differs in length: committed {len(before)} lines, rebuilt {len(after)}"


def main() -> int:
    problems = []
    for path in BUILT:
        try:
            with open(path, "rb") as handle:
                on_disk = handle.read()
        except OSError as exc:
            problems.append(f"{path}: {exc}")
            continue
        problem = stale(path, on_disk)
        if problem:
            problems.append(problem)
    if problems:
        print("\n\n".join(problems))
        print(
            "\nThe committed stylesheet is out of step with styles/*.css. Both the .exe\n"
            "and the .dmg serve the committed copy, so this means the two builds would\n"
            "not look alike. Run `npm run css` and commit static/."
        )
        return 1
    print("the committed stylesheet is what styles/*.css compiles to")
    return 0


if __name__ == "__main__":
    sys.exit(main())
