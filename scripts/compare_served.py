"""Do two running copies of this app serve the same page?

Used by the release workflow to compare what the packaged .exe serves
against what the same commit serves straight from source, on the same
runner and the same Python -- so the only variable left is PyInstaller.

The question it answers is "does the .exe look like the .dmg", and the
reason it needs answering automatically is that the old smoke test only
checked the exe returned HTTP 200. A bundle that lost templates/ or
static/ still returns 200; it just returns a different, unstyled page,
and the only way anyone found out was by opening it and looking.

    python scripts/compare_served.py exe.html source.html

Exits 0 when the two match once the parts that are *meant* to differ are
normalised away, and 1 with an excerpt of the first difference otherwise.
"""

from __future__ import annotations

import difflib
import re
import sys

# Things that differ between two runs of identical code, and say nothing
# about whether the build is sound:
_NOISE = (
    # The build stamp is the newest source file's mtime. A checkout and a
    # PyInstaller bundle date their files differently by nature.
    (re.compile(r"code build:\s*[\d\-: ]+"), "code build: NORMALISED"),
    # session_id and the progress token are fresh hex per request.
    (re.compile(r"\b[0-9a-f]{32}\b"), "HEX32"),
    (re.compile(r"\b[0-9a-f]{24}\b"), "HEX24"),
    # Whichever port each copy landed on.
    (re.compile(r"127\.0\.0\.1:\d+"), "127.0.0.1:PORT"),
)


def normalise(html: str) -> list[str]:
    """The page as it should be identical between the two, line by line."""
    for pattern, replacement in _NOISE:
        html = pattern.sub(replacement, html)
    # Trailing whitespace and line endings: one copy is served from a
    # Windows checkout, the other from a bundle.
    return [line.rstrip() for line in html.replace("\r\n", "\n").split("\n")]


def compare(packaged: str, source: str, packaged_name: str, source_name: str) -> str:
    """"" when the two agree, else a readable excerpt of the difference."""
    left, right = normalise(packaged), normalise(source)
    if left == right:
        return ""
    diff = list(difflib.unified_diff(right, left, source_name, packaged_name, lineterm="", n=2))
    return "\n".join(diff[:60])


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    packaged_name, source_name = argv[1], argv[2]
    with open(packaged_name, encoding="utf-8", errors="replace") as handle:
        packaged = handle.read()
    with open(source_name, encoding="utf-8", errors="replace") as handle:
        source = handle.read()
    difference = compare(packaged, source, packaged_name, source_name)
    if difference:
        print("The packaged app serves a different page than the source does.\n")
        print(difference)
        print(
            "\nUsually this means a file did not make it into the bundle -- check the\n"
            "datas list in windows/CreativeAutomationPipeline.spec."
        )
        return 1
    print("the packaged app serves the same page as the source")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
