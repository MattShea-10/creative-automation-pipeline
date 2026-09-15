"""Access to the shipped PSD templates for tests that need a real one.

They used to sit loose in default_templates/ and several tests still
looked for them there. They live in per-campaign subfolders now, and
those are gitignored, so a fresh clone has none of them and the tests
either skipped or raised FileNotFoundError. default_templates.zip is
committed and holds the originals, so extract from that instead: the
same source the app seeds a new campaign folder from.
"""
from __future__ import annotations

import atexit
import shutil
import tempfile
import zipfile
from pathlib import Path

ZIP = Path(__file__).resolve().parent.parent / "default_templates" / "default_templates.zip"

_extracted_to: Path | None = None


def template(name: str) -> Path | None:
    """A shipped template as a real file on disk, or None if unavailable.

    Extracted once per process into a temp dir that is cleaned up at
    exit. Callers that modify the file should copy it first.
    """
    global _extracted_to
    if not ZIP.is_file():
        return None
    if _extracted_to is None:
        _extracted_to = Path(tempfile.mkdtemp(prefix="shipped-templates-"))
        atexit.register(shutil.rmtree, _extracted_to, ignore_errors=True)
    out = _extracted_to / name
    if out.is_file():
        return out
    try:
        with zipfile.ZipFile(ZIP) as archive, archive.open(name) as src:
            with open(out, "wb") as dest:
                shutil.copyfileobj(src, dest)
    except KeyError:
        return None
    return out


def names() -> list:
    """Every shipped template in the zip, smallest name first, with
    Finder's shadow entries left out. Empty when there is no zip."""
    if not ZIP.is_file():
        return []
    try:
        with zipfile.ZipFile(ZIP) as archive:
            return sorted(
                n for n in archive.namelist()
                if n.lower().endswith(".psd")
                and not n.startswith("__MACOSX")
                and not Path(n).name.startswith("._")
            )
    except Exception:  # noqa: BLE001
        return []
