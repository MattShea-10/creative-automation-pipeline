"""Detect legible text baked into a generated image.

Image models are notoriously bad at lettering, and a backdrop is the one
place it is never wanted -- the template's own header, description and CTA
sit on top, so anything the model invented underneath reads as a mistake.
The prompt already asks for "no text, no lettering, no logos", and models
ignore that often enough to be worth verifying rather than trusting.

This is a *verification* step, deliberately biased toward staying quiet:
a false alarm costs a needless regeneration (real money on a paid
provider) and teaches people to ignore the warning. See find_text() for
the specific filters that buy that quiet.

Needs Tesseract, which is a system binary rather than a Python package:

    macOS         brew install tesseract
    Debian/Ubuntu apt install tesseract-ocr

Without it every function here reports "unavailable" and callers skip the
check -- never a hard failure, since a missing OCR engine is a reason to
not verify, not a reason to refuse to render a campaign.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field

from PIL import Image

# A word has to clear all of these to count. Each one exists to suppress a
# specific kind of false positive seen on real photographic backdrops.
#
# MIN_CONFIDENCE: Tesseract reports per-word confidence, and hallucinated
# "words" found in foliage, gravel and cloth texture come back low. Real
# baked-in lettering -- even garbled lettering, which is the usual failure
# -- comes back high, because it genuinely looks like type.
MIN_CONFIDENCE = 70.0
# MIN_LETTERS: single characters and stray punctuation are almost always
# noise. Two letters is where "this is actually type" starts.
MIN_LETTERS = 2
# MIN_HEIGHT_FRACTION: a "word" a few pixels tall in a 2000px image is
# texture being over-read. Anything a viewer would actually see as text
# occupies a meaningful slice of the frame.
MIN_HEIGHT_FRACTION = 0.012

_LETTERS = re.compile(r"[A-Za-z]")


@dataclass
class TextFinding:
    """One word the OCR engine was confident about."""

    text: str
    confidence: float
    box: tuple  # (left, top, width, height), pixels


@dataclass
class TextCheckResult:
    available: bool
    findings: list = field(default_factory=list)

    @property
    def found_text(self) -> bool:
        return bool(self.findings)

    def summary(self, limit: int = 4) -> str:
        """The found words, quoted, for a warning message."""
        words = [f'"{finding.text}"' for finding in self.findings[:limit]]
        extra = len(self.findings) - len(words)
        if extra > 0:
            words.append(f"and {extra} more")
        return ", ".join(words)


def ocr_available() -> bool:
    """Whether both halves of the OCR stack are actually present."""
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return shutil.which("tesseract") is not None


def find_text(image: Image.Image) -> TextCheckResult:
    """Words the OCR engine is confident it can read in `image`.

    Returns an empty result (available=False) when Tesseract isn't
    installed, so a caller can tell "checked, clean" from "couldn't
    check" -- which matter differently and must never be conflated.
    """
    if not ocr_available():
        return TextCheckResult(available=False)

    import pytesseract

    # Greyscale: colour carries nothing for OCR and the conversion makes
    # the engine's own preprocessing more predictable.
    prepared = image.convert("L")
    try:
        data = pytesseract.image_to_data(
            prepared, output_type=pytesseract.Output.DICT
        )
    except Exception:  # noqa: BLE001
        # A broken or half-installed Tesseract reports as "can't check"
        # rather than failing the render around it.
        return TextCheckResult(available=False)

    min_height = max(1.0, image.height * MIN_HEIGHT_FRACTION)
    findings = []
    for i, raw in enumerate(data.get("text", [])):
        word = (raw or "").strip()
        if len(_LETTERS.findall(word)) < MIN_LETTERS:
            continue
        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if confidence < MIN_CONFIDENCE:
            continue
        height = int(data["height"][i])
        if height < min_height:
            continue
        findings.append(
            TextFinding(
                text=word,
                confidence=confidence,
                box=(
                    int(data["left"][i]),
                    int(data["top"][i]),
                    int(data["width"][i]),
                    height,
                ),
            )
        )
    return TextCheckResult(available=True, findings=findings)
