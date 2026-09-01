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

import os
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
#
# 70 measured, on 8 real backdrops from outputs/: 2 of 8 clean images
# flagged something (a 25% false-alarm rate), against 6 of 8 and 7 of 8
# caught when text was painted onto them at 16% and 9% of frame height.
# Raise it to trade recall for quiet; AI_TEXT_MIN_CONFIDENCE overrides.
MIN_CONFIDENCE = float(os.environ.get("AI_TEXT_MIN_CONFIDENCE") or 70.0)

# Tesseract's page-segmentation modes. The default (3, "fully automatic
# page segmentation") assumes a scanned document and is worthless here:
# benchmarked against real generated backdrops with text painted onto
# them, it found nothing at all -- 0 out of 8 at every size tried. 6
# ("assume a single uniform block") and 11 ("sparse text, find as much as
# possible") are the ones that see lettering sitting in a photograph.
# Every mode is run and the findings pooled, because they disagree
# constantly and a word only has to be caught once.
PAGE_SEGMENTATION_MODES = (3, 6, 11)
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
    pages = []
    for mode in PAGE_SEGMENTATION_MODES:
        try:
            pages.append(
                pytesseract.image_to_data(
                    prepared,
                    config=f"--psm {mode}",
                    output_type=pytesseract.Output.DICT,
                )
            )
        except Exception:  # noqa: BLE001
            # A broken or half-installed Tesseract reports as "can't
            # check" rather than failing the render around it.
            continue
    if not pages:
        return TextCheckResult(available=False)

    min_height = max(1.0, image.height * MIN_HEIGHT_FRACTION)
    findings = []
    seen_boxes = set()
    for data in pages:
        findings.extend(_findings_from(data, min_height, seen_boxes))
    findings.sort(key=lambda f: -f.confidence)
    return TextCheckResult(available=True, findings=findings)


def _findings_from(data, min_height, seen_boxes):
    """The words in one OCR pass that clear every filter.

    `seen_boxes` is shared across passes: the modes overlap heavily and
    the same word found three times must not be counted three times --
    it would inflate the covered area and trip the "too much to paint
    out" limit on a single word.
    """
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
        box = (
            int(data["left"][i]),
            int(data["top"][i]),
            int(data["width"][i]),
            height,
        )
        # Rounded, so near-identical boxes from different modes collapse
        # onto each other rather than counting twice.
        key = tuple(v // 8 for v in box)
        if key in seen_boxes:
            continue
        seen_boxes.add(key)
        findings.append(TextFinding(text=word, confidence=confidence, box=box))
    return findings


# How far the removal mask is grown beyond each word's reported box, as a
# fraction of that word's height. OCR boxes hug the glyphs; antialiasing,
# drop shadows and glow spill past them, and leaving that halo behind
# reads worse than the text did -- a ghost of the word in exactly its
# shape. Generous, because inpainting a few extra pixels of background is
# cheap and missing the halo is not.
MASK_PADDING_FRACTION = 0.35

# Below this, inpainting is the wrong tool. Removing a word means
# inventing what was behind it, which only looks right when the
# surroundings are small and similar; asked to replace a banner across
# half the frame it produces a smear that is more distracting than the
# lettering was. Expressed as a fraction of total image area.
MAX_REMOVABLE_AREA_FRACTION = 0.06


def build_text_mask(image: Image.Image, result: "TextCheckResult"):
    """A white-on-black mask covering the words in `result`, padded.

    Returns None when there is nothing to cover.
    """
    if not result.findings:
        return None
    import numpy as np

    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    for finding in result.findings:
        left, top, width, height = finding.box
        pad = max(2, int(round(height * MASK_PADDING_FRACTION)))
        x0 = max(0, left - pad)
        y0 = max(0, top - pad)
        x1 = min(image.width, left + width + pad)
        y1 = min(image.height, top + height + pad)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return mask


def masked_area_fraction(image: Image.Image, mask) -> float:
    """How much of the frame the mask covers, 0..1."""
    if mask is None:
        return 0.0
    return float((mask > 0).sum()) / float(image.width * image.height)


def remove_text(image: Image.Image, result: "TextCheckResult"):
    """Paint out the words in `result`, reconstructing the background.

    Returns (cleaned_image, removed_count, reason). `reason` is None on
    success and a short explanation when nothing was done -- refusing
    loudly matters more than trying: a failed inpaint doesn't leave the
    image as it was, it leaves a smear where the text used to be.
    """
    if not result.findings:
        return image, 0, "nothing to remove"
    try:
        import cv2
        import numpy as np
    except ImportError:
        return image, 0, "OpenCV isn't installed"

    mask = build_text_mask(image, result)
    fraction = masked_area_fraction(image, mask)
    if fraction > MAX_REMOVABLE_AREA_FRACTION:
        return (
            image,
            0,
            f"the text covers {fraction:.0%} of the frame, too much to paint out "
            f"convincingly (limit {MAX_REMOVABLE_AREA_FRACTION:.0%})",
        )

    rgb = image.convert("RGB")
    array = np.array(rgb)[:, :, ::-1].copy()  # PIL RGB -> OpenCV BGR
    # Telea over Navier-Stokes: markedly faster at this size, and the
    # difference in quality is invisible on the small, isolated regions
    # this is limited to.
    radius = max(3, int(round(image.height * 0.006)))
    painted = cv2.inpaint(array, mask, radius, cv2.INPAINT_TELEA)
    cleaned = Image.fromarray(painted[:, :, ::-1])
    return cleaned, len(result.findings), None
