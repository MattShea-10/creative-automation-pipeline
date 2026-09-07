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


def tesseract_available() -> bool:
    """Whether both halves of the Tesseract stack are actually present."""
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return shutil.which("tesseract") is not None


# The scene-text detector. Tesseract is a document reader: it reads a
# page of type and is close to blind to a headline set over a photograph
# -- a run with "a drink," lettered across the sky and a wordmark down
# the bottle came back from it as "no text found", and every check
# built on it was a check that never fired. RapidOCR bundles PaddleOCR's
# DB detector and recogniser as ONNX models inside the pip package (no
# separate model download, about 15 MB), finds that headline, the
# wordmark and the label, and runs in a couple of seconds on a CPU.
# Installed into the running interpreter on first use, the way ffmpeg
# is, so a packaged or fresh install doesn't need a setup step.
DETECTOR_PACKAGE = "rapidocr-onnxruntime"
DETECTOR_MIN_SCORE = float(os.environ.get("AI_TEXT_DETECTOR_MIN_SCORE") or 0.5)
_detector = None
_detector_state = None  # None = untried, "ready", or an error message


def ensure_text_detector(install: bool = True):
    """The RapidOCR engine, importing it and installing it first if
    allowed; None (with the reason in `_detector_state`) when it can't
    be had."""
    global _detector, _detector_state
    if _detector is not None:
        return _detector
    if _detector_state not in (None, "ready") and not install:
        return None
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        if not install or os.environ.get("AI_TEXT_DETECTOR_NO_INSTALL"):
            _detector_state = f"{DETECTOR_PACKAGE} isn't installed"
            return None
        import subprocess
        import sys

        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", DETECTOR_PACKAGE],
                check=True, capture_output=True, timeout=600,
            )
            import importlib

            importlib.invalidate_caches()
            from rapidocr_onnxruntime import RapidOCR
        except Exception as exc:  # noqa: BLE001
            _detector_state = (
                f"{DETECTOR_PACKAGE} isn't installed and couldn't be installed automatically "
                f"({type(exc).__name__}). In the terminal you start the app from, run:  "
                f"{sys.executable} -m pip install {DETECTOR_PACKAGE}"
            )
            return None
    try:
        _detector = RapidOCR()
    except Exception as exc:  # noqa: BLE001
        _detector_state = f"{DETECTOR_PACKAGE} failed to start: {exc}"
        return None
    _detector_state = "ready"
    return _detector


def detector_available() -> bool:
    return ensure_text_detector(install=False) is not None


def ocr_available() -> bool:
    """Whether anything can read text out of a picture: the scene-text
    detector, or failing that Tesseract."""
    return detector_available() or tesseract_available()


def _detector_findings(image: Image.Image, min_height: float, min_score: float = None):
    """Text the scene-text detector finds, as TextFindings with the
    detector's 0..1 score scaled to Tesseract's 0..100."""
    engine = ensure_text_detector()
    if engine is None:
        return None
    import numpy as np

    try:
        result, _elapsed = engine(np.asarray(image.convert("RGB")))
    except Exception:  # noqa: BLE001
        return None
    threshold = DETECTOR_MIN_SCORE if min_score is None else min_score
    findings = []
    for box, text, score in result or []:
        score = float(score)
        if score < threshold:
            continue
        word = (text or "").strip()
        if len(_LETTERS.findall(word)) < MIN_LETTERS and not any(ch.isdigit() for ch in word):
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        left, top = int(min(xs)), int(min(ys))
        width, height = int(max(xs)) - left, int(max(ys)) - top
        if min(width, height) < min_height:
            continue
        findings.append(TextFinding(text=word, confidence=score * 100.0, box=(left, top, width, height)))
    return findings


def find_text(image: Image.Image, min_confidence: float = None) -> TextCheckResult:
    """Words the OCR engine is confident it can read in `image`.

    `min_confidence` overrides MIN_CONFIDENCE for callers that want the
    doubtful words too (the whole-ad layer split accepts a low-scoring
    word when it sits on the same line as a confident one).

    Returns an empty result (available=False) when Tesseract isn't
    installed, so a caller can tell "checked, clean" from "couldn't
    check" -- which matter differently and must never be conflated.
    """
    # The detector first: it is the one that actually sees lettering on
    # a photograph. Tesseract's passes are kept for what it is good at
    # (small, clean, document-like type) and merged in.
    min_height = max(1.0, image.height * MIN_HEIGHT_FRACTION)
    detected = _detector_findings(
        image, min_height,
        None if min_confidence is None else float(min_confidence) / 100.0,
    )
    if not tesseract_available():
        if detected is None:
            return TextCheckResult(available=False)
        detected.sort(key=lambda f: -f.confidence)
        return TextCheckResult(available=True, findings=detected)

    import pytesseract

    # Greyscale: colour carries nothing for OCR and the conversion makes
    # the engine's own preprocessing more predictable.
    # Both polarities: Tesseract is trained on dark-on-light and reads
    # light lettering on a darker backdrop -- the usual case for a
    # headline over a photograph -- far better with the image inverted.
    from PIL import ImageOps

    grey = image.convert("L")
    pages = []
    for prepared in (grey, ImageOps.invert(grey)):
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
    if not pages and detected is None:
        return TextCheckResult(available=False)

    findings = list(detected or [])
    seen_boxes = {tuple(v // 8 for v in f.box) for f in findings}
    threshold = MIN_CONFIDENCE if min_confidence is None else float(min_confidence)
    for data in pages:
        for finding in _findings_from(data, min_height, seen_boxes, threshold):
            # A Tesseract word inside a detector box is the same text
            # read twice; the detector's box already covers it.
            if any(_inside(finding.box, other.box) for other in findings):
                continue
            findings.append(finding)
    findings.sort(key=lambda f: -f.confidence)
    return TextCheckResult(available=True, findings=findings)


def _inside(box, other) -> bool:
    l, t, w, h = box
    ol, ot, ow, oh = other
    cx, cy = l + w / 2, t + h / 2
    return ol <= cx <= ol + ow and ot <= cy <= ot + oh


def _findings_from(data, min_height, seen_boxes, min_confidence=None):
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
        if confidence < (MIN_CONFIDENCE if min_confidence is None else min_confidence):
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
        # The shorter side is the letter height, whichever way the text
        # runs: a wordmark down a bottle is a tall thin box, and padding
        # by its height would swallow the bottle.
        pad = max(2, int(round(min(width, height) * MASK_PADDING_FRACTION)))
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


def remove_text(image: Image.Image, result: "TextCheckResult", force: bool = False):
    """Paint out the words in `result`, reconstructing the background.

    Returns (cleaned_image, removed_count, reason). `reason` is None on
    success and a short explanation when nothing was done -- refusing
    loudly matters more than trying: a failed inpaint doesn't leave the
    image as it was, it leaves a smear where the text used to be.
    `force` paints out regardless of size, for a caller that has decided
    a smear beats lettering (see scrub_text()).
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
    if fraction > MAX_REMOVABLE_AREA_FRACTION and not force:
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
    if fraction > MAX_REMOVABLE_AREA_FRACTION:
        # A big hole (a forced paint-out of a headline) is filled at a
        # reduced size and scaled back, which gives a smooth fill in a
        # fraction of the time; the fill only lands inside the hole.
        h, w = mask.shape
        scale = 320.0 / max(h, w)
        sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
        small = cv2.resize(array, (sw, sh), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
        small_painted = cv2.inpaint(small, small_mask, 5, cv2.INPAINT_TELEA)
        fill = cv2.resize(small_painted, (w, h), interpolation=cv2.INTER_CUBIC)
        k = max(3, radius * 2 + 1)
        blend = cv2.GaussianBlur(mask, (k, k), 0).astype("float32")[:, :, None] / 255.0
        painted = (fill * blend + array * (1 - blend)).astype("uint8")
    else:
        painted = cv2.inpaint(array, mask, radius, cv2.INPAINT_TELEA)
    cleaned = Image.fromarray(painted[:, :, ::-1])
    return cleaned, len(result.findings), None


# scrub_text(): how many paint-out passes before falling back to a crop.
# Inpainting can leave enough of a letter to still read; a second pass
# over what is left usually finishes it.
SCRUB_PASSES = 3
# The crop fallback keeps the tallest text-free band of the frame, if it
# is at least this much of the height; below that too little picture
# remains to be worth calling a backdrop.
SCRUB_CROP_MIN_FRACTION = 0.45


def scrub_text(image: Image.Image):
    """Make `image` text-free, whatever it takes short of a new image.

    A backdrop goes under the template's own words, so lettering on it
    is never acceptable -- and a soft patch where a word was is. This
    paints out every word the OCR engine finds, re-reads, and repeats;
    if a headline is too big to paint out convincingly it crops to the
    largest band of the frame with no text in it and scales that back
    up to size. Returns (image, notes, leftover) where `leftover` is the
    TextCheckResult of the final read -- empty when the picture came out
    clean, `available=False` when nothing could be checked.
    """
    notes = []
    found = find_text(image)
    if not found.available:
        return image, notes, found
    if not found.found_text:
        return image, notes, found
    working = image
    removed_total = 0
    for _ in range(SCRUB_PASSES):
        cleaned, removed, reason = remove_text(working, found, force=True)
        if reason:
            break
        removed_total += removed
        working = cleaned
        found = find_text(working)
        if not found.found_text:
            break
    if removed_total:
        notes.append(f"{removed_total} word(s) painted out.")
    if not found.found_text:
        return working, notes, found

    # Still readable after painting: crop the lettering off instead.
    try:
        import numpy as np
    except ImportError:
        return working, notes, found
    mask = build_text_mask(working, found)
    rows = np.asarray(mask).max(axis=1) > 0
    best, start = (0, 0), None
    for y, has_text in enumerate(list(rows) + [True]):
        if not has_text and start is None:
            start = y
        elif has_text and start is not None:
            if y - start > best[1] - best[0]:
                best = (start, y)
            start = None
    top, bottom = best
    band = bottom - top
    if band < working.height * SCRUB_CROP_MIN_FRACTION:
        return working, notes, found
    # Keep the frame's own shape: the band is full width, so the width
    # is trimmed from the centre by the same proportion, then the crop
    # is scaled back to the original size.
    keep_w = max(8, int(round(working.width * band / working.height)))
    left = (working.width - keep_w) // 2
    cropped = working.crop((left, top, left + keep_w, bottom)).resize(working.size, Image.LANCZOS)
    notes.append(
        f"A headline too large to paint out was cropped off: the text-free "
        f"{band / working.height:.0%} of the frame was scaled back up to size."
    )
    found = find_text(cropped)
    if found.found_text:
        cleaned, removed, reason = remove_text(cropped, found, force=True)
        if not reason:
            cropped = cleaned
            found = find_text(cropped)
    return cropped, notes, found
