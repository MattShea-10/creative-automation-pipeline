"""Best-effort layer reconstruction for a flat, model-generated ad.

A "whole ad" generation is one picture: the headline, the hero and the
logo are pixels the model painted together, and nothing can recover the
layers it never had. What CAN be done is a reconstruction after the
fact, and this module does the honest version of it:

  text        -- the words the OCR engine finds, cut out as pixels (a
                 picture of text: movable, hideable, not retypeable)
  subject     -- the main foreground, cut out by a background-removal
                 model (rembg's isnet, or the small u2netp) when one is
                 installed, else by OpenCV's GrabCut seeded from the
                 frame -- far rougher
  background  -- what is left, with the subject and text painted out
                 (inpainted -- soft, and smeared where a large subject
                 was)

Every layer is optional: without Tesseract there is no text layer,
without a plausible foreground there is no subject layer, and the
result says so in `notes` rather than inventing one. A painted logo
cannot be told from the artwork and lands wherever it falls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

# GrabCut's initial rectangle: everything outside it is taken as certain
# background. An ad's subject is rarely at the very edge, and a margin
# this size leaves room for a bleed without clipping a large hero.
SUBJECT_MARGIN = 0.06
GRABCUT_ITERATIONS = 5
# A "subject" that is nearly the whole frame is the background wearing
# a hat; one that is a few pixels is noise. Neither is worth a layer.
SUBJECT_MIN_FRACTION = 0.015
SUBJECT_MAX_FRACTION = 0.85
# Inpainting a large region at full size is slow and no better than
# doing it small and scaling up, since the fill is soft either way.
INPAINT_WORK_EDGE = 320


@dataclass
class AdSplit:
    background: Image.Image
    subject: Optional[Image.Image] = None  # RGBA, full canvas
    text: Optional[Image.Image] = None  # RGBA, full canvas
    notes: list = field(default_factory=list)

    def layers(self) -> list:
        """(name, RGBA) pairs, back to front, for save_layered_psd()."""
        out = [("background", self.background.convert("RGBA"))]
        if self.subject is not None:
            out.append(("subject (painted)", self.subject))
        if self.text is not None:
            out.append(("text (painted)", self.text))
        return out


def _cutout(image: Image.Image, alpha) -> Image.Image:
    """The image with `alpha` (uint8 HxW) as its transparency."""
    import numpy as np

    rgba = np.dstack([np.array(image.convert("RGB")), alpha])
    return Image.fromarray(rgba.astype("uint8"), "RGBA")


def _feather(mask, radius_px: int):
    """A soft-edged copy of a hard 0/255 mask."""
    import cv2

    if radius_px <= 0:
        return mask
    k = radius_px * 2 + 1
    return cv2.GaussianBlur(mask, (k, k), 0)


def _inpaint_under(image: Image.Image, mask, small: bool):
    """Paint out `mask` (uint8, 255 = remove). `small` regions are
    inpainted at full size; large ones at a reduced size and scaled back,
    which is what keeps a hero-sized hole from taking half a minute."""
    import cv2
    import numpy as np

    bgr = np.array(image.convert("RGB"))[:, :, ::-1].copy()
    h, w = mask.shape
    # Grow the hole a little so the edge of the removed thing goes too.
    grow = max(2, int(round(max(h, w) * 0.006)))
    dilated = cv2.dilate(mask, np.ones((grow * 2 + 1, grow * 2 + 1), np.uint8))
    if small:
        painted = cv2.inpaint(bgr, dilated, max(3, grow), cv2.INPAINT_TELEA)
    else:
        scale = INPAINT_WORK_EDGE / max(h, w)
        sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
        small_bgr = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(dilated, (sw, sh), interpolation=cv2.INTER_NEAREST)
        small_painted = cv2.inpaint(small_bgr, small_mask, 5, cv2.INPAINT_TELEA)
        fill = cv2.resize(small_painted, (w, h), interpolation=cv2.INTER_CUBIC)
        # Only the hole takes the soft fill; everything else stays sharp.
        blend = _feather(dilated, grow).astype("float32")[:, :, None] / 255.0
        painted = (fill * blend + bgr * (1 - blend)).astype("uint8")
    return Image.fromarray(painted[:, :, ::-1])


# Words the OCR engine is sure of, and the doubtful ones it will accept
# when they sit on the same line as a sure one ("for" beside
# "rehydrate"): short words score low on their own.
SURE_CONFIDENCE = 70.0
LOOSE_CONFIDENCE = 40.0

# Background-removal models, best first. The first is a 176 MB download
# on first use (kept in ~/.u2net); the second is 4.6 MB and noticeably
# rougher at hair and edges.
REMBG_MODELS = ("isnet-general-use", "u2netp")


def _same_line(a, b) -> bool:
    """Two OCR boxes on one line of text, close enough to be neighbours."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if abs((ay + ah / 2) - (by + bh / 2)) > max(ah, bh) * 0.6:
        return False
    if abs(ah - bh) > max(ah, bh) * 0.6:
        return False
    gap = max(bx - (ax + aw), ax - (bx + bw))
    return gap < max(ah, bh) * 2.5


def _text_findings(image: Image.Image):
    """The sure words plus the doubtful ones that share a line with a
    sure one. Returns (result, findings)."""
    from .text_check import find_text

    loose = find_text(image, min_confidence=LOOSE_CONFIDENCE)
    if not loose.available:
        return loose, []
    sure = [f for f in loose.findings if f.confidence >= SURE_CONFIDENCE]
    kept = list(sure)
    for f in loose.findings:
        if f.confidence >= SURE_CONFIDENCE:
            continue
        if any(_same_line(f.box, anchor.box) for anchor in sure):
            kept.append(f)
    return loose, kept


def _find_subject_rembg(image: Image.Image):
    """A 0..255 alpha of the main foreground from a background-removal
    model, or None when rembg (or a model) isn't available."""
    try:
        from rembg import new_session, remove
    except ImportError:
        return None, "rembg isn't installed"
    import numpy as np

    last = None
    for model in REMBG_MODELS:
        try:
            cut = remove(image.convert("RGB"), session=new_session(model))
        except Exception as exc:  # noqa: BLE001 -- a failed model download, most often
            last = f"{model}: {str(exc)[:80]}"
            continue
        alpha = np.array(cut.convert("RGBA"))[:, :, 3]
        fraction = float((alpha > 128).sum()) / alpha.size
        if fraction < SUBJECT_MIN_FRACTION or fraction > SUBJECT_MAX_FRACTION:
            return None, f"{model} found no clear subject"
        return alpha, model
    return None, last or "no model could be loaded"


def _find_subject(image: Image.Image):
    """A 0/255 mask of the main foreground, or None when GrabCut finds
    nothing worth calling one."""
    import cv2
    import numpy as np

    bgr = np.array(image.convert("RGB"))[:, :, ::-1].copy()
    h, w = bgr.shape[:2]
    mx, my = int(w * SUBJECT_MARGIN), int(h * SUBJECT_MARGIN)
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    # Tidy: close small gaps, drop specks, keep the big pieces.
    k = max(3, int(round(max(h, w) * 0.01)) | 1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    keep = np.zeros_like(fg)
    area = h * w
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= area * 0.003:
            keep[labels == i] = 255
    fraction = float((keep > 0).sum()) / area
    if fraction < SUBJECT_MIN_FRACTION or fraction > SUBJECT_MAX_FRACTION:
        return None
    return keep


def split_ad(image: Image.Image) -> AdSplit:
    """Reconstruct background / subject / text layers from a flat ad."""
    try:
        import cv2  # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError:
        return AdSplit(background=image.convert("RGB"), notes=["OpenCV isn't installed, so the ad was kept as one layer."])

    from .text_check import TextCheckResult, build_text_mask

    working = image.convert("RGB")
    notes = []
    text_layer = None
    edge = max(1, int(round(max(working.size) * 0.002)))

    # 1. Text first: it sits on top of everything, and taking it out
    #    before looking for the subject stops a headline being read as
    #    one.
    found, words = _text_findings(working)
    if not found.available:
        notes.append("Tesseract isn't installed, so no text layer was cut out.")
    elif words:
        mask = build_text_mask(working, TextCheckResult(available=True, findings=words))
        text_layer = _cutout(working, _feather(mask, edge))
        working = _inpaint_under(working, mask, small=True)
        notes.append(
            f"{len(words)} word(s) cut to a 'text (painted)' layer: "
            + ", ".join(sorted({w.text for w in words}))[:120]
            + "."
        )
    else:
        notes.append("No text found to cut out.")

    # 2. The subject: a real background-removal model when there is one,
    #    GrabCut when there isn't.
    subject_layer = None
    alpha, how = _find_subject_rembg(working)
    if alpha is not None:
        subject_mask = (alpha > 128).astype("uint8") * 255
        subject_layer = _cutout(working, alpha)
        working = _inpaint_under(working, subject_mask, small=False)
        fraction = float((subject_mask > 0).sum()) / (working.width * working.height)
        notes.append(
            f"Subject cut to its own layer with {how} ({fraction:.0%} of the frame); "
            "the background beneath it is inpainted."
        )
    else:
        subject_mask = _find_subject(working)
        if subject_mask is not None:
            subject_layer = _cutout(working, _feather(subject_mask, edge * 2))
            working = _inpaint_under(working, subject_mask, small=False)
            fraction = float((subject_mask > 0).sum()) / (working.width * working.height)
            notes.append(
                f"Subject cut to its own layer with OpenCV GrabCut ({fraction:.0%} of the frame) -- "
                f"rough edges are likely; install rembg for a clean cutout ({how}). "
                "The background beneath it is inpainted."
            )
        else:
            notes.append(f"No clear subject found ({how}) -- the picture is one background layer.")

    return AdSplit(background=working, subject=subject_layer, text=text_layer, notes=notes)
