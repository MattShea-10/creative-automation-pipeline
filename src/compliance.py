"""Lightweight brand + legal compliance checks (the 'nice to have' bonus items).

These are heuristics, not real brand/legal review -- documented clearly as
such in the README. The point is to demonstrate the pipeline *hook* for
compliance gating (e.g. failing a creative, or routing it to human review)
rather than to build production-grade brand/legal detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import Image

from better_profanity import profanity as _profanity

_profanity.load_censor_words()

# Optional -- real trademark/logo detection needs a vision model or a paid
# API; this is deliberately just OCR text matching (catches a brand NAME
# printed in an uploaded image, not a logo mark), and it's fine for the
# whole feature to no-op if the system doesn't have the `tesseract` binary
# installed. See check_trademark_text() below.
try:
    import pytesseract as _pytesseract
except ImportError:
    _pytesseract = None

# Not exhaustive -- a short, illustrative list of well-known brand names to
# flag if they turn up as literal text in an uploaded image. A real system
# would use a maintained trademark database; this is a heuristic bonus
# check, same spirit as the rest of this module.
KNOWN_BRAND_NAMES = [
    "Nike", "Adidas", "Puma", "Reebok", "Under Armour",
    "Apple", "Google", "Microsoft", "Amazon", "Samsung", "Sony",
    "Coca-Cola", "Coca Cola", "Pepsi", "McDonald's", "McDonalds",
    "Starbucks", "Disney", "Netflix", "Meta", "Facebook", "Instagram",
    "Twitter", "Walmart", "Target", "IKEA", "Tesla", "BMW", "Mercedes-Benz",
    "Louis Vuitton", "Gucci", "Chanel", "Rolex", "Toyota", "Honda",
]

DEFAULT_PROHIBITED_WORDS = [
    "guaranteed",
    "cure",
    "miracle",
    "risk-free",
    "no side effects",
    "free money",
    "clinically proven",  # often requires substantiation/legal sign-off
]


@dataclass
class ComplianceResult:
    logo_present: Optional[bool] = None
    brand_color_match: Optional[bool] = None
    brand_color_distance: Optional[float] = None
    legal_flags: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        legal_ok = not self.legal_flags
        brand_ok = self.logo_present is not False and self.brand_color_match is not False
        return legal_ok and brand_ok


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def check_brand_colors(image: Image.Image, brand_colors: List[str], threshold: float = 90.0):
    """Heuristic: does the image's average color fall reasonably close to
    one of the declared brand colors? This is intentionally simple (a real
    system would use palette extraction / clustering) but is enough to
    demonstrate a brand-compliance gate.
    """
    if not brand_colors:
        return None, None

    small = image.convert("RGB").resize((32, 32))
    pixels = list(small.getdata())
    avg = tuple(sum(c[i] for c in pixels) / len(pixels) for i in range(3))

    best_distance = min(
        sum((avg[i] - _hex_to_rgb(hexc)[i]) ** 2 for i in range(3)) ** 0.5
        for hexc in brand_colors
    )
    return best_distance <= threshold, round(best_distance, 1)


def check_logo_present(image_size: Tuple[int, int], logo_composited: bool) -> bool:
    """We know deterministically whether add_logo_watermark() ran for this
    creative, since the pipeline controls that step. Kept as a separate
    function (rather than inlined) so a future version can swap in real
    logo-detection (template matching / a small vision model) without
    touching pipeline.py's call site.
    """
    return logo_composited


def check_legal_content(message: str, prohibited_words: Optional[List[str]] = None) -> List[str]:
    words = prohibited_words or DEFAULT_PROHIBITED_WORDS
    lowered = message.lower()
    return [w for w in words if w in lowered]


def run_compliance_checks(
    image: Image.Image,
    message: str,
    brand_colors: List[str],
    logo_composited: bool,
    prohibited_words: Optional[List[str]] = None,
) -> ComplianceResult:
    color_match, distance = check_brand_colors(image, brand_colors)
    return ComplianceResult(
        logo_present=check_logo_present(image.size, logo_composited),
        brand_color_match=color_match,
        brand_color_distance=distance,
        legal_flags=check_legal_content(message, prohibited_words),
    )


def check_profanity(text: str) -> bool:
    """True if `text` contains profanity (handles common leetspeak/spacing
    tricks via the better-profanity library's built-in wordlist). Used as a
    hard gate -- unlike everything else in this module, this one is meant
    to block, not just warn.
    """
    if not text:
        return False
    return bool(_profanity.contains_profanity(text))


def check_trademark_text(image: Image.Image, brand_names: Optional[List[str]] = None) -> List[str]:
    """Best-effort: OCRs `image` and returns any known brand names found as
    literal text in it. This is text-only -- it can't recognize an actual
    logo mark, only a brand *name* someone typed into the creative. Always
    returns [] rather than raising, including when the `tesseract` binary
    isn't installed on this machine (a "plus" feature, not a requirement --
    see README for the optional setup).
    """
    if _pytesseract is None:
        return []
    names = brand_names or KNOWN_BRAND_NAMES
    try:
        extracted = _pytesseract.image_to_string(image.convert("RGB"))
    except Exception:
        return []
    found = []
    for name in names:
        pattern = r"\b" + re.escape(name).replace(r"\-", "[- ]?") + r"\b"
        if re.search(pattern, extracted, re.IGNORECASE):
            found.append(name)
    return found
