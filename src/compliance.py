"""Lightweight brand + legal compliance checks (the 'nice to have' bonus items).

These are heuristics, not real brand/legal review -- documented clearly as
such in the README. The point is to demonstrate the pipeline *hook* for
compliance gating (e.g. failing a creative, or routing it to human review)
rather than to build production-grade brand/legal detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import Image

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
