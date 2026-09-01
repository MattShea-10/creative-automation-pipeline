"""Ideogram image provider.

Included because it is the one in this set aimed squarely at graphic
design rather than photography -- it holds typography, posters and
brand-style layouts together where a general model smears them. For a
campaign backdrop that is often the difference between usable and not.

Renders a 1536px long edge in its widest aspects, so it sits between
SDXL (~1024) and DALL-E 3 (1792) on resolution.

Needs a paid API key from https://ideogram.ai (Developer / API section).
"""

from __future__ import annotations

import io
import os

import requests
from PIL import Image

from .base import ImageProvider, ImageProviderError

API_URL = "https://api.ideogram.ai/generate"
DEFAULT_MODEL = "V_2"

# Ideogram takes a named aspect ratio, not pixel dimensions. These are
# the ones it renders, with the pixel size each produces -- the size is
# what matters here, since a requested size gets snapped to whichever
# ratio is closest and the result is cropped to the target afterwards.
ASPECT_RATIOS = [
    ("ASPECT_1_1", (1024, 1024)),
    ("ASPECT_16_9", (1536, 864)),
    ("ASPECT_9_16", (864, 1536)),
    ("ASPECT_3_2", (1248, 832)),
    ("ASPECT_2_3", (832, 1248)),
    ("ASPECT_4_3", (1280, 960)),
    ("ASPECT_3_4", (960, 1280)),
    ("ASPECT_16_10", (1456, 912)),
    ("ASPECT_10_16", (912, 1456)),
]


def _closest_aspect(width: int, height: int) -> str:
    """The named ratio whose shape best matches the request.

    Ratio first, then pixels: the result is center-cropped to the target's
    shape afterwards, so a badly matched aspect throws away more than a
    slightly smaller render does.
    """
    if width <= 0 or height <= 0:
        return ASPECT_RATIOS[0][0]
    wanted = width / height
    return min(
        ASPECT_RATIOS,
        key=lambda entry: (
            round(abs((entry[1][0] / entry[1][1]) - wanted), 3),
            -(entry[1][0] * entry[1][1]),
        ),
    )[0]


class IdeogramProvider(ImageProvider):
    name = "ideogram"

    def __init__(self, api_token: str = None, model: str = None, timeout: int = 120):
        self.api_token = api_token or os.environ.get("IDEOGRAM_API_KEY")
        self.model = model or os.environ.get("IDEOGRAM_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        if not self.api_token:
            raise ImageProviderError(
                "IDEOGRAM_API_KEY is not set. Create a key in the API section at "
                "https://ideogram.ai and put it in .env."
            )

    def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
        payload = {
            "image_request": {
                "prompt": prompt,
                "model": self.model,
                "aspect_ratio": _closest_aspect(width, height),
                "magic_prompt_option": "AUTO",
            }
        }
        try:
            resp = requests.post(
                API_URL,
                headers={"Api-Key": self.api_token, "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise ImageProviderError(f"Ideogram request failed: {exc}") from exc

        if resp.status_code == 401:
            raise ImageProviderError("Ideogram rejected the key (401). Check IDEOGRAM_API_KEY.")
        if resp.status_code != 200:
            raise ImageProviderError(
                f"Ideogram returned HTTP {resp.status_code}: {resp.text[:300]}"
            )

        try:
            url = resp.json()["data"][0]["url"]
        except Exception as exc:  # noqa: BLE001
            raise ImageProviderError(
                f"Ideogram response wasn't in the expected shape: {exc}"
            ) from exc

        try:
            # Ideogram hands back a short-lived URL rather than the bytes.
            raw = requests.get(url, timeout=self.timeout).content
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise ImageProviderError(f"Couldn't fetch the image Ideogram returned: {exc}") from exc
