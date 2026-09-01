"""Ideogram image provider (Ideogram 3.0 / 4.0 API).

Included because it is the one option here aimed at graphic design rather
than photography -- it holds typography, posters and brand-style layouts
together where a general model smears them. For a campaign backdrop that
is often the difference between usable and not.

Needs a paid API key from https://ideogram.ai/platform. Keys can be
created before billing is set up, but stay inactive until a payment
method and credits are added.
"""

from __future__ import annotations

import io
import os

import requests
from PIL import Image

from .base import ImageProvider, ImageProviderError

API_BASE = "https://api.ideogram.ai"
DEFAULT_MODEL = "ideogram-v3"

# The API takes a named ratio ("16x9"), not pixel dimensions. A requested
# size is matched to the closest of these and the result is center-cropped
# to the target afterwards, so the ratio matters more than the exact
# pixels.
ASPECT_RATIOS = [
    "1x1", "16x9", "9x16", "3x2", "2x3", "4x3", "3x4",
    "16x10", "10x16", "2x1", "1x2", "3x1", "1x3", "4x5", "5x4",
]


def _closest_aspect(width: int, height: int) -> str:
    """The named ratio closest in shape to the size being asked for."""
    if width <= 0 or height <= 0:
        return "1x1"
    wanted = width / height

    def ratio_of(name):
        w, h = name.split("x")
        return int(w) / int(h)

    return min(ASPECT_RATIOS, key=lambda name: abs(ratio_of(name) - wanted))


class IdeogramProvider(ImageProvider):
    name = "ideogram"

    def __init__(self, api_token: str = None, model: str = None, timeout: int = 120):
        self.api_token = api_token or os.environ.get("IDEOGRAM_API_KEY")
        # "ideogram-v3" or "ideogram-v4" -- the model is part of the path
        # on this API rather than a body field.
        #
        # `or DEFAULT`, not a get() default: a .env written from the
        # example ships IDEOGRAM_MODEL= with nothing after it, and an
        # empty string is present as far as os.environ is concerned. That
        # produced a "/v1//generate" URL and a 404 on the first call.
        self.model = model or os.environ.get("IDEOGRAM_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        if not self.api_token:
            raise ImageProviderError(
                "IDEOGRAM_API_KEY is not set. Create one under API Keys at "
                "https://ideogram.ai/platform and put it in .env. Note that a key stays "
                "inactive until billing and credits are added there."
            )

    def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
        url = f"{API_BASE}/v1/{self.model}/generate"
        headers = {"Api-Key": self.api_token}
        fields = {
            "prompt": prompt,
            "aspect_ratio": _closest_aspect(width, height),
            "rendering_speed": "QUALITY",
            "num_images": 1,
        }

        try:
            resp = requests.post(url, headers=headers, json=fields, timeout=self.timeout)
            # This endpoint also accepts multipart (it takes reference
            # image files), and has been seen to reject a JSON body. Retry
            # the same fields as form data rather than failing on a
            # content-type technicality.
            if resp.status_code in (400, 415, 422):
                form = {k: str(v) for k, v in fields.items()}
                resp = requests.post(url, headers=headers, data=form, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            raise ImageProviderError(f"Ideogram request failed: {exc}") from exc

        if resp.status_code == 401:
            raise ImageProviderError(
                "Ideogram rejected the key (401). Check IDEOGRAM_API_KEY, and that billing "
                "and credits are set up at https://ideogram.ai/platform -- a key without "
                "them stays inactive."
            )
        if resp.status_code == 402:
            raise ImageProviderError(
                "Ideogram says the account is out of credits (402). Top up at "
                "https://ideogram.ai/platform under Billing."
            )
        if resp.status_code == 404:
            raise ImageProviderError(
                f"Ideogram has no endpoint at {url} (404). IDEOGRAM_MODEL should be a model "
                "path segment such as 'ideogram-v3' or 'ideogram-v4'."
            )
        if resp.status_code != 200:
            raise ImageProviderError(
                f"Ideogram returned HTTP {resp.status_code}: {resp.text[:300]}"
            )

        try:
            image_url = resp.json()["data"][0]["url"]
        except Exception as exc:  # noqa: BLE001
            raise ImageProviderError(
                f"Ideogram response wasn't in the expected shape: {exc}"
            ) from exc

        try:
            # The API returns a short-lived URL rather than the bytes.
            raw = requests.get(image_url, timeout=self.timeout).content
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise ImageProviderError(f"Couldn't fetch the image Ideogram returned: {exc}") from exc
