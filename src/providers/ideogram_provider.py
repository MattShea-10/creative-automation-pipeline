"""Ideogram image provider (Ideogram 3.0 / 4.0 API).

Included because it is the one option here aimed at graphic design rather
than photography -- it holds typography, posters and brand-style layouts
together where a general model smears them. For a campaign backdrop that
is often the difference between usable and not.

Needs a paid API key from https://developer.ideogram.ai. Keys can be
created before billing is set up, but stay inactive until a payment
method and a prepaid balance are added there. API billing is separate
from an ideogram.ai subscription -- plan credits are spent by the web
app and buy nothing here, so an account showing plenty of them still
returns 402 until the API balance is funded.
"""

from __future__ import annotations

import io
import os
import re

import requests
from PIL import Image

from .base import ImageProvider, ImageProviderError

API_BASE = "https://api.ideogram.ai"
DEFAULT_MODEL = "ideogram-v3"

# The model is a path segment, so it has to look like one.
MODEL_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,31}$")

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
        if not MODEL_RE.match(self.model):
            # The model goes straight into the URL path, so anything that
            # isn't a model slug produces a mystifying 404 rather than an
            # obvious configuration error. This has actually happened: an
            # API key pasted onto the IDEOGRAM_MODEL line as well as the
            # key line, and the request went to /v1/<the key>/generate.
            raise ImageProviderError(
                f"IDEOGRAM_MODEL is set to something that isn't a model name "
                f"({len(self.model)} characters). It should be a short slug such as "
                f"'ideogram-v3' or 'ideogram-v4' -- the API key belongs on the "
                f"IDEOGRAM_API_KEY line instead."
            )
        self.timeout = timeout
        self._last_json_error = None
        if not self.api_token:
            raise ImageProviderError(
                "IDEOGRAM_API_KEY is not set. Create one under API Keys at "
                "https://developer.ideogram.ai and put it in .env. Note that a key stays "
                "inactive until a payment method and a prepaid balance are added there -- "
                "an ideogram.ai subscription is billed separately and doesn't cover the API."
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
            # This endpoint takes reference image files, so it speaks
            # multipart/form-data, and has been seen to reject a JSON
            # body outright. Retry the same fields as real multipart --
            # the (None, value) tuple is what makes requests send
            # multipart rather than urlencoded, which is a different
            # content type again and gets rejected just the same.
            if resp.status_code in (400, 415, 422):
                self._last_json_error = resp.text[:300]
                multipart = {k: (None, str(v)) for k, v in fields.items()}
                resp = requests.post(url, headers=headers, files=multipart, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            raise ImageProviderError(f"Ideogram request failed: {exc}") from exc

        if resp.status_code == 401:
            raise ImageProviderError(
                "Ideogram rejected the key (401). Check IDEOGRAM_API_KEY, and that a payment "
                "method and prepaid balance are set up at https://developer.ideogram.ai -- a "
                "key stays inactive until both exist, and an ideogram.ai subscription does not "
                "count towards it."
            )
        if resp.status_code == 402:
            raise ImageProviderError(
                "Ideogram says the API account has no credit (402). Note that this is a "
                "separate balance from an ideogram.ai subscription -- plan credits are for "
                "the web app and buy nothing on the API, so a healthy credit count there "
                "still gives a 402 here. Add a payment method and a prepaid balance under "
                "Billing at https://developer.ideogram.ai (from $1, $20 is the smallest "
                "preset)."
            )
        if resp.status_code == 404:
            raise ImageProviderError(
                f"Ideogram has no endpoint at {url} (404). IDEOGRAM_MODEL should be a model "
                "path segment such as 'ideogram-v3' or 'ideogram-v4'."
            )
        if resp.status_code != 200:
            detail = resp.text[:300]
            first = getattr(self, "_last_json_error", None)
            if first and first != detail:
                # Both shapes were refused -- report both, since which one
                # failed and how is the whole diagnosis.
                raise ImageProviderError(
                    f"Ideogram returned HTTP {resp.status_code} for {url}. "
                    f"As JSON: {first} | As multipart: {detail}"
                )
            raise ImageProviderError(
                f"Ideogram returned HTTP {resp.status_code} for {url}: {detail}"
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
