"""Pollinations.ai image provider.

Chosen as the default 'real' GenAI provider because it is genuinely free,
requires no signup and no API key, and exposes a plain HTTP GET endpoint --
which minimizes setup friction for whoever is reviewing this take-home.
(Trade-off, documented in the README: no uptime/quality SLA, and output
quality/consistency is lower than a paid model like gpt-image-1 or Firefly.)
"""

from __future__ import annotations

import io
import random
import urllib.parse

import requests
from PIL import Image

from .base import ImageProvider, ImageProviderError

BASE_URL = "https://image.pollinations.ai/prompt/{prompt}"


class PollinationsProvider(ImageProvider):
    name = "pollinations"

    def __init__(self, timeout: int = 40, seed: int = None):
        self.timeout = timeout
        # A caller that wants the same prompt to come back identical can
        # pin the seed; left alone, every request gets a fresh one.
        self.seed = seed

    def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
        encoded = urllib.parse.quote(prompt)
        url = BASE_URL.format(prompt=encoded)
        params = {
            "width": width,
            "height": height,
            "nologo": "true",
            # A NEW seed per request. This used to be derived from the
            # prompt, so re-running the same prompt returned a
            # byte-identical image -- which reads as the generator being
            # stuck rather than as reproducibility, since asking again is
            # how you ask for a different take. Pin `seed` on the
            # provider if you want the old behaviour.
            "seed": self.seed if self.seed is not None else random.randrange(10**6),
        }
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - deliberately broad, wrapped below
            raise ImageProviderError(f"Pollinations request failed: {exc}") from exc
