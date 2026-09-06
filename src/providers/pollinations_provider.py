"""Pollinations.ai image provider.

Chosen as the default 'real' GenAI provider because it is genuinely free,
requires no signup and no API key, and exposes a plain HTTP GET endpoint --
which minimizes setup friction for whoever is reviewing this take-home.
(Trade-off, documented in the README: no uptime/quality SLA, and output
quality/consistency is lower than a paid model like gpt-image-1 or Firefly.)
"""

from __future__ import annotations

import io
import os
import random
import urllib.parse

import requests
from PIL import Image

from .base import ImageProvider, ImageProviderError

BASE_URL = "https://image.pollinations.ai/prompt/{prompt}"

# The largest edge worth asking for. The default model returns 768x768
# whatever size is requested, so a 1920x1920 request buys nothing but a
# heavier job on a free, oversubscribed service -- and that is where the
# 500s and 40-second timeouts were coming from. Anything larger is asked
# for at this size (shape kept) and upscaled afterwards, which is what
# happened anyway.
MAX_REQUEST_EDGE = 1024

# One more go after a server-side failure (5xx, timeout, dropped
# connection). The service fails intermittently rather than for long, so
# a second request a moment later usually goes through; anything past
# one retry is just waiting on an outage.
RETRIES = 1


class PollinationsProvider(ImageProvider):
    name = "pollinations"

    def __init__(self, timeout: int = 40, seed: int = None, model: str = None):
        self.timeout = timeout
        # Which model to ask for. Left unset the endpoint picks its own,
        # which in practice returns 768x768 however large a size is
        # requested -- so anything bigger in the batch is upscaled from
        # that, and looks it. Some models honour larger dimensions, so
        # this is exposed as a knob rather than hard-coded: set
        # POLLINATIONS_MODEL in .env (e.g. "flux") and compare the size
        # reported on the results page.
        self.model = model if model is not None else (os.environ.get("POLLINATIONS_MODEL") or None)
        # A caller that wants the same prompt to come back identical can
        # pin the seed; left alone, every request gets a fresh one.
        self.seed = seed

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        negative_prompt: str = None,
        style_reference: bytes = None,  # not supported here; described in words by the caller
    ) -> Image.Image:
        # The endpoint is a plain GET with no exclusion field, so the
        # only place an exclusion can go is the prompt itself. Weaker
        # than a real negative prompt -- see ImageProvider's note on why
        # negation in a positive prompt can backfire -- but it is this or
        # nothing here.
        if negative_prompt:
            prompt = f"{prompt}, {negative_prompt}"
        encoded = urllib.parse.quote(prompt)
        url = BASE_URL.format(prompt=encoded)
        longest = max(width, height)
        if longest > MAX_REQUEST_EDGE:
            scale = MAX_REQUEST_EDGE / longest
            width, height = max(64, round(width * scale)), max(64, round(height * scale))
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
        if self.model:
            params["model"] = self.model
        last_exc = None
        for attempt in range(RETRIES + 1):
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
                if getattr(resp, "status_code", 200) >= 500:
                    # Their side, not ours: try once more with a new seed.
                    last_exc = f"HTTP {resp.status_code} from image.pollinations.ai"
                    if self.seed is None:
                        params["seed"] = random.randrange(10**6)
                    continue
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            except ImageProviderError:
                raise
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                if attempt < RETRIES:
                    continue
            except Exception as exc:  # noqa: BLE001 - deliberately broad, wrapped below
                raise ImageProviderError(f"Pollinations request failed: {exc}") from exc
        raise ImageProviderError(
            f"Pollinations request failed after {RETRIES + 1} attempts: {last_exc}. The service "
            "is free and goes down intermittently -- try again in a minute, or switch the "
            "provider to Ideogram."
        )
