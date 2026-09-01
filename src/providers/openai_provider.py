"""OpenAI image provider.

The highest-resolution option wired into this app. DALL-E 3 renders a
1792px long edge, which is close enough to the 1920px this project's
largest templates need that the upscale is barely there -- against 768
from Pollinations and roughly 1024 from SDXL.

Needs a paid API key from platform.openai.com. Worth stating plainly
because it catches people out: a ChatGPT Plus or Pro subscription does
NOT include API access. They are separately billed products, and a
ChatGPT plan will return 401 here.
"""

from __future__ import annotations

import base64
import io
import os

import requests
from PIL import Image

from .base import ImageProvider, ImageProviderError

API_URL = "https://api.openai.com/v1/images/generations"
DEFAULT_MODEL = "dall-e-3"

# What each model will actually render. Asking for anything else is an
# error from the API, so a requested size is snapped to one of these
# rather than passed through and rejected.
MODEL_SIZES = {
    "dall-e-3": [(1024, 1024), (1792, 1024), (1024, 1792)],
    "dall-e-2": [(256, 256), (512, 512), (1024, 1024)],
    "gpt-image-1": [(1024, 1024), (1536, 1024), (1024, 1536)],
}
FALLBACK_SIZES = [(1024, 1024)]


def _closest_supported(width: int, height: int, sizes) -> tuple:
    """Pick the supported size whose shape best matches what was asked for.

    Aspect ratio first, pixel count second: a 1920x1080 request wants the
    landscape option even though the square one is closer in area, because
    the result is center-cropped to the target's ratio afterwards and a
    mismatched shape throws away the difference.
    """
    if width <= 0 or height <= 0:
        return sizes[0]
    wanted = width / height
    return min(
        sizes,
        key=lambda s: (round(abs((s[0] / s[1]) - wanted), 3), -(s[0] * s[1])),
    )


class OpenAIProvider(ImageProvider):
    name = "openai"

    def __init__(self, api_token: str = None, model: str = None, timeout: int = 120):
        self.api_token = api_token or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("OPENAI_IMAGE_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        if not self.api_token:
            raise ImageProviderError(
                "OPENAI_API_KEY is not set. Create a key at "
                "https://platform.openai.com/api-keys and put it in .env. Note that a "
                "ChatGPT Plus subscription does not include API access -- the API is "
                "billed separately."
            )

    def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
        sizes = MODEL_SIZES.get(self.model, FALLBACK_SIZES)
        size_w, size_h = _closest_supported(width, height, sizes)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "size": f"{size_w}x{size_h}",
        }
        # Quality is spelled differently per model family, and sending the
        # wrong word is a 400 rather than something ignored.
        if self.model == "dall-e-3":
            payload["quality"] = "hd"
            payload["response_format"] = "b64_json"
        elif self.model.startswith("gpt-image"):
            payload["quality"] = "high"

        try:
            resp = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise ImageProviderError(f"OpenAI request failed: {exc}") from exc

        if resp.status_code == 401:
            raise ImageProviderError(
                "OpenAI rejected the key (401). Check OPENAI_API_KEY, and note that a "
                "ChatGPT subscription is not API access -- the API needs its own billing."
            )
        if resp.status_code != 200:
            detail = resp.text[:300]
            raise ImageProviderError(f"OpenAI returned HTTP {resp.status_code}: {detail}")

        try:
            entry = resp.json()["data"][0]
        except Exception as exc:  # noqa: BLE001
            raise ImageProviderError(f"OpenAI response wasn't in the expected shape: {exc}") from exc

        try:
            if entry.get("b64_json"):
                raw = base64.b64decode(entry["b64_json"])
            elif entry.get("url"):
                # dall-e-2 and any model that ignores response_format hand
                # back a short-lived URL instead of the bytes.
                raw = requests.get(entry["url"], timeout=self.timeout).content
            else:
                raise ImageProviderError("OpenAI returned neither image bytes nor a URL.")
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except ImageProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ImageProviderError(f"Couldn't read the image OpenAI returned: {exc}") from exc
