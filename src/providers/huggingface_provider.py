"""Optional Hugging Face Inference API provider.

Not the default, but documented as an easy swap-in: HF's free tier is a
reasonable stand-in for a 'real' hosted diffusion model (SDXL, etc.) when
Pollinations isn't suitable and a paid vendor (OpenAI, Firefly, Stability)
isn't available.
"""

from __future__ import annotations

import io
import os

import requests
from PIL import Image

from .base import ImageProvider, ImageProviderError

API_URL_TEMPLATE = "https://api-inference.huggingface.co/models/{model}"
DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


class HuggingFaceProvider(ImageProvider):
    name = "huggingface"

    def __init__(self, api_token: str | None = None, model: str | None = None, timeout: int = 60):
        self.api_token = api_token or os.environ.get("HUGGINGFACE_API_TOKEN")
        self.model = model or os.environ.get("HUGGINGFACE_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        if not self.api_token:
            raise ImageProviderError(
                "HUGGINGFACE_API_TOKEN is not set. Get a free token at "
                "https://huggingface.co/settings/tokens and set it in .env."
            )

    def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
        url = API_URL_TEMPLATE.format(model=self.model)
        headers = {"Authorization": f"Bearer {self.api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {"width": width, "height": height},
            "options": {"wait_for_model": True},
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "application/json" in content_type:
                # Model still loading or returned a structured error.
                raise ImageProviderError(f"HF Inference API returned JSON, not an image: {resp.text[:300]}")
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except ImageProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ImageProviderError(f"Hugging Face request failed: {exc}") from exc
