"""GenAI image provider registry.

get_provider(name) is the single place that knows how to construct each
provider, so pipeline.py and main.py stay decoupled from provider-specific
constructor args (API tokens, model ids, etc.).
"""

from __future__ import annotations

from .base import ImageProvider, ImageProviderError
from .mock_provider import MockImageProvider
from .pollinations_provider import PollinationsProvider
from .huggingface_provider import HuggingFaceProvider

PROVIDER_NAMES = ["mock", "pollinations", "huggingface"]


def get_provider(name: str) -> ImageProvider:
    name = (name or "pollinations").lower()
    if name == "mock":
        return MockImageProvider()
    if name == "pollinations":
        return PollinationsProvider()
    if name == "huggingface":
        return HuggingFaceProvider()
    raise ValueError(f"Unknown IMAGE_PROVIDER '{name}'. Choose from: {PROVIDER_NAMES}")


__all__ = [
    "ImageProvider",
    "ImageProviderError",
    "MockImageProvider",
    "PollinationsProvider",
    "HuggingFaceProvider",
    "get_provider",
    "PROVIDER_NAMES",
]
