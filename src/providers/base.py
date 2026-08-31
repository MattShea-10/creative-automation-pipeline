"""Abstract interface for GenAI image providers.

Design decision: the pipeline never talks to a specific vendor SDK directly.
Everything goes through this interface, so swapping providers (or adding
Adobe Firefly, Midjourney, a fine-tuned in-house model, etc. later) is a
matter of writing one small adapter class -- nothing in pipeline.py changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from PIL import Image


class ImageProviderError(Exception):
    """Raised when a provider fails to produce an image."""


class ImageProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
        """Generate a single hero image for the given prompt.

        Implementations should raise ImageProviderError on failure so the
        pipeline can decide whether to fall back to another provider.
        """
        raise NotImplementedError
