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

    # Whether the vendor has a real "exclude this" field. It matters more
    # than it looks: diffusion models handle negation in the positive
    # prompt badly, and "no text, no words" there puts the tokens "text"
    # and "words" into the very prompt that is steering the image --
    # which can produce more lettering, not less. Ideogram's own docs say
    # the positive prompt takes precedence over the negative one, so the
    # exclusion belongs in the negative field wherever there is one.
    #
    # Providers that leave this False still accept the argument and fold
    # it into the prompt as best they can, so callers never branch.
    supports_negative_prompt: bool = False

    @abstractmethod
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        negative_prompt: str = None,
    ) -> Image.Image:
        """Generate a single hero image for the given prompt.

        `negative_prompt` describes what to keep OUT of the image.

        Implementations should raise ImageProviderError on failure so the
        pipeline can decide whether to fall back to another provider.
        """
        raise NotImplementedError
