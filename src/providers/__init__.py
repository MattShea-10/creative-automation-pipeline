"""GenAI image provider registry.

get_provider(name) is the single place that knows how to construct each
provider, so pipeline.py and main.py stay decoupled from provider-specific
constructor args (API tokens, model ids, etc.).

Two providers are offered: Pollinations (free, no key) and Ideogram (paid
key, much better output). The mock provider is not one of them -- it draws
an offline placeholder and exists only as the automatic fallback when a
real provider fails, so it is deliberately absent from PROVIDER_NAMES (the
list the web form's dropdowns are built from) while staying constructible
by name for the CLI and for pipeline.py's fallback path.
"""

from __future__ import annotations

from .base import ImageProvider, ImageProviderError
from .mock_provider import MockImageProvider
from .pollinations_provider import PollinationsProvider
from .ideogram_provider import IdeogramProvider

# Selectable providers, in the order they're offered.
PROVIDER_NAMES = ["pollinations", "ideogram"]

# What an unset, unsubmitted or unrecognised provider field falls back
# to. Named rather than repeated as a literal, because code that has to
# tell "the user chose this" from "nothing was submitted" needs to test
# against it -- see the provider reconciliation in webapp.generate().
DEFAULT_PROVIDER_NAME = PROVIDER_NAMES[0]

# Plus the offline placeholder, which the CLI can still ask for by name.
ALL_PROVIDER_NAMES = PROVIDER_NAMES + ["mock"]


def get_provider(name: str) -> ImageProvider:
    name = (name or "pollinations").lower()
    if name == "pollinations":
        return PollinationsProvider()
    if name == "ideogram":
        return IdeogramProvider()
    if name == "mock":
        return MockImageProvider()
    raise ValueError(f"Unknown IMAGE_PROVIDER '{name}'. Choose from: {ALL_PROVIDER_NAMES}")


__all__ = [
    "ImageProvider",
    "ImageProviderError",
    "MockImageProvider",
    "PollinationsProvider",
    "IdeogramProvider",
    "get_provider",
    "PROVIDER_NAMES",
    "DEFAULT_PROVIDER_NAME",
    "ALL_PROVIDER_NAMES",
]
