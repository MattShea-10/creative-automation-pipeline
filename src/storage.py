"""Asset storage abstraction.

The brief said storage 'can be Azure, AWS or Dropbox' -- for a local
proof-of-concept, a real cloud dependency would add setup friction (and
credentials) without changing anything about the pipeline logic. So this
defines the same narrow interface (get/put by product slug) a cloud-backed
implementation would need, and backs it with the local filesystem.

To point this at real cloud storage later: implement AssetStore with the
same two methods using boto3 (S3), azure-storage-blob, or the dropbox SDK,
and pass an instance of it into CreativePipeline instead of
LocalAssetStore -- nothing else in the pipeline changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from .image_ops import VIDEO_EXTENSIONS, open_as_rgb

# Video files found via the naming convention always use open_as_rgb()'s
# default (middle) frame -- there's no per-asset config attached to a
# convention-based lookup the way there is for a product's explicit
# asset_path, which additionally supports a video_frame_seconds override.
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff") + VIDEO_EXTENSIONS


class AssetStore(ABC):
    @abstractmethod
    def get_hero_image(self, product_slug: str) -> Optional[Tuple[Image.Image, Path]]:
        """Return (image, source_path) for this product if one already exists, else None."""

    @abstractmethod
    def put_hero_image(self, product_slug: str, image: Image.Image) -> Path:
        """Persist a (generated) hero image for this product and return its path."""


class LocalAssetStore(AssetStore):
    def __init__(self, input_dir: str = "assets", cache_dir: str = "assets/generated_cache"):
        self.input_dir = Path(input_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _find_user_provided(self, product_slug: str) -> Optional[Path]:
        # Look for assets/<slug>.png, assets/<slug>/hero.png, etc.
        candidates = [self.input_dir / f"{product_slug}{ext}" for ext in SUPPORTED_EXTENSIONS]
        candidates += [
            self.input_dir / product_slug / f"hero{ext}" for ext in SUPPORTED_EXTENSIONS
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def get_hero_image(self, product_slug: str) -> Optional[Tuple[Image.Image, Path]]:
        user_path = self._find_user_provided(product_slug)
        if user_path:
            return open_as_rgb(user_path), user_path

        cached = self.cache_dir / f"{product_slug}.png"
        if cached.exists():
            return Image.open(cached).convert("RGB"), cached

        return None

    def put_hero_image(self, product_slug: str, image: Image.Image) -> Path:
        out_path = self.cache_dir / f"{product_slug}.png"
        image.save(out_path)
        return out_path
