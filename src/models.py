"""Data models for the creative automation pipeline.

Keeping these as plain dataclasses (rather than a heavier schema library)
so the project has zero extra dependencies beyond Pillow/requests/yaml.
Validation is intentionally simple and explicit -- see load_brief() in
brief_loader.py for the actual required-field checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class BrandGuidelines:
    logo_path: Optional[str] = None
    colors: List[str] = field(default_factory=list)  # hex colors, e.g. "#0057B8"


@dataclass
class Product:
    name: str
    slug: str
    prompt_hint: Optional[str] = None  # extra detail to steer image generation
    asset_path: Optional[str] = None  # explicit path to a pre-made hero image (image OR video file)
    headline: Optional[str] = None  # per-product title override; falls back to brief.headline, then product name
    video_frame_seconds: Optional[float] = None  # only used when asset_path points to a video; None = middle frame


@dataclass
class CampaignBrief:
    name: str
    target_region: str
    target_audience: str
    message: str
    products: List[Product]
    brand: BrandGuidelines = field(default_factory=BrandGuidelines)
    language: Optional[str] = None  # explicit override; otherwise inferred from region
    output_sizes: Optional[List[Tuple[int, int]]] = None  # explicit (width, height) pairs; None = use CLI/defaults
    fit_mode: Optional[str] = None  # "crop" (fill, default) or "contain" (letterbox, no cropping); None = use CLI/default
    headline: Optional[str] = None  # campaign-wide title/tagline shown in the header band; a product's own
    # `headline` overrides this. If neither is set, each product's name is used instead.
    show_header: Optional[bool] = None  # None/True = show the header band (default); False = brief opts out.
    # A CLI --no-header flag always wins over this and forces the header off.

    def __post_init__(self):
        # One is enough. The web app writes briefs now, and a campaign
        # starts life with the single product that was just generated:
        # held to two, that brief fails to load and the campaign it was
        # meant to record never reaches the picker. The same rule was
        # also enforced in brief_loader.load_brief() -- two copies of it,
        # so relaxing one alone changed nothing.
        if len(self.products) < 1:
            raise ValueError(
                "Campaign brief must include at least one product "
                f"(got {len(self.products)})."
            )
