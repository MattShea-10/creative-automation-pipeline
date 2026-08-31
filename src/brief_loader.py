"""Load a campaign brief from JSON or YAML into a CampaignBrief object."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import yaml

from .image_ops import SIZE_PRESETS, parse_size, parse_sizes
from .models import BrandGuidelines, CampaignBrief, Product

REQUIRED_TOP_LEVEL = ["target_region", "target_audience", "message", "products"]


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "product"


def _read_raw(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    # YAML is a superset of JSON, so this also handles .yaml/.yml/.json fine.
    return yaml.safe_load(text)


def load_brief(path: str) -> CampaignBrief:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Campaign brief not found: {path}")

    raw = _read_raw(p)
    # Allow either a top-level "campaign:" wrapper or a flat document.
    data = raw.get("campaign", raw) if isinstance(raw, dict) else raw
    if not isinstance(data, dict):
        raise ValueError("Campaign brief must be a mapping/object at the top level.")

    missing = [k for k in REQUIRED_TOP_LEVEL if not data.get(k)]
    if missing:
        raise ValueError(f"Campaign brief is missing required field(s): {missing}")

    products_raw = data["products"]
    if not isinstance(products_raw, list) or len(products_raw) < 2:
        raise ValueError(
            "Campaign brief must include a 'products' list with at least two products."
        )

    products = []
    for item in products_raw:
        if isinstance(item, str):
            name = item
            item = {}
        else:
            name = item.get("name")
        if not name:
            raise ValueError(f"Each product needs a 'name': {item}")
        video_frame_seconds = item.get("video_frame_seconds")
        if video_frame_seconds is not None:
            try:
                video_frame_seconds = float(video_frame_seconds)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Product '{name}': 'video_frame_seconds' must be a number, got {video_frame_seconds!r}"
                )
            if video_frame_seconds < 0:
                raise ValueError(f"Product '{name}': 'video_frame_seconds' must not be negative.")
        products.append(
            Product(
                name=name,
                slug=item.get("slug") or _slugify(name),
                prompt_hint=item.get("prompt_hint"),
                asset_path=item.get("asset_path"),
                headline=item.get("headline"),
                video_frame_seconds=video_frame_seconds,
            )
        )

    brand_raw = data.get("brand", {}) or {}
    brand = BrandGuidelines(
        logo_path=brand_raw.get("logo"),
        colors=brand_raw.get("colors", []) or [],
    )

    output_sizes = None
    sizes_raw = data.get("output_sizes")
    if isinstance(sizes_raw, str):
        # A bare (optionally comma-separated) preset/size spec, e.g.
        # output_sizes: "web-top7" or output_sizes: "default,web-top7".
        output_sizes = parse_sizes(sizes_raw)
    elif sizes_raw:
        # A YAML/JSON list where each entry is a preset name, a
        # "WIDTHxHEIGHT" string, or a {width, height} object -- any mix of
        # the three, combined and de-duplicated (order preserved) so a
        # brief can request more than one size family at once, e.g.:
        #   output_sizes: ["default", "web-top7", {width: 1200, height: 628}]
        resolved = []
        for item in sizes_raw:
            if isinstance(item, str) and item.strip().lower() in SIZE_PRESETS:
                resolved.extend(SIZE_PRESETS[item.strip().lower()])
            elif isinstance(item, str):
                resolved.append(parse_size(item))
            elif isinstance(item, dict) and "width" in item and "height" in item:
                resolved.append((int(item["width"]), int(item["height"])))
            else:
                raise ValueError(
                    f"Invalid entry in 'output_sizes': {item!r}. "
                    "Use a preset name, a 'WIDTHxHEIGHT' string, or {width, height}."
                )
        seen = set()
        output_sizes = []
        for size in resolved:
            if size not in seen:
                seen.add(size)
                output_sizes.append(size)

    fit_mode = data.get("fit_mode")
    if fit_mode is not None and fit_mode not in ("crop", "contain"):
        raise ValueError(f"'fit_mode' must be 'crop' or 'contain', got {fit_mode!r}")

    show_header = data.get("show_header")
    if show_header is not None and not isinstance(show_header, bool):
        raise ValueError(f"'show_header' must be true or false, got {show_header!r}")

    return CampaignBrief(
        name=data.get("name", "Untitled Campaign"),
        target_region=data["target_region"],
        target_audience=data["target_audience"],
        message=data["message"],
        products=products,
        brand=brand,
        language=data.get("language"),
        output_sizes=output_sizes,
        fit_mode=fit_mode,
        headline=data.get("headline"),
        show_header=show_header,
    )
