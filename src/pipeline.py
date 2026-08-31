"""Pipeline orchestrator: brief -> hero images -> per-ratio creatives -> report."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from .compliance import ComplianceResult, run_compliance_checks
from .creative_render import VALID_FIT_MODES, render_creative
from .image_ops import (
    DEFAULT_SIZES,
    VIDEO_EXTENSIONS,
    device_category,
    open_as_rgb,
    ratio_label,
    size_label,
    size_name,
)
from .localization import infer_language, localize_message
from .models import CampaignBrief, Product
from .providers.base import ImageProvider, ImageProviderError
from .providers.mock_provider import MockImageProvider
from .storage import AssetStore

logger = logging.getLogger("creative_pipeline")


@dataclass
class CreativeResult:
    product: str
    size: str  # pixel dimensions, e.g. "1080x1080"
    ratio: str  # derived aspect ratio label, e.g. "1:1"
    name: str  # friendly name if this matches a known standard (e.g. "Leaderboard"), else same as ratio
    device: Optional[str]  # "mobile" | "desktop" for a recognized display ad size, else None
    output_path: str
    source: str  # "user-provided" | "cache" | "generated" | "generated (mock fallback)"
    compliance: ComplianceResult
    headline: Optional[str] = None  # the header/title text actually rendered, if the header band was shown


@dataclass
class PipelineReport:
    campaign_name: str
    language: str
    was_translated: bool
    localized_message: str
    fit_mode: str = "crop"
    show_header: bool = True
    creatives: List[CreativeResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self):
        return {
            "campaign_name": self.campaign_name,
            "language": self.language,
            "was_translated": self.was_translated,
            "localized_message": self.localized_message,
            "fit_mode": self.fit_mode,
            "show_header": self.show_header,
            "duration_seconds": round(self.duration_seconds, 2),
            "warnings": self.warnings,
            "creatives": [
                {
                    "product": c.product,
                    "size": c.size,
                    "ratio": c.ratio,
                    "name": c.name,
                    "device": c.device,
                    "output_path": c.output_path,
                    "source": c.source,
                    "headline": c.headline,
                    "compliance": {
                        "passed": c.compliance.passed,
                        "logo_present": c.compliance.logo_present,
                        "brand_color_match": c.compliance.brand_color_match,
                        "brand_color_distance": c.compliance.brand_color_distance,
                        "legal_flags": c.compliance.legal_flags,
                    },
                }
                for c in self.creatives
            ],
        }


class CreativePipeline:
    def __init__(
        self,
        provider: ImageProvider,
        store: AssetStore,
        output_dir: str = "outputs",
        fallback_to_mock: bool = True,
        sizes: Optional[List[Tuple[int, int]]] = None,
        fit_mode: Optional[str] = None,
        no_header: bool = False,
    ):
        self.provider = provider
        self.store = store
        self.output_dir = Path(output_dir)
        self.fallback_to_mock = fallback_to_mock
        # CLI-level overrides, if provided; otherwise resolved per-brief in
        # run() as: this value -> brief's setting -> built-in default.
        self.sizes_override = sizes
        if fit_mode is not None and fit_mode not in VALID_FIT_MODES:
            raise ValueError(f"fit_mode must be one of {VALID_FIT_MODES}, got {fit_mode!r}")
        self.fit_mode_override = fit_mode
        # --no-header always wins; there's no equivalent "force on" flag
        # since the header is already on by default.
        self.no_header = no_header
        self._mock = MockImageProvider()

    def _get_or_generate_hero(self, product: Product) -> tuple[Image.Image, str, Optional[Path]]:
        if product.asset_path:
            explicit_path = Path(product.asset_path)
            if explicit_path.exists():
                logger.info("Product '%s': using explicit asset_path %s", product.name, explicit_path)
                image = open_as_rgb(explicit_path, frame_seconds=product.video_frame_seconds)
                source = "user-provided (explicit asset_path)"
                if explicit_path.suffix.lower() in VIDEO_EXTENSIONS:
                    source += ", extracted video frame"
                return image, source, explicit_path
            logger.warning(
                "Product '%s': asset_path '%s' does not exist; falling back to convention-based lookup.",
                product.name,
                explicit_path,
            )

        cached = self.store.get_hero_image(product.slug)
        if cached:
            image, path = cached
            source = "user-provided" if Path(path).parent.name != "generated_cache" else "cache"
            if source == "user-provided" and Path(path).suffix.lower() in VIDEO_EXTENSIONS:
                source += " (extracted video frame)"
            logger.info("Product '%s': reusing existing asset (%s)", product.name, source)
            return image, source, Path(path)

        prompt = product.prompt_hint or f"professional studio product photo of {product.name}, clean background"
        logger.info("Product '%s': no existing asset found, generating via %s", product.name, self.provider.name)
        try:
            image = self.provider.generate(prompt)
            self.store.put_hero_image(product.slug, image)
            return image, "generated", None
        except ImageProviderError as exc:
            logger.warning("Provider '%s' failed (%s).", self.provider.name, exc)
            if not self.fallback_to_mock:
                raise
            logger.warning("Falling back to offline mock image for '%s'.", product.name)
            image = self._mock.generate(prompt)
            self.store.put_hero_image(product.slug, image)
            return image, "generated (mock fallback)", None

    def run(self, brief: CampaignBrief, prohibited_words: Optional[List[str]] = None) -> PipelineReport:
        start = time.time()
        language = infer_language(brief.target_region, brief.language)
        localized_message, was_translated = localize_message(brief.message, language)

        # Precedence: --fit CLI flag > brief's fit_mode > default "crop".
        fit_mode = self.fit_mode_override or brief.fit_mode or "crop"
        if fit_mode not in VALID_FIT_MODES:
            raise ValueError(f"Brief's fit_mode must be one of {VALID_FIT_MODES}, got {fit_mode!r}")

        # Precedence: --no-header CLI flag (always wins) > brief's show_header > default True.
        show_header = not self.no_header and brief.show_header is not False

        report = PipelineReport(
            campaign_name=brief.name,
            language=language,
            was_translated=was_translated,
            localized_message=localized_message,
            fit_mode=fit_mode,
            show_header=show_header,
        )

        # Precedence: --sizes CLI flag > brief's output_sizes > built-in defaults.
        sizes = self.sizes_override or brief.output_sizes or DEFAULT_SIZES
        if len(sizes) < 3:
            report.warnings.append(
                f"Only {len(sizes)} output size(s) configured; the brief asks for at least three."
            )

        brand_logo = None
        if brief.brand.logo_path and Path(brief.brand.logo_path).exists():
            brand_logo = Image.open(brief.brand.logo_path).convert("RGBA")
        elif brief.brand.logo_path:
            report.warnings.append(f"Brand logo path not found: {brief.brand.logo_path}")

        # Cache headline translations by source text so a headline shared
        # across products (e.g. a campaign-wide brief.headline) is only
        # translated once, not once per product.
        headline_translation_cache: dict = {}

        for product in brief.products:
            hero_image, source, hero_path = self._get_or_generate_hero(product)
            product_dir = self.output_dir / product.slug
            product_dir.mkdir(parents=True, exist_ok=True)

            product_logo = brand_logo

            headline_text = None
            if show_header:
                custom_headline = product.headline or brief.headline
                if custom_headline:
                    if custom_headline not in headline_translation_cache:
                        headline_translation_cache[custom_headline] = localize_message(custom_headline, language)
                    headline_text, _ = headline_translation_cache[custom_headline]
                else:
                    # No custom headline was set anywhere -- fall back to the
                    # product's own name. Product names are proper nouns, so
                    # (unlike a custom tagline) this is intentionally left
                    # untranslated.
                    headline_text = product.name

            for width, height in sizes:
                label = size_label(width, height)
                ratio = ratio_label(width, height)
                name = size_name(width, height)
                device = device_category(width, height)

                final_image, logo_composited = render_creative(
                    hero_image,
                    (width, height),
                    message=localized_message,
                    headline=headline_text if show_header else None,
                    fit_mode=fit_mode,
                    logo=product_logo,
                )

                filename = f"{product.slug}_{label}.png"
                # Recognized display ad sizes (web-top7) are further sorted
                # into mobile/ and desktop/ subfolders, since "mobile" vs.
                # "desktop" is a meaningful distinction for those units.
                # Social (1:1/9:16/16:9) and broadcast/video sizes aren't
                # classic "ad sizes" in that sense, so they stay directly
                # under the product folder as before.
                target_dir = product_dir / device if device else product_dir
                target_dir.mkdir(parents=True, exist_ok=True)
                out_path = target_dir / filename
                final_image.save(out_path)

                compliance = run_compliance_checks(
                    image=final_image,
                    message=localized_message,
                    brand_colors=brief.brand.colors,
                    logo_composited=logo_composited,
                    prohibited_words=prohibited_words,
                )
                if not compliance.passed:
                    report.warnings.append(
                        f"{product.name} ({label}) failed compliance checks: "
                        f"logo_present={compliance.logo_present}, "
                        f"brand_color_match={compliance.brand_color_match}, "
                        f"legal_flags={compliance.legal_flags}"
                    )

                report.creatives.append(
                    CreativeResult(
                        product=product.name,
                        size=label,
                        ratio=ratio,
                        name=name,
                        device=device,
                        output_path=str(out_path),
                        source=source,
                        compliance=compliance,
                        headline=headline_text if show_header else None,
                    )
                )
                logger.info("Saved %s (%s / %s) -> %s", product.name, label, name, out_path)

        report.duration_seconds = time.time() - start
        return report
