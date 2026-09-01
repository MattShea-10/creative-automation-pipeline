"""CLI entry point for the creative automation pipeline.

Usage:
    python -m src.main --brief briefs/sample_campaign.yaml
    python -m src.main --brief briefs/sample_campaign.yaml --provider mock
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv  # optional; not a hard requirement

    load_dotenv()
except ImportError:
    pass

from .brief_loader import load_brief
from .image_ops import parse_sizes
from .pipeline import CreativePipeline
from .providers import ALL_PROVIDER_NAMES, get_provider
from .storage import LocalAssetStore


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Creative automation pipeline for social ad campaigns.")
    parser.add_argument("--brief", required=True, help="Path to a campaign brief (YAML or JSON).")
    parser.add_argument(
        "--provider",
        choices=ALL_PROVIDER_NAMES,
        default=os.environ.get("IMAGE_PROVIDER", "pollinations"),
        help="GenAI image provider to use (default: %(default)s). "
        "Falls back to the offline mock provider automatically if the call fails.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable automatic fallback to the mock provider on API failure.",
    )
    parser.add_argument(
        "--sizes",
        default=None,
        help="A comma-separated list where each item is a preset name (default | web-top7 | "
        "broadcast) and/or an explicit WIDTHxHEIGHT pixel size -- freely mixable, e.g. "
        "'web-top7,broadcast' or 'default,1200x628'. 'web-top7' is 9 common web/display ad sizes; "
        "'broadcast' is TV/video delivery resolutions (1920x1080, 1280x720, 3840x2160). Duplicate "
        "pixel sizes across presets are only rendered once. Overrides any 'output_sizes' in the "
        "brief. Defaults to 1080x1080, 1080x1920, 1920x1080.",
    )
    parser.add_argument(
        "--fit",
        choices=["crop", "contain"],
        default=None,
        help="How to fit the hero image into each output size. 'crop' (default) fills the whole "
        "frame, cropping the longer dimension -- good for generic product photos. 'contain' scales "
        "the image down to fit entirely within the frame with no cropping, letterboxed with a "
        "blurred backdrop -- use this for a finished, already-composed creative (e.g. a flattened "
        "Photoshop export) where cropping would cut off text/logo/CTA. Overrides any 'fit_mode' in "
        "the brief.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Disable the header/title band at the top of each creative (shown by default). Each product's "
        "title defaults to its name, or a custom 'headline' set on the product or the whole brief. This flag "
        "always overrides the brief -- there's no equivalent flag to force it on, since it's already on by default.",
    )
    parser.add_argument("--assets-dir", default="assets", help="Folder to look for/reuse input assets in.")
    parser.add_argument("--output-dir", default="outputs", help="Folder to write generated creatives into.")
    parser.add_argument("--report", default=None, help="Path to write a JSON run report (default: <output-dir>/run_report.json).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose (DEBUG) logging.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("creative_pipeline")

    try:
        brief = load_brief(args.brief)
    except Exception as exc:
        logger.error("Failed to load campaign brief: %s", exc)
        return 1

    logger.info("Loaded campaign '%s' (%d products, region=%s)", brief.name, len(brief.products), brief.target_region)

    sizes = None
    if args.sizes:
        try:
            sizes = parse_sizes(args.sizes)
        except ValueError as exc:
            logger.error("Invalid --sizes value: %s", exc)
            return 1

    provider = get_provider(args.provider)
    store = LocalAssetStore(input_dir=args.assets_dir, cache_dir=str(Path(args.assets_dir) / "generated_cache"))
    pipeline = CreativePipeline(
        provider=provider,
        store=store,
        output_dir=args.output_dir,
        fallback_to_mock=not args.no_fallback,
        sizes=sizes,
        fit_mode=args.fit,
        no_header=args.no_header,
    )

    report = pipeline.run(brief)

    report_path = args.report or str(Path(args.output_dir) / "run_report.json")
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    print("\n=== Creative Automation Run Summary ===")
    print(f"Campaign:     {report.campaign_name}")
    print(f"Language:     {report.language} (translated: {report.was_translated})")
    print(f"Fit mode:     {report.fit_mode}")
    print(f"Header:       {'shown' if report.show_header else 'hidden'}")
    print(f"Creatives:    {len(report.creatives)}")
    print(f"Duration:     {report.duration_seconds:.1f}s")
    print(f"Report file:  {report_path}")
    if report.warnings:
        print(f"\nWarnings ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"  - {w}")
    print()
    for c in report.creatives:
        status = "OK" if c.compliance.passed else "FLAGGED"
        print(f"  [{status}] {c.product:<20} {c.size:<10} {c.name:<26} {c.source:<24} -> {c.output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
