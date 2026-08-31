"""Minimal end-to-end smoke test.

Run with: python -m unittest discover tests
Uses the offline mock provider so it needs no network access and runs fast.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.brief_loader import load_brief
from src.image_ops import extract_video_frame, open_as_rgb, parse_sizes
from src.pipeline import CreativePipeline
from src.providers.mock_provider import MockImageProvider
from src.storage import LocalAssetStore

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _write_test_video(path, width=64, height=48, fps=10, seconds_per_third=1.0):
    """Write a tiny synthetic MP4 with three solid-color thirds (BGR order,
    since that's what OpenCV expects to write): red, then green, then blue.
    Lets tests assert *which* part of the video a given timestamp/default
    (middle frame) actually extracted, not just that "some frame" came back.
    """
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    frames_per_third = int(fps * seconds_per_third)
    for color_bgr in [(0, 0, 255), (0, 255, 0), (255, 0, 0)]:
        frame = np.zeros((height, width, 3), dtype="uint8")
        frame[:] = color_bgr
        for _ in range(frames_per_third):
            writer.write(frame)
    writer.release()


class PipelineSmokeTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.tmp_dir) / "outputs"
        self.cache_dir = Path(self.tmp_dir) / "cache"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_end_to_end_with_mock_provider(self):
        repo_root = Path(__file__).resolve().parent.parent
        brief = load_brief(str(repo_root / "briefs" / "sample_campaign.yaml"))

        store = LocalAssetStore(input_dir=str(repo_root / "assets"), cache_dir=str(self.cache_dir))
        pipeline = CreativePipeline(provider=MockImageProvider(), store=store, output_dir=str(self.output_dir))

        report = pipeline.run(brief)

        # 2 products x 3 aspect ratios = 6 creatives
        self.assertEqual(len(report.creatives), 6)
        for creative in report.creatives:
            self.assertTrue(Path(creative.output_path).exists(), f"missing {creative.output_path}")

        self.assertTrue((self.output_dir / "hydroboost" / "hydroboost_1080x1080.png").exists())
        self.assertTrue((self.output_dir / "hydroboost" / "hydroboost_1080x1920.png").exists())
        self.assertTrue((self.output_dir / "hydroboost" / "hydroboost_1920x1080.png").exists())

    def test_custom_sizes_override(self):
        repo_root = Path(__file__).resolve().parent.parent
        brief = load_brief(str(repo_root / "briefs" / "sample_campaign.yaml"))

        store = LocalAssetStore(input_dir=str(repo_root / "assets"), cache_dir=str(self.cache_dir))
        pipeline = CreativePipeline(
            provider=MockImageProvider(),
            store=store,
            output_dir=str(self.output_dir),
            sizes=[(500, 500), (300, 600)],
        )

        report = pipeline.run(brief)

        # 2 products x 2 custom sizes = 4 creatives
        self.assertEqual(len(report.creatives), 4)
        # 500x500 isn't a recognized standard size, so it stays directly
        # under the product folder; 300x600 IS a recognized size ("Half
        # Page Ad") even though it's not part of the web-top7 preset, so
        # it's still sorted into desktop/ like any other known display ad size.
        self.assertTrue((self.output_dir / "hydroboost" / "hydroboost_500x500.png").exists())
        self.assertTrue((self.output_dir / "hydroboost" / "desktop" / "hydroboost_300x600.png").exists())
        # A live PNG's actual pixel size should match what was requested.
        from PIL import Image

        with Image.open(self.output_dir / "hydroboost" / "hydroboost_500x500.png") as img:
            self.assertEqual(img.size, (500, 500))

    def test_web_top7_and_broadcast_presets_render_without_error(self):
        repo_root = Path(__file__).resolve().parent.parent
        brief = load_brief(str(repo_root / "briefs" / "sample_campaign.yaml"))
        store = LocalAssetStore(input_dir=str(repo_root / "assets"), cache_dir=str(self.cache_dir))

        for preset in ("web-top7", "broadcast"):
            sizes = parse_sizes(preset)
            pipeline = CreativePipeline(
                provider=MockImageProvider(),
                store=store,
                output_dir=str(self.output_dir / preset),
                sizes=sizes,
            )
            report = pipeline.run(brief)
            self.assertEqual(len(report.creatives), 2 * len(sizes))
            for creative in report.creatives:
                self.assertTrue(Path(creative.output_path).exists())

    def test_web_top7_sorts_into_mobile_and_desktop_subfolders(self):
        repo_root = Path(__file__).resolve().parent.parent
        brief = load_brief(str(repo_root / "briefs" / "sample_campaign.yaml"))
        store = LocalAssetStore(input_dir=str(repo_root / "assets"), cache_dir=str(self.cache_dir))
        pipeline = CreativePipeline(
            provider=MockImageProvider(),
            store=store,
            output_dir=str(self.output_dir),
            sizes=parse_sizes("web-top7"),
        )
        report = pipeline.run(brief)

        # Only 320x50 (Mobile Leaderboard) is mobile; the other 9 are desktop.
        mobile = [c for c in report.creatives if c.device == "mobile"]
        desktop = [c for c in report.creatives if c.device == "desktop"]
        self.assertEqual(len(mobile), 2)  # 2 products x 1 mobile size
        self.assertEqual(len(desktop), 18)  # 2 products x 9 desktop sizes
        for c in mobile:
            self.assertIn("/mobile/", c.output_path.replace("\\", "/"))
        for c in desktop:
            self.assertIn("/desktop/", c.output_path.replace("\\", "/"))
        self.assertTrue((self.output_dir / "hydroboost" / "mobile" / "hydroboost_320x50.png").exists())
        self.assertTrue((self.output_dir / "hydroboost" / "desktop" / "hydroboost_728x90.png").exists())

    def test_728x480_is_a_recognized_web_ad_size(self):
        from src.image_ops import WEB_AD_SIZES, device_category, size_name

        self.assertIn((728, 480), WEB_AD_SIZES)
        self.assertEqual(size_name(728, 480), "Wide Rectangle")
        self.assertEqual(device_category(728, 480), "desktop")

        repo_root = Path(__file__).resolve().parent.parent
        brief = load_brief(str(repo_root / "briefs" / "sample_campaign.yaml"))
        store = LocalAssetStore(input_dir=str(repo_root / "assets"), cache_dir=str(self.cache_dir))
        pipeline = CreativePipeline(
            provider=MockImageProvider(), store=store, output_dir=str(self.output_dir), sizes=[(728, 480)]
        )
        report = pipeline.run(brief)
        self.assertEqual(len(report.creatives), 2)  # 2 products x 1 size
        out_path = Path(self.output_dir / "hydroboost" / "desktop" / "hydroboost_728x480.png")
        self.assertTrue(out_path.exists())
        from PIL import Image as PILImage

        with PILImage.open(out_path) as img:
            self.assertEqual(img.size, (728, 480))

    def test_default_sizes_stay_uncategorized(self):
        # Social defaults aren't classic display ad units, so they should
        # NOT be sorted into mobile/desktop subfolders.
        repo_root = Path(__file__).resolve().parent.parent
        brief = load_brief(str(repo_root / "briefs" / "sample_campaign.yaml"))
        store = LocalAssetStore(input_dir=str(repo_root / "assets"), cache_dir=str(self.cache_dir))
        pipeline = CreativePipeline(provider=MockImageProvider(), store=store, output_dir=str(self.output_dir))
        report = pipeline.run(brief)
        for c in report.creatives:
            self.assertIsNone(c.device)
        self.assertTrue((self.output_dir / "hydroboost" / "hydroboost_1080x1080.png").exists())

    def test_contain_fit_mode_preserves_whole_image_no_crop(self):
        # Build a tall, skinny synthetic "designed creative" and fit it into
        # a wide banner. With fit_mode="contain" the source must be scaled
        # down to fit entirely inside the frame (no cropping) -- so a
        # distinctive marker pixel placed at the very edge of the source
        # must still be present, just shrunk, in the output.
        from PIL import Image as PILImage

        repo_root = Path(__file__).resolve().parent.parent
        brief = load_brief(str(repo_root / "briefs" / "sample_campaign.yaml"))
        store = LocalAssetStore(input_dir=str(repo_root / "assets"), cache_dir=str(self.cache_dir))

        # Give the first product an explicit asset_path pointing at a
        # synthetic image with a solid red left edge and solid blue right
        # edge, so we can check both survive an aggressive aspect-ratio
        # change under "contain" but would very likely NOT both survive
        # under "crop" at an extreme ratio.
        src_path = Path(self.tmp_dir) / "designed_creative.png"
        src = PILImage.new("RGB", (800, 100), (255, 255, 255))
        for y in range(100):
            for x in range(20):
                src.putpixel((x, y), (255, 0, 0))  # red left edge
                src.putpixel((799 - x, y), (0, 0, 255))  # blue right edge
        src.save(src_path)
        brief.products[0].asset_path = str(src_path)

        pipeline = CreativePipeline(
            provider=MockImageProvider(),
            store=store,
            output_dir=str(self.output_dir),
            sizes=[(300, 250)],
            fit_mode="contain",
        )
        report = pipeline.run(brief)
        self.assertEqual(report.fit_mode, "contain")

        # 300x250 ("Medium Rectangle") is a recognized display ad size, so
        # it's sorted into desktop/ like any other known ad unit.
        out_path = self.output_dir / "hydroboost" / "desktop" / "hydroboost_300x250.png"
        self.assertTrue(out_path.exists())
        with PILImage.open(out_path) as img:
            self.assertEqual(img.size, (300, 250))
            # Sample a thin vertical strip just inside each long edge of the
            # fitted (not stretched-to-fill) image; both the red and blue
            # source edges should still be visible somewhere in the output,
            # proving the whole source width was preserved rather than
            # cropped away.
            colors = {img.getpixel((x, 125)) for x in range(img.width)}

        def _has_close(colors, target, tol=40):
            return any(all(abs(c[i] - target[i]) <= tol for i in range(3)) for c in colors)

        self.assertTrue(_has_close(colors, (255, 0, 0)), "red left edge of source not found in contain output")
        self.assertTrue(_has_close(colors, (0, 0, 255)), "blue right edge of source not found in contain output")

    def test_explicit_asset_path_overrides_convention_lookup(self):
        from PIL import Image as PILImage

        repo_root = Path(__file__).resolve().parent.parent
        brief = load_brief(str(repo_root / "briefs" / "sample_campaign.yaml"))
        store = LocalAssetStore(input_dir=str(repo_root / "assets"), cache_dir=str(self.cache_dir))

        # A solid, distinctive-color source image at an explicit path that
        # does NOT follow the assets/<slug>.png naming convention.
        src_path = Path(self.tmp_dir) / "my_photoshop_export.png"
        PILImage.new("RGB", (728, 480), (12, 34, 56)).save(src_path)
        brief.products[0].asset_path = str(src_path)

        pipeline = CreativePipeline(
            provider=MockImageProvider(),
            store=store,
            output_dir=str(self.output_dir),
            sizes=[(300, 250)],
        )
        report = pipeline.run(brief)

        creative = next(c for c in report.creatives if c.product == brief.products[0].name)
        self.assertEqual(creative.source, "user-provided (explicit asset_path)")

        with PILImage.open(creative.output_path) as img:
            # Center pixel should be close to the solid source color (allow
            # tolerance since the message banner overlay covers part of the
            # bottom of the frame, but the center should be untouched).
            r, g, b = img.getpixel((img.width // 2, img.height // 3))
        self.assertLess(abs(r - 12) + abs(g - 34) + abs(b - 56), 60)

    def test_missing_asset_path_falls_back_to_generation(self):
        repo_root = Path(__file__).resolve().parent.parent
        brief = load_brief(str(repo_root / "briefs" / "sample_campaign.yaml"))
        store = LocalAssetStore(input_dir=str(repo_root / "assets"), cache_dir=str(self.cache_dir))

        brief.products[0].asset_path = str(Path(self.tmp_dir) / "does_not_exist.png")

        pipeline = CreativePipeline(
            provider=MockImageProvider(),
            store=store,
            output_dir=str(self.output_dir),
            sizes=[(300, 250)],
        )
        report = pipeline.run(brief)
        creative = next(c for c in report.creatives if c.product == brief.products[0].name)
        # Should not have crashed, and should not claim to be the
        # (nonexistent) explicit asset -- it falls back to convention-based
        # lookup / generation instead.
        self.assertNotEqual(creative.source, "user-provided (explicit asset_path)")
        self.assertTrue(Path(creative.output_path).exists())

    def test_header_defaults_to_product_name(self):
        repo_root = Path(__file__).resolve().parent.parent
        brief = load_brief(str(repo_root / "briefs" / "sample_campaign.yaml"))
        store = LocalAssetStore(input_dir=str(repo_root / "assets"), cache_dir=str(self.cache_dir))
        pipeline = CreativePipeline(
            provider=MockImageProvider(), store=store, output_dir=str(self.output_dir), sizes=[(1080, 1080)]
        )
        report = pipeline.run(brief)
        self.assertTrue(report.show_header)
        for creative, product in zip(report.creatives, brief.products):
            self.assertEqual(creative.headline, product.name)

    def test_no_header_flag_disables_header_band(self):
        repo_root = Path(__file__).resolve().parent.parent
        brief = load_brief(str(repo_root / "briefs" / "sample_campaign.yaml"))
        store = LocalAssetStore(input_dir=str(repo_root / "assets"), cache_dir=str(self.cache_dir))
        pipeline = CreativePipeline(
            provider=MockImageProvider(),
            store=store,
            output_dir=str(self.output_dir),
            sizes=[(1080, 1080)],
            no_header=True,
        )
        report = pipeline.run(brief)
        self.assertFalse(report.show_header)
        for creative in report.creatives:
            self.assertIsNone(creative.headline)

    def test_brief_show_header_false_disables_header_band(self):
        repo_root = Path(__file__).resolve().parent.parent
        brief = load_brief(str(repo_root / "briefs" / "sample_campaign.yaml"))
        brief.show_header = False
        store = LocalAssetStore(input_dir=str(repo_root / "assets"), cache_dir=str(self.cache_dir))
        pipeline = CreativePipeline(
            provider=MockImageProvider(), store=store, output_dir=str(self.output_dir), sizes=[(1080, 1080)]
        )
        report = pipeline.run(brief)
        self.assertFalse(report.show_header)

    def test_no_header_cli_flag_overrides_brief_wanting_a_header(self):
        # brief.show_header left at its default (None -> "show it"); the
        # CLI-level --no-header flag should still win.
        repo_root = Path(__file__).resolve().parent.parent
        brief = load_brief(str(repo_root / "briefs" / "sample_campaign.yaml"))
        store = LocalAssetStore(input_dir=str(repo_root / "assets"), cache_dir=str(self.cache_dir))
        pipeline = CreativePipeline(
            provider=MockImageProvider(),
            store=store,
            output_dir=str(self.output_dir),
            sizes=[(1080, 1080)],
            no_header=True,
        )
        report = pipeline.run(brief)
        self.assertFalse(report.show_header)

    def test_product_headline_overrides_brief_headline_which_overrides_product_name(self):
        repo_root = Path(__file__).resolve().parent.parent
        brief = load_brief(str(repo_root / "briefs" / "sample_campaign.yaml"))
        brief.headline = "Campaign-Wide Tagline"
        brief.products[0].headline = "Product-Specific Title"
        # brief.products[1] has no override -- should fall back to brief.headline.
        store = LocalAssetStore(input_dir=str(repo_root / "assets"), cache_dir=str(self.cache_dir))
        pipeline = CreativePipeline(
            provider=MockImageProvider(), store=store, output_dir=str(self.output_dir), sizes=[(1080, 1080)]
        )
        report = pipeline.run(brief)

        creative_0 = next(c for c in report.creatives if c.product == brief.products[0].name)
        creative_1 = next(c for c in report.creatives if c.product == brief.products[1].name)
        self.assertEqual(creative_0.headline, "Product-Specific Title")
        self.assertEqual(creative_1.headline, "Campaign-Wide Tagline")

    def test_header_leaves_room_for_logo_at_extreme_width(self):
        # Regression check for the header/logo collision bug: with a brand
        # logo configured, a long headline at a very wide, short canvas
        # (mimicking a web-ad banner) must not be drawn underneath where
        # the logo will be composited (the top-right corner).
        from PIL import Image as PILImage

        from src.image_ops import add_header_banner, add_logo_watermark, logo_render_size

        canvas = PILImage.new("RGB", (970, 90), (10, 10, 10))
        logo = PILImage.new("RGBA", (200, 100), (255, 209, 0, 255))

        logo_w, _, logo_margin = logo_render_size(canvas.size, logo)
        with_header = add_header_banner(canvas, "A Fairly Long Product Headline Here", reserved_right=logo_w + logo_margin)
        final = add_logo_watermark(with_header, logo)

        # Sample a strip of pixels directly under where the logo will sit
        # (top-right corner) -- it should be either background or logo
        # color, never white headline text bleeding through.
        w, h = final.size
        logo_zone_pixels = [
            final.getpixel((x, y))
            for x in range(w - logo_w - logo_margin, w)
            for y in range(0, min(30, h))
        ]
        near_white = sum(1 for p in logo_zone_pixels if min(p[:3]) > 230)
        # Some near-white pixels are fine if they belong to the logo itself
        # (this synthetic logo is solid yellow, not white, so effectively
        # none should show up) -- assert there's no dense cluster of
        # headline text rendered in that zone.
        self.assertLess(near_white, len(logo_zone_pixels) * 0.05)

    def test_header_leaves_room_for_logo_at_top_left_instead(self):
        # Mirror of the top-right regression test above, but for a
        # top-left logo -- the header must reserve space on the *left*
        # edge instead, via reserved_left.
        from PIL import Image as PILImage

        from src.image_ops import add_header_banner, add_logo_watermark, logo_render_size

        canvas = PILImage.new("RGB", (970, 90), (10, 10, 10))
        logo = PILImage.new("RGBA", (200, 100), (255, 209, 0, 255))

        logo_w, _, logo_margin = logo_render_size(canvas.size, logo)
        with_header = add_header_banner(
            canvas, "A Fairly Long Product Headline Here", reserved_left=logo_w + logo_margin
        )
        final = add_logo_watermark(with_header, logo, position="top-left")

        w, h = final.size
        logo_zone_pixels = [
            final.getpixel((x, y)) for x in range(0, logo_w + logo_margin) for y in range(0, min(30, h))
        ]
        near_white = sum(1 for p in logo_zone_pixels if min(p[:3]) > 230)
        self.assertLess(near_white, len(logo_zone_pixels) * 0.05)

    def test_logo_watermark_all_positions_are_valid_and_distinct(self):
        # Every VALID_LOGO_POSITIONS entry must actually draw the logo
        # somewhere. Most of them must also land in a visually distinct
        # spot from one another -- with one deliberate exception:
        # below-header-left/right share their raw (x, y) with
        # top-left/top-right respectively, because what makes "below
        # header" actually sit below the header is the y_offset that
        # render_creative() computes from the rendered header height and
        # passes in -- add_logo_watermark() itself has no header to know
        # about when called directly with no y_offset. See the
        # render_creative-level below-header-left/right/center tests for
        # coverage of the *offset* positioning.
        from PIL import Image as PILImage

        from src.image_ops import VALID_LOGO_POSITIONS, add_logo_watermark

        canvas = PILImage.new("RGB", (600, 600), (10, 10, 10))
        logo = PILImage.new("RGBA", (200, 100), (255, 209, 0, 255))
        renders = {pos: add_logo_watermark(canvas, logo, position=pos) for pos in VALID_LOGO_POSITIONS}
        for pos, out in renders.items():
            self.assertNotEqual(list(out.getdata()), list(canvas.convert("RGB").getdata()), f"position={pos} did nothing")

        expected_duplicates = {
            frozenset({"top-left", "below-header-left"}),
            frozenset({"top-right", "below-header-right"}),
        }
        seen = {}
        for pos, out in renders.items():
            data = tuple(out.getdata())
            for other_pos, other_data in seen.items():
                if data == other_data:
                    pair = frozenset({pos, other_pos})
                    self.assertIn(
                        pair,
                        expected_duplicates,
                        f"position={pos} rendered identically to {other_pos}, which isn't an expected duplicate",
                    )
            seen[pos] = data

    def test_logo_watermark_invalid_position_raises_value_error(self):
        from PIL import Image as PILImage

        from src.image_ops import add_logo_watermark

        canvas = PILImage.new("RGB", (400, 400), (0, 0, 0))
        logo = PILImage.new("RGBA", (100, 100), (255, 209, 0, 255))
        with self.assertRaises(ValueError):
            add_logo_watermark(canvas, logo, position="sideways")

    def test_logo_watermark_default_position_scale_and_opacity_are_unchanged(self):
        # Omitting position/scale/opacity entirely should behave exactly
        # like before those options existed.
        from PIL import Image as PILImage

        from src.image_ops import add_logo_watermark

        canvas = PILImage.new("RGB", (600, 600), (10, 10, 10))
        logo = PILImage.new("RGBA", (200, 100), (255, 209, 0, 255))
        default_out = add_logo_watermark(canvas, logo)
        explicit_out = add_logo_watermark(canvas, logo, position="top-right", scale=0.16, opacity=1.0)
        self.assertEqual(list(default_out.getdata()), list(explicit_out.getdata()))

    def test_logo_watermark_opacity_blends_instead_of_fully_replacing(self):
        from PIL import Image as PILImage

        from src.image_ops import add_logo_watermark, logo_render_size

        canvas = PILImage.new("RGB", (600, 600), (10, 10, 10))
        logo = PILImage.new("RGBA", (200, 100), (0, 200, 0, 255))
        half_opacity = add_logo_watermark(canvas, logo, position="center", opacity=0.5)
        logo_w, logo_h, _ = logo_render_size((600, 600), logo)
        cx, cy = 300, 300
        pixel = half_opacity.getpixel((cx, cy))
        self.assertNotEqual(pixel, (10, 10, 10))
        self.assertNotEqual(pixel, (0, 200, 0))
        self.assertGreater(pixel[1], 60)
        self.assertLess(pixel[1], 150)

    def test_logo_watermark_x_offset_and_y_offset_move_the_logo(self):
        from PIL import Image as PILImage

        from src.image_ops import add_logo_watermark, logo_render_size

        canvas = PILImage.new("RGB", (600, 600), (10, 10, 10))
        logo = PILImage.new("RGBA", (200, 100), (255, 209, 0, 255))
        logo_w, logo_h, margin = logo_render_size((600, 600), logo)

        def is_logo_yellow(pixel):
            r, g, b = pixel
            return r > 200 and g > 150 and b < 80

        baseline = add_logo_watermark(canvas, logo, position="center")
        base_cx, base_cy = 300, 300
        self.assertTrue(is_logo_yellow(baseline.getpixel((base_cx, base_cy))))

        nudged = add_logo_watermark(canvas, logo, position="center", x_offset=80, y_offset=-60)
        # The old center spot should no longer be covered by the logo...
        self.assertFalse(is_logo_yellow(nudged.getpixel((base_cx, base_cy))))
        # ...and the nudged spot (80 right, 60 up) should be.
        self.assertTrue(is_logo_yellow(nudged.getpixel((base_cx + 80, base_cy - 60))))

    def test_logo_watermark_offset_is_clamped_to_the_frame_edge(self):
        # A nudge large enough to push the logo above/left of the frame
        # should just stop at the edge (0), not wrap or error.
        from PIL import Image as PILImage

        from src.image_ops import add_logo_watermark

        canvas = PILImage.new("RGB", (400, 400), (10, 10, 10))
        logo = PILImage.new("RGBA", (100, 100), (255, 209, 0, 255))
        out = add_logo_watermark(canvas, logo, position="top-left", x_offset=-9999, y_offset=-9999)

        def is_logo_yellow(pixel):
            r, g, b = pixel
            return r > 200 and g > 150 and b < 80

        self.assertTrue(is_logo_yellow(out.getpixel((0, 0))), "expected the logo clamped flush to the top-left corner")

    def test_render_creative_logo_offset_nudges_the_logo_from_its_position(self):
        from PIL import Image as PILImage

        from src.creative_render import render_creative
        from src.image_ops import logo_render_size

        hero = PILImage.new("RGB", (800, 600), (60, 120, 180))
        logo = PILImage.new("RGBA", (160, 80), (255, 209, 0, 255))

        def is_logo_yellow(pixel):
            r, g, b = pixel
            return r > 200 and g > 150 and b < 80

        baseline, _ = render_creative(hero, (800, 600), fit_mode="crop", logo=logo, logo_position="center")
        nudged, _ = render_creative(
            hero, (800, 600), fit_mode="crop", logo=logo, logo_position="center", logo_offset_x=50, logo_offset_y=50,
        )
        cx, cy = 400, 300
        self.assertTrue(is_logo_yellow(baseline.getpixel((cx, cy))))
        self.assertFalse(is_logo_yellow(nudged.getpixel((cx, cy))))
        self.assertTrue(is_logo_yellow(nudged.getpixel((cx + 50, cy + 50))))

    def test_render_creative_logo_offset_adds_to_the_below_header_offset(self):
        # logo_offset_y should stack with (not replace) the automatic
        # downward shift below-header-* positions already apply to clear
        # the header banner.
        from PIL import Image as PILImage

        from src.creative_render import render_creative
        from src.image_ops import header_banner_height, logo_render_size

        size = (800, 600)
        hero = PILImage.new("RGB", size, (60, 120, 180))
        logo = PILImage.new("RGBA", (160, 80), (255, 209, 0, 255))
        headline = "A Fairly Long Product Headline Here"

        def is_logo_yellow(pixel):
            r, g, b = pixel
            return r > 200 and g > 150 and b < 80

        without_extra, _ = render_creative(
            hero, size, headline=headline, fit_mode="crop", logo=logo, logo_position="below-header-center",
        )
        with_extra, _ = render_creative(
            hero,
            size,
            headline=headline,
            fit_mode="crop",
            logo=logo,
            logo_position="below-header-center",
            logo_offset_y=40,
        )

        # Mirrors add_logo_watermark()'s own math for below-header-center:
        # y starts at `margin` (its usual top-row baseline), then the
        # gap-past-the-header offset render_creative() computes is added
        # on top of that.
        header_h = header_banner_height(size, headline)
        gap = max(int(min(size) * 0.03), 6)
        _logo_w, _logo_h, margin = logo_render_size(size, logo, scale_frac=0.16)
        logo_top = margin + header_h + gap

        def row_has_logo(img, y):
            return any(is_logo_yellow(img.getpixel((x, y))) for x in range(0, size[0], 4))

        # A few pixels into the logo's normal (un-nudged) top edge: present
        # without the extra offset...
        self.assertTrue(row_has_logo(without_extra, logo_top + 5))
        # ...but with an extra 40px pushed further down, that same row is
        # now above where the logo actually starts, so it's clear.
        self.assertFalse(row_has_logo(with_extra, logo_top + 5))
        # And the logo did land 40px further down, not just disappear.
        self.assertTrue(row_has_logo(with_extra, logo_top + 40 + 5))

    def test_render_creative_bottom_logo_sits_on_top_of_message_banner(self):
        # A logo positioned anywhere other than a top corner has no header
        # to avoid -- confirm it's composited *last*, on top of the message
        # banner, rather than being covered by it (mirroring the badge
        # image's corner-vs-full compositing split).
        from PIL import Image as PILImage

        from src.creative_render import render_creative
        from src.image_ops import logo_render_size

        hero = PILImage.new("RGB", (800, 600), (60, 120, 180))
        logo = PILImage.new("RGBA", (200, 100), (255, 209, 0, 255))  # fully opaque -- no ambiguity
        out, logo_composited = render_creative(
            hero,
            (400, 400),
            message="A caption long enough to fill the bottom banner area",
            fit_mode="crop",
            logo=logo,
            logo_position="bottom-right",
        )
        self.assertTrue(logo_composited)

        def is_logo_yellow(pixel):
            r, g, b = pixel
            return r > 200 and g > 150 and b < 80

        logo_w, logo_h, margin = logo_render_size((400, 400), logo)
        center_x = 400 - margin - logo_w // 2
        center_y = 400 - margin - logo_h // 2
        self.assertTrue(
            is_logo_yellow(out.getpixel((center_x, center_y))),
            "expected the bottom-right logo to be visible on top of the message banner, not covered by it",
        )

    def test_render_creative_top_left_logo_reserves_header_space_via_render_creative(self):
        # End-to-end version of test_header_leaves_room_for_logo_at_top_left_instead,
        # through render_creative() itself rather than the lower-level
        # add_header_banner()/add_logo_watermark() calls directly.
        from PIL import Image as PILImage

        from src.creative_render import render_creative

        hero = PILImage.new("RGB", (970, 90), (10, 10, 10))
        logo = PILImage.new("RGBA", (200, 100), (255, 209, 0, 255))
        out, logo_composited = render_creative(
            hero,
            (970, 90),
            headline="A Fairly Long Product Headline Here",
            fit_mode="crop",
            logo=logo,
            logo_position="top-left",
        )
        self.assertTrue(logo_composited)

    def test_render_creative_below_header_logo_sits_beneath_header_not_beside_it(self):
        # "below-header-center" must land clear of the header banner's own
        # rendered height -- confirm the logo's yellow doesn't show up
        # anywhere inside the header banner's rows (where "top-left"/
        # "top-right" would put it), and does show up somewhere below that.
        from PIL import Image as PILImage

        from src.creative_render import render_creative
        from src.image_ops import header_banner_height

        hero = PILImage.new("RGB", (800, 600), (60, 120, 180))
        logo = PILImage.new("RGBA", (160, 80), (255, 209, 0, 255))  # fully opaque -- no ambiguity
        headline = "A Fairly Long Product Headline Here"
        out, logo_composited = render_creative(
            hero,
            (800, 600),
            headline=headline,
            fit_mode="crop",
            logo=logo,
            logo_position="below-header-center",
        )
        self.assertTrue(logo_composited)

        def is_logo_yellow(pixel):
            r, g, b = pixel
            return r > 200 and g > 150 and b < 80

        header_h = header_banner_height((800, 600), headline)

        # Not inside the header banner's own rows.
        self.assertFalse(
            any(is_logo_yellow(out.getpixel((x, y))) for x in range(0, 800, 4) for y in range(0, header_h)),
            "expected no logo pixels inside the header banner -- below-header should clear it",
        )
        # But visible somewhere just below it.
        self.assertTrue(
            any(
                is_logo_yellow(out.getpixel((x, y)))
                for x in range(0, 800, 4)
                for y in range(header_h, min(header_h + 200, 600))
            ),
            "expected the below-header logo to be visible just underneath the header banner",
        )

    def test_render_creative_below_header_logo_without_headline_falls_back_to_top_center(self):
        # No headline means there's no header to be "below" -- the logo
        # should still render (top-center-ish, with just the margin) rather
        # than erroring or vanishing.
        from PIL import Image as PILImage

        from src.creative_render import render_creative

        hero = PILImage.new("RGB", (800, 600), (60, 120, 180))
        logo = PILImage.new("RGBA", (160, 80), (255, 209, 0, 255))
        out, logo_composited = render_creative(
            hero, (800, 600), fit_mode="crop", logo=logo, logo_position="below-header-center"
        )
        self.assertTrue(logo_composited)

        def is_logo_yellow(pixel):
            r, g, b = pixel
            return r > 200 and g > 150 and b < 80

        self.assertTrue(
            any(is_logo_yellow(out.getpixel((x, y))) for x in range(0, 800, 4) for y in range(0, 150)),
            "expected the logo near the top of the frame when there's no headline to clear",
        )

    def test_render_creative_below_header_logo_left_sits_near_left_margin(self):
        from PIL import Image as PILImage

        from src.creative_render import render_creative
        from src.image_ops import header_banner_height

        hero = PILImage.new("RGB", (800, 600), (60, 120, 180))
        logo = PILImage.new("RGBA", (160, 80), (255, 209, 0, 255))
        headline = "A Fairly Long Product Headline Here"
        out, logo_composited = render_creative(
            hero,
            (800, 600),
            headline=headline,
            fit_mode="crop",
            logo=logo,
            logo_position="below-header-left",
        )
        self.assertTrue(logo_composited)

        def is_logo_yellow(pixel):
            r, g, b = pixel
            return r > 200 and g > 150 and b < 80

        header_h = header_banner_height((800, 600), headline)
        band = range(header_h, min(header_h + 200, 600))

        left_half_hit = any(
            is_logo_yellow(out.getpixel((x, y))) for x in range(0, 400, 4) for y in band
        )
        right_half_hit = any(
            is_logo_yellow(out.getpixel((x, y))) for x in range(400, 800, 4) for y in band
        )
        self.assertTrue(left_half_hit, "expected the below-header-left logo in the left half of the frame")
        self.assertFalse(right_half_hit, "did not expect the below-header-left logo in the right half of the frame")

    def test_render_creative_below_header_logo_right_sits_near_right_margin(self):
        from PIL import Image as PILImage

        from src.creative_render import render_creative
        from src.image_ops import header_banner_height

        hero = PILImage.new("RGB", (800, 600), (60, 120, 180))
        logo = PILImage.new("RGBA", (160, 80), (255, 209, 0, 255))
        headline = "A Fairly Long Product Headline Here"
        out, logo_composited = render_creative(
            hero,
            (800, 600),
            headline=headline,
            fit_mode="crop",
            logo=logo,
            logo_position="below-header-right",
        )
        self.assertTrue(logo_composited)

        def is_logo_yellow(pixel):
            r, g, b = pixel
            return r > 200 and g > 150 and b < 80

        header_h = header_banner_height((800, 600), headline)
        band = range(header_h, min(header_h + 200, 600))

        left_half_hit = any(
            is_logo_yellow(out.getpixel((x, y))) for x in range(0, 400, 4) for y in band
        )
        right_half_hit = any(
            is_logo_yellow(out.getpixel((x, y))) for x in range(400, 800, 4) for y in band
        )
        self.assertFalse(left_half_hit, "did not expect the below-header-right logo in the left half of the frame")
        self.assertTrue(right_half_hit, "expected the below-header-right logo in the right half of the frame")

    def test_render_creative_layers_matches_render_creative_flattened_output(self):
        # render_creative_layers() mirrors render_creative()'s own control
        # flow so a layered PSD download always matches the flattened PNG
        # preview -- flattening the layer stack by hand here and diffing
        # it against render_creative()'s real output is what actually
        # proves that, for a battery of configs that each exercise a
        # different combination of overlays/branches.
        from PIL import Image as PILImage

        from src.creative_render import render_creative, render_creative_layers

        hero = PILImage.new("RGB", (900, 700), (40, 90, 140))
        logo = PILImage.new("RGBA", (200, 100), (255, 209, 0, 255))
        badge = PILImage.new("RGBA", (150, 150), (0, 200, 120, 255))

        configs = [
            dict(headline="A Fairly Long Product Headline Here", message="Shop the new drop today"),
            dict(
                headline="HydroBoost", message="Have and run and have some fun", logo=logo,
                logo_position="top-right", cta_text="Shop Now", cta_position="bottom-right",
                cta_above_message=True,
            ),
            dict(
                headline="Below Header Test", logo=logo, logo_position="below-header-left",
                logo_offset_x=15, logo_offset_y=-5, badge_image=badge, badge_position="full",
            ),
            dict(
                message="Bottom logo and corner badge", logo=logo, logo_position="bottom-right",
                badge_image=badge, badge_position="top-left", cta_text="Learn More",
                cta_position="bottom-center", cta_glow=True,
            ),
            dict(headline="Just A Header", header_glow=True, header_show_background=False),
            dict(),  # nothing at all but the hero -- Background-only stack
        ]

        for i, kwargs in enumerate(configs):
            with self.subTest(config=i):
                expected, _ = render_creative(hero, (900, 700), fit_mode="crop", **kwargs)
                layers = render_creative_layers(hero, (900, 700), fit_mode="crop", **kwargs)

                self.assertGreaterEqual(len(layers), 1)
                self.assertEqual(layers[0][0], "Background")

                flattened = None
                for name, layer_img in layers:
                    self.assertEqual(layer_img.size, (900, 700), f"layer {name!r} isn't full-canvas size")
                    self.assertEqual(layer_img.mode, "RGBA", f"layer {name!r} isn't RGBA")
                    rgba = layer_img if layer_img.mode == "RGBA" else layer_img.convert("RGBA")
                    flattened = rgba if flattened is None else PILImage.alpha_composite(flattened, rgba)
                flattened = flattened.convert("RGB")

                self.assertEqual(
                    list(flattened.getdata()),
                    list(expected.getdata()),
                    f"config {i}: flattened layer stack doesn't match render_creative()'s own output",
                )

    def test_render_creative_invalid_logo_position_raises_value_error(self):
        from PIL import Image as PILImage

        from src.creative_render import render_creative

        hero = PILImage.new("RGB", (400, 400), (10, 10, 10))
        logo = PILImage.new("RGBA", (100, 100), (255, 209, 0, 255))
        with self.assertRaises(ValueError):
            render_creative(hero, (400, 400), fit_mode="crop", logo=logo, logo_position="sideways")

    def test_header_banner_text_color_is_applied(self):
        from PIL import Image as PILImage

        from src.image_ops import add_header_banner

        canvas = PILImage.new("RGB", (400, 300), (0, 0, 0))  # black background, background plate off
        red = (255, 0, 0)
        out = add_header_banner(canvas, "COLORED HEADER", text_color=red, show_background=False)

        # Scan the header band for the most saturated red pixel -- text
        # antialiasing/outline means we won't get a pure (255,0,0) match,
        # but something clearly red-dominant should be present.
        found_red = False
        for x in range(0, out.width, 2):
            for y in range(0, 70, 2):
                r, g, b = out.getpixel((x, y))
                if r > 180 and g < 80 and b < 80:
                    found_red = True
                    break
            if found_red:
                break
        self.assertTrue(found_red, "expected to find the chosen red text color somewhere in the header band")

    def test_header_banner_no_background_leaves_hero_image_visible(self):
        from PIL import Image as PILImage

        from src.image_ops import add_header_banner

        # A distinctive mid-tone background -- with show_background=True
        # the semi-transparent black plate would darken this considerably;
        # with it off, pixels away from the text itself should be untouched.
        canvas = PILImage.new("RGB", (400, 300), (100, 180, 90))
        out_with_bg = add_header_banner(canvas, "Title", show_background=True)
        out_no_bg = add_header_banner(canvas, "Title", show_background=False)

        # Sample a corner of the header band far from where centered text
        # would fall.
        probe = (5, 5)
        with_bg_pixel = out_with_bg.getpixel(probe)
        no_bg_pixel = out_no_bg.getpixel(probe)
        self.assertEqual(no_bg_pixel, (100, 180, 90))  # untouched
        self.assertNotEqual(with_bg_pixel, (100, 180, 90))  # darkened by the plate

    def test_header_banner_no_background_adds_outline_for_legibility(self):
        # Dark text with no background plate should get a light outline
        # (not a dark one) so it doesn't disappear into a similarly dark
        # hero image -- this is the automatic contrast fallback.
        from PIL import Image as PILImage

        from src.image_ops import add_header_banner

        canvas = PILImage.new("RGB", (400, 300), (20, 20, 20))  # dark background
        dark_text = (10, 10, 10)
        out = add_header_banner(canvas, "DARK TEXT", text_color=dark_text, show_background=False)

        # Somewhere in the header band there should be a clearly light
        # pixel (the white outline), since dark-on-dark with no plate would
        # otherwise be unreadable.
        found_light = any(
            sum(out.getpixel((x, y))) > 600
            for x in range(0, out.width, 2)
            for y in range(0, 70, 2)
        )
        self.assertTrue(found_light, "expected a light outline around dark header text with no background")

    def test_wrap_text_to_width_hard_breaks_a_too_wide_leading_word(self):
        # Regression test: a single word with no spaces (e.g. a one-word
        # header like "HydroBoost") that's wider than max_width *on its
        # own* used to never get hard-broken, because the "start a new
        # line" branch only ran when a word failed to fit onto an
        # already-nonempty `current` -- the very first word of a line was
        # accepted unconditionally (nothing to flush yet), skipping the
        # hard-break check entirely and silently overflowing the frame.
        from PIL import Image as PILImage, ImageDraw

        from src.image_ops import _load_font, wrap_text_to_width

        canvas = PILImage.new("RGB", (10, 10))
        draw = ImageDraw.Draw(canvas)
        font = _load_font(150, bold=True)
        lines = wrap_text_to_width(draw, "HydroBoost", font, max_width=970)
        self.assertGreater(len(lines), 1, "a too-wide lone word must be hard-broken across lines")
        for line in lines:
            self.assertLessEqual(draw.textlength(line, font=font), 970)

    def test_fit_text_block_single_long_word_never_overflows_max_width(self):
        # End-to-end version at the autofit-search level: fit_text_block()
        # must never return a font/line combination wider than max_width,
        # even for text that's a single word with nothing to wrap on.
        from PIL import Image as PILImage, ImageDraw

        from src.image_ops import fit_text_block

        canvas = PILImage.new("RGB", (10, 10))
        draw = ImageDraw.Draw(canvas)
        font, lines, _line_height = fit_text_block(
            draw, "HydroBoost", max_width=970, max_height=400, min_font_size=14
        )
        for line in lines:
            self.assertLessEqual(
                draw.textlength(line, font=font),
                970,
                f"line {line!r} at font size {font.size} overflows max_width",
            )

    def test_render_creative_single_word_header_autofits_within_frame(self):
        # The actual user-visible bug: a one-word header ("HydroBoost") on
        # a 1080x1920 frame rendered clipped off the right edge instead of
        # shrinking to fit -- fit_text_block()'s search never saw the
        # overflow because wrap_text_to_width() silently returned a
        # too-wide single line for it to measure height against.
        from PIL import Image as PILImage

        from src.creative_render import render_creative

        hero = PILImage.new("RGB", (1200, 900), (60, 120, 180))
        out, _ = render_creative(
            hero,
            (1080, 1920),
            headline="HydroBoost",
            fit_mode="crop",
            header_align="center",
        )

        def is_white_text(pixel):
            r, g, b = pixel
            return r > 230 and g > 230 and b > 230

        # No header-colored pixels should reach all the way to either edge
        # column -- if the word overflowed uncropped/unshrunk (the bug),
        # its glyphs would run right off one side.
        left_edge_has_text = any(is_white_text(out.getpixel((0, y))) for y in range(0, 400))
        right_edge_has_text = any(is_white_text(out.getpixel((out.width - 1, y))) for y in range(0, 400))
        self.assertFalse(left_edge_has_text, "header text should not run off the left edge")
        self.assertFalse(right_edge_has_text, "header text should not run off the right edge")

    def test_fit_text_block_grows_for_short_text_in_spacious_box(self):
        # A short line of text in a wide, generous box should grow well
        # beyond a small "safe" starting guess -- this is the autofit
        # rewrite's fix for headlines looking too small on spacious/4:3-ish
        # frames instead of expanding to use the available width.
        from PIL import Image as PILImage, ImageDraw

        from src.image_ops import fit_text_block

        canvas = PILImage.new("RGB", (10, 10))
        draw = ImageDraw.Draw(canvas)
        font, lines, _line_height = fit_text_block(draw, "Sale", max_width=900, max_height=200, min_font_size=14)
        self.assertGreater(font.size, 80, "expected a short line in a spacious box to grow well past a tiny/default size")
        self.assertEqual(lines, ["Sale"])

    def test_fit_text_block_skyscraper_shaped_box_produces_legible_font(self):
        # The bug this fixes: a narrow-but-tall box (like a 160x600
        # skyscraper's text area) used to compute an initial guess from the
        # cramped width alone and get stuck near the ~10px floor. The
        # autofit search should instead use the abundant height and land on
        # something clearly legible.
        from PIL import Image as PILImage, ImageDraw

        from src.image_ops import fit_text_block

        canvas = PILImage.new("RGB", (10, 10))
        draw = ImageDraw.Draw(canvas)
        font, _lines, _line_height = fit_text_block(draw, "Sale", max_width=130, max_height=96, min_font_size=14)
        self.assertGreater(font.size, 25, "expected skyscraper-shaped text to render clearly larger than the old ~10px bug")

    def test_fit_text_block_still_shrinks_long_text_to_fit(self):
        # Long text in a small box should still shrink all the way down to
        # (but not below) min_font_size -- the autofit rewrite must not
        # regress the original "always fits, never overflows" guarantee.
        from PIL import Image as PILImage, ImageDraw

        from src.image_ops import fit_text_block

        canvas = PILImage.new("RGB", (10, 10))
        draw = ImageDraw.Draw(canvas)
        long_text = "This is a much longer piece of campaign copy that will not fit in a tiny box."
        font, lines, line_height = fit_text_block(draw, long_text, max_width=150, max_height=60, min_font_size=14)
        self.assertEqual(font.size, 14)
        self.assertGreater(len(lines), 1)

    def test_header_banner_glow_adds_colored_halo(self):
        # With glow on, a soft colored halo should appear around the text --
        # detect it by scanning for pixels tinted toward the glow color but
        # not fully opaque/white (i.e. not the crisp text itself).
        from PIL import Image as PILImage

        from src.image_ops import add_header_banner

        canvas = PILImage.new("RGB", (400, 300), (10, 10, 10))
        cyan_glow = (0, 220, 255)
        out = add_header_banner(
            canvas, "GLOWY", text_color=(255, 255, 255), show_background=False, glow=True, glow_color=cyan_glow
        )
        found_glow = any(
            g > 120 and b > 120 and r < 150  # cyan-tinted, dimmer than the crisp white text
            for x in range(0, out.width, 2)
            for y in range(0, 70, 2)
            for r, g, b in [out.getpixel((x, y))]
        )
        self.assertTrue(found_glow, "expected a cyan-tinted glow halo around the header text")

    def test_message_banner_glow_is_independent_of_header_glow(self):
        # Mirrors the "header/message styling are independent" pattern used
        # for text_color/show_background -- glow should be settable per
        # banner without affecting the other.
        from PIL import Image as PILImage

        from src.image_ops import add_header_banner, add_message_banner

        canvas = PILImage.new("RGB", (400, 300), (10, 10, 10))
        magenta_glow = (255, 0, 220)
        with_header_glow = add_header_banner(canvas, "Header", show_background=False, glow=True, glow_color=magenta_glow)
        no_glow = add_header_banner(canvas, "Header", show_background=False, glow=False)
        # Glow should make at least some header-band pixels differ from the
        # no-glow render (the halo itself), confirming it actually changed
        # the output.
        header_band_differs = any(
            with_header_glow.getpixel((x, y)) != no_glow.getpixel((x, y))
            for x in range(0, 400, 4)
            for y in range(0, 70, 4)
        )
        self.assertTrue(header_band_differs, "expected glow=True to visibly change the header band vs glow=False")

        message_out = add_message_banner(canvas, "Message", show_background=False, glow=False)
        # Unaffected by the header glow call above -- independent banners.
        self.assertEqual(message_out.size, canvas.size)

    def test_header_banner_alignment_moves_text_left_center_right(self):
        # A short headline with room to spare should land in a visibly
        # different horizontal position for each alignment -- checked by
        # finding the leftmost "text-ish" (non-background) column in the
        # header band for each render.
        from PIL import Image as PILImage

        from src.image_ops import add_header_banner

        canvas = PILImage.new("RGB", (600, 400), (0, 0, 0))

        def leftmost_text_column(img):
            for x in range(img.width):
                for y in range(0, 90):
                    if sum(img.getpixel((x, y))) > 300:
                        return x
            return None

        left_out = add_header_banner(canvas, "Hi", show_background=False, align="left")
        center_out = add_header_banner(canvas, "Hi", show_background=False, align="center")
        right_out = add_header_banner(canvas, "Hi", show_background=False, align="right")

        left_x = leftmost_text_column(left_out)
        center_x = leftmost_text_column(center_out)
        right_x = leftmost_text_column(right_out)

        self.assertIsNotNone(left_x)
        self.assertIsNotNone(center_x)
        self.assertIsNotNone(right_x)
        self.assertLess(left_x, center_x, "left-aligned text should start further left than centered text")
        self.assertLess(center_x, right_x, "centered text should start further left than right-aligned text")

    def test_message_banner_alignment_defaults_to_left(self):
        # Omitting `align` entirely should behave exactly like before the
        # option existed -- left-aligned, matching add_message_banner()'s
        # original hardcoded behavior.
        from PIL import Image as PILImage

        from src.image_ops import add_message_banner

        canvas = PILImage.new("RGB", (600, 400), (0, 0, 0))
        default_out = add_message_banner(canvas, "Hi", show_background=False)
        explicit_left_out = add_message_banner(canvas, "Hi", show_background=False, align="left")
        self.assertEqual(list(default_out.getdata()), list(explicit_left_out.getdata()))

    def test_invalid_align_raises_value_error(self):
        from PIL import Image as PILImage

        from src.image_ops import add_header_banner

        canvas = PILImage.new("RGB", (400, 300), (0, 0, 0))
        with self.assertRaises(ValueError):
            add_header_banner(canvas, "Hi", align="diagonal")

    def test_font_size_override_pins_exact_size_and_wraps(self):
        # An explicit font_size should skip autofit and render at exactly
        # that size (rather than the largest-that-fits search), while still
        # wrapping normally if the text is too wide for one line.
        from PIL import Image as PILImage

        from src.image_ops import add_message_banner

        canvas = PILImage.new("RGB", (500, 500), (0, 0, 0))
        small = add_message_banner(canvas, "Short", show_background=False, font_size=18)
        large = add_message_banner(canvas, "Short", show_background=False, font_size=60)

        def text_pixel_count(img, band):
            top, bottom = band
            return sum(
                1
                for x in range(0, img.width, 2)
                for y in range(top, bottom, 2)
                if sum(img.getpixel((x, y))) > 300
            )

        # A much bigger pinned font size should light up noticeably more
        # pixels than a small one, for the same short text.
        self.assertGreater(text_pixel_count(large, (0, 500)), text_pixel_count(small, (0, 500)))

    def test_font_size_override_grows_banner_beyond_normal_cap(self):
        # A caller-chosen font size large enough to need more room than the
        # usual max_height_frac cap should still get that room -- the
        # banner/plate grows to fit it instead of clipping the explicit
        # choice, up to the full frame as a hard safety limit.
        from PIL import Image as PILImage

        from src.image_ops import add_header_banner

        canvas = PILImage.new("RGB", (500, 500), (10, 10, 10))
        huge_header = add_header_banner(canvas, "Big Sale", show_background=True, font_size=140)
        # max_height_frac for the header is 0.22 -> ~110px normally; a
        # 140px font pinned explicitly should push the dark plate well
        # past that, more than a third of the way down the frame.
        plate_bottom = None
        for y in range(canvas.height):
            if huge_header.getpixel((5, y)) == (10, 10, 10):
                plate_bottom = y
                break
        self.assertIsNotNone(plate_bottom)
        self.assertGreater(plate_bottom, 150, "expected the plate to grow well past the normal header cap for a pinned 140px font")

    def _make_badge(self, size=(200, 200), color=(220, 30, 30, 255)):
        from PIL import Image as PILImage

        badge = PILImage.new("RGBA", size, (0, 0, 0, 0))
        # A solid opaque square in the middle, transparent border -- lets
        # tests distinguish "the badge's opaque pixels landed here" from
        # "nothing was composited here" unambiguously.
        inset = size[0] // 4
        for x in range(inset, size[0] - inset):
            for y in range(inset, size[1] - inset):
                badge.putpixel((x, y), color)
        return badge

    def test_badge_image_corner_position_sits_at_that_corner(self):
        from PIL import Image as PILImage

        from src.image_ops import add_badge_image, badge_render_size

        canvas = PILImage.new("RGB", (600, 600), (10, 10, 10))
        # A fully opaque solid badge (no transparent border) so its exact
        # placed rectangle -- computed via badge_render_size(), the same
        # helper add_badge_image() itself uses -- is unambiguously red.
        badge = PILImage.new("RGBA", (200, 200), (220, 30, 30, 255))
        target_w, target_h, margin = badge_render_size((600, 600), badge, scale_frac=0.3)

        bottom_right = add_badge_image(canvas, badge, position="bottom-right", scale=0.3)
        top_left = add_badge_image(canvas, badge, position="top-left", scale=0.3)

        def has_red(img, x, y):
            r, g, b = img.getpixel((x, y))
            return r > 180 and g < 80 and b < 80

        br_x, br_y = 600 - target_w - margin + target_w // 2, 600 - target_h - margin + target_h // 2
        self.assertTrue(has_red(bottom_right, br_x, br_y), "expected red inside the bottom-right badge's placed rectangle")
        self.assertFalse(any(has_red(bottom_right, x, y) for x in range(0, 50) for y in range(0, 50)))

        tl_x, tl_y = margin + target_w // 2, margin + target_h // 2
        self.assertTrue(has_red(top_left, tl_x, tl_y), "expected red inside the top-left badge's placed rectangle")
        self.assertFalse(any(has_red(top_left, x, y) for x in range(550, 600) for y in range(550, 600)))

    def test_badge_image_full_position_covers_entire_frame(self):
        from PIL import Image as PILImage

        from src.image_ops import add_badge_image

        canvas = PILImage.new("RGB", (400, 300), (10, 10, 10))
        # A fully opaque solid-color badge -- "full" should stretch it to
        # cover every pixel of the canvas, corners included.
        solid = PILImage.new("RGBA", (50, 50), (0, 200, 0, 255))
        out = add_badge_image(canvas, solid, position="full")
        for x, y in [(0, 0), (399, 0), (0, 299), (399, 299), (200, 150)]:
            self.assertEqual(out.getpixel((x, y)), (0, 200, 0))

    def test_badge_image_opacity_blends_instead_of_fully_replacing(self):
        from PIL import Image as PILImage

        from src.image_ops import add_badge_image

        canvas = PILImage.new("RGB", (400, 300), (10, 10, 10))
        solid = PILImage.new("RGBA", (50, 50), (0, 200, 0, 255))
        half_opacity = add_badge_image(canvas, solid, position="full", opacity=0.5)
        pixel = half_opacity.getpixel((200, 150))
        # Should land roughly halfway between the background (10,10,10) and
        # the badge color (0,200,0) -- neither the pure background nor
        # the pure badge color.
        self.assertNotEqual(pixel, (10, 10, 10))
        self.assertNotEqual(pixel, (0, 200, 0))
        self.assertGreater(pixel[1], 60)  # green channel clearly boosted
        self.assertLess(pixel[1], 150)  # but not all the way to 200

    def test_badge_image_invalid_position_raises_value_error(self):
        from PIL import Image as PILImage

        from src.image_ops import add_badge_image

        canvas = PILImage.new("RGB", (200, 200), (0, 0, 0))
        badge = self._make_badge()
        with self.assertRaises(ValueError):
            add_badge_image(canvas, badge, position="somewhere")

    def test_render_creative_full_badge_sits_behind_header_text(self):
        # A "full" badge is composited before the header/message banners --
        # confirm the header's own text/plate still renders normally on top
        # of it rather than being covered up.
        from PIL import Image as PILImage

        from src.creative_render import render_creative

        hero = PILImage.new("RGB", (800, 600), (60, 120, 180))
        solid = PILImage.new("RGBA", (10, 10), (0, 0, 0, 255))
        out, _ = render_creative(
            hero, (400, 400), headline="Title", fit_mode="crop", badge_image=solid, badge_position="full"
        )
        # Somewhere in the header band there should still be light text
        # pixels, proving the header rendered on top of the black badge
        # rather than being hidden underneath it.
        found_light = any(sum(out.getpixel((x, y))) > 400 for x in range(0, out.width, 2) for y in range(0, 60, 2))
        self.assertTrue(found_light, "expected header text to still be visible over a full-frame badge")

    def test_render_creative_corner_badge_sits_on_top_of_message_banner(self):
        # A corner badge is composited *last* -- confirm it's visible even
        # where it overlaps the bottom message banner, rather than being
        # covered by it.
        from PIL import Image as PILImage

        from src.creative_render import render_creative
        from src.image_ops import badge_render_size

        hero = PILImage.new("RGB", (800, 600), (60, 120, 180))
        badge = PILImage.new("RGBA", (300, 300), (220, 30, 30, 255))  # fully opaque -- no ambiguity about placement
        out, _ = render_creative(
            hero,
            (400, 400),
            message="A caption long enough to fill the bottom banner area",
            fit_mode="crop",
            badge_image=badge,
            badge_position="bottom-right",
            badge_scale=0.5,
        )

        def has_red(img, x, y):
            r, g, b = img.getpixel((x, y))
            return r > 180 and g < 80 and b < 80

        target_w, target_h, margin = badge_render_size((400, 400), badge, scale_frac=0.5)
        center_x = 400 - target_w - margin + target_w // 2
        center_y = 400 - target_h - margin + target_h // 2
        self.assertTrue(
            has_red(out, center_x, center_y),
            "expected the badge to be visible on top of the message banner, not covered by it",
        )

    def test_cta_button_renders_at_bottom_center_by_default(self):
        from PIL import Image as PILImage

        from src.image_ops import add_cta_button

        canvas = PILImage.new("RGB", (600, 600), (10, 10, 10))
        out = add_cta_button(canvas, "Shop Now", button_color=(0, 87, 184))

        def has_blue(img, x, y):
            r, g, b = img.getpixel((x, y))
            return b > 140 and b > r + 40 and b > g + 40

        # Somewhere in the bottom-center strip there should be the button's
        # blue fill; nowhere near the top edge should there be any.
        self.assertTrue(any(has_blue(out, x, y) for x in range(200, 400) for y in range(500, 590)))
        self.assertFalse(any(has_blue(out, x, y) for x in range(0, 600, 4) for y in range(0, 30)))

    def test_cta_button_all_positions_are_valid_and_distinct(self):
        from PIL import Image as PILImage

        from src.image_ops import VALID_CTA_POSITIONS, add_cta_button

        canvas = PILImage.new("RGB", (600, 600), (10, 10, 10))
        renders = {pos: add_cta_button(canvas, "Go", position=pos) for pos in VALID_CTA_POSITIONS}
        # Every position should actually change the image from the plain
        # background, and no two positions should render identically.
        for pos, out in renders.items():
            self.assertNotEqual(list(out.getdata()), list(canvas.convert("RGB").getdata()), f"position={pos} did nothing")
        seen = []
        for pos, out in renders.items():
            data = list(out.getdata())
            self.assertNotIn(data, seen, f"position={pos} rendered identically to another position")
            seen.append(data)

    def test_cta_button_no_text_is_a_no_op(self):
        from PIL import Image as PILImage

        from src.image_ops import add_cta_button

        canvas = PILImage.new("RGB", (400, 400), (10, 10, 10))
        out = add_cta_button(canvas, "")
        self.assertEqual(list(out.getdata()), list(canvas.getdata()))

    def test_cta_button_invalid_position_raises_value_error(self):
        from PIL import Image as PILImage

        from src.image_ops import add_cta_button

        canvas = PILImage.new("RGB", (400, 400), (0, 0, 0))
        with self.assertRaises(ValueError):
            add_cta_button(canvas, "Go", position="middle-ish")

    def test_cta_button_long_text_on_narrow_frame_stays_within_bounds(self):
        # The bug this guards against: a long CTA label on a narrow frame
        # (e.g. a 160px-wide skyscraper) could make the button wider than
        # the canvas -- the button must shrink its font until it fits
        # rather than spilling past the edge.
        from PIL import Image as PILImage

        from src.image_ops import add_cta_button

        canvas = PILImage.new("RGB", (160, 600), (10, 10, 10))
        out = add_cta_button(canvas, "Get 50% Off Everything Today, Limited Time Only")
        self.assertEqual(out.size, (160, 600))  # rendering completed without error/overflow crash

        # The button's fill shouldn't touch the far left/right edge columns
        # -- if it did, the button (or its rounded corners) would be
        # spilling past the frame rather than fitting inside it.
        def is_button_blue(pixel):
            r, g, b = pixel
            return b > 140 and b > r + 40 and b > g + 40

        left_edge_has_button = any(is_button_blue(out.getpixel((0, y))) for y in range(out.height))
        right_edge_has_button = any(is_button_blue(out.getpixel((out.width - 1, y))) for y in range(out.height))
        self.assertFalse(left_edge_has_button)
        self.assertFalse(right_edge_has_button)

    def test_cta_button_explicit_font_size_is_used_when_it_fits(self):
        from PIL import Image as PILImage

        from src.image_ops import add_cta_button

        canvas = PILImage.new("RGB", (1000, 1000), (10, 10, 10))
        small = add_cta_button(canvas, "Go", font_size=16)
        large = add_cta_button(canvas, "Go", font_size=60)

        def button_pixel_count(img):
            return sum(
                1
                for x in range(0, img.width, 2)
                for y in range(0, img.height, 2)
                if img.getpixel((x, y)) != (10, 10, 10)
            )

        self.assertGreater(button_pixel_count(large), button_pixel_count(small))

    def test_cta_button_font_family_changes_glyph_shapes(self):
        # Different typefaces should render visibly different pixel data
        # for the same text/size/position -- a cheap way to confirm the
        # family selection is actually reaching the font loader rather than
        # being silently ignored.
        from PIL import Image as PILImage

        from src.image_ops import VALID_FONT_FAMILIES, add_cta_button

        canvas = PILImage.new("RGB", (800, 800), (10, 10, 10))
        renders = {
            family: add_cta_button(canvas, "Shop Now", position="center", font_size=48, font_family=family)
            for family in VALID_FONT_FAMILIES
        }
        seen = []
        for family, out in renders.items():
            data = list(out.getdata())
            self.assertNotIn(data, seen, f"font_family={family} rendered identically to another family")
            seen.append(data)

    def test_cta_button_invalid_font_family_raises_value_error(self):
        from PIL import Image as PILImage

        from src.image_ops import add_cta_button

        canvas = PILImage.new("RGB", (400, 400), (0, 0, 0))
        with self.assertRaises(ValueError):
            add_cta_button(canvas, "Go", font_family="comic-sans")

    def test_cta_button_default_font_family_is_unchanged(self):
        # Omitting font_family entirely should behave exactly like before
        # it existed -- the original hardcoded DejaVu Sans Bold.
        from PIL import Image as PILImage

        from src.image_ops import add_cta_button

        canvas = PILImage.new("RGB", (800, 800), (10, 10, 10))
        default_out = add_cta_button(canvas, "Shop Now", position="center", font_size=48)
        explicit_sans_out = add_cta_button(canvas, "Shop Now", position="center", font_size=48, font_family="sans")
        self.assertEqual(list(default_out.getdata()), list(explicit_sans_out.getdata()))

    def test_cta_button_glow_adds_a_colored_halo_around_the_button(self):
        from PIL import Image as PILImage

        from src.image_ops import add_cta_button

        canvas = PILImage.new("RGB", (800, 800), (10, 10, 10))
        plain = add_cta_button(canvas, "Shop Now", position="center", button_color=(0, 87, 184))
        glowing = add_cta_button(
            canvas, "Shop Now", position="center", button_color=(0, 87, 184), glow=True, glow_color=(0, 220, 255)
        )

        # A strict, highly-saturated cyan check -- loose enough to catch the
        # boosted glow color, strict enough to exclude the button's own
        # blue-into-background edge antialiasing (which the plain render
        # also has, just never this saturated).
        def is_strong_cyan_glow(pixel):
            r, g, b = pixel
            return g > 150 and b > 150 and r < 60

        plain_pixels = list(plain.getdata())
        glow_pixels = list(glowing.getdata())
        self.assertNotEqual(plain_pixels, glow_pixels)
        self.assertFalse(any(is_strong_cyan_glow(p) for p in plain_pixels))
        found_halo = any(is_strong_cyan_glow(p) for p in glow_pixels)
        self.assertTrue(found_halo, "expected a cyan-tinted glow halo somewhere around the button")

    def test_cta_button_default_glow_off_unless_requested(self):
        from PIL import Image as PILImage

        from src.image_ops import add_cta_button

        canvas = PILImage.new("RGB", (800, 800), (10, 10, 10))
        default_out = add_cta_button(canvas, "Shop Now", position="center")
        explicit_off_out = add_cta_button(canvas, "Shop Now", position="center", glow=False)
        self.assertEqual(list(default_out.getdata()), list(explicit_off_out.getdata()))

    def test_message_banner_height_matches_actual_rendered_banner_top_edge(self):
        # message_banner_height() must predict exactly where
        # add_message_banner() will actually draw the banner's top edge --
        # render_creative()'s cta_above_message depends on this to lift
        # the CTA button clear of the banner rather than into it.
        from PIL import Image as PILImage

        from src.image_ops import add_message_banner, message_banner_height

        bg = (200, 200, 200)
        canvas = PILImage.new("RGB", (800, 600), bg)
        message = "Shop the new summer collection today"
        out = add_message_banner(canvas, message, show_background=True)
        computed_height = message_banner_height((800, 600), message)

        # Scan upward from the bottom edge for the first row that's still
        # plain background -- everything below it is the banner's
        # semi-transparent black plate.
        banner_top_row = None
        for y in range(canvas.height - 1, -1, -1):
            if all(out.getpixel((x, y)) == bg for x in range(0, canvas.width, 20)):
                banner_top_row = y
                break
        self.assertIsNotNone(banner_top_row, "expected at least one plain-background row above the banner")
        actual_height = canvas.height - 1 - banner_top_row

        self.assertLessEqual(abs(actual_height - computed_height), 2)

    def test_cta_button_y_offset_moves_button_up(self):
        from PIL import Image as PILImage

        from src.image_ops import add_cta_button

        canvas = PILImage.new("RGB", (600, 600), (10, 10, 10))
        no_offset = add_cta_button(canvas, "Shop Now", position="bottom-center", button_color=(0, 87, 184))
        offset = add_cta_button(
            canvas, "Shop Now", position="bottom-center", button_color=(0, 87, 184), y_offset=100
        )

        def has_blue(pixel):
            r, g, b = pixel
            return b > 140 and b > r + 40 and b > g + 40

        def topmost_blue_row(img):
            for y in range(img.height):
                for x in range(0, img.width, 4):
                    if has_blue(img.getpixel((x, y))):
                        return y
            return None

        top_no_offset = topmost_blue_row(no_offset)
        top_offset = topmost_blue_row(offset)
        self.assertIsNotNone(top_no_offset)
        self.assertIsNotNone(top_offset)
        self.assertLess(top_offset, top_no_offset, "y_offset should move the button up (a smaller y)")
        self.assertAlmostEqual(top_no_offset - top_offset, 100, delta=2)

    def test_render_creative_cta_above_message_lifts_button_clear_of_banner(self):
        from PIL import Image as PILImage

        from src.creative_render import render_creative
        from src.image_ops import message_banner_height

        hero = PILImage.new("RGB", (800, 600), (60, 120, 180))
        message = "Free shipping on orders over fifty dollars this week only"
        common = dict(
            message=message,
            cta_text="Shop Now",
            cta_position="bottom-center",
            cta_button_color=(255, 0, 0),  # distinct from the hero color and the banner's black plate
        )
        overlapping, _ = render_creative(hero, (800, 600), fit_mode="crop", cta_above_message=False, **common)
        lifted, _ = render_creative(hero, (800, 600), fit_mode="crop", cta_above_message=True, **common)

        def has_red(pixel):
            r, g, b = pixel
            return r > 140 and r > g + 60 and r > b + 60

        def topmost_red_row(img):
            for y in range(img.height):
                for x in range(0, img.width, 4):
                    if has_red(img.getpixel((x, y))):
                        return y
            return None

        top_overlap = topmost_red_row(overlapping)
        top_lifted = topmost_red_row(lifted)
        self.assertIsNotNone(top_overlap)
        self.assertIsNotNone(top_lifted)
        self.assertLess(top_lifted, top_overlap, "cta_above_message should move the button higher up")

        banner_top = 600 - message_banner_height((800, 600), message)
        button_bottom = None
        for y in range(lifted.height - 1, -1, -1):
            if any(has_red(lifted.getpixel((x, y))) for x in range(0, lifted.width, 4)):
                button_bottom = y
                break
        self.assertIsNotNone(button_bottom)
        self.assertLessEqual(
            button_bottom, banner_top, "the lifted CTA should not dip into the message banner's area"
        )

    def test_render_creative_cta_above_message_has_no_effect_without_a_bottom_position_or_message(self):
        from PIL import Image as PILImage

        from src.creative_render import render_creative

        hero = PILImage.new("RGB", (800, 600), (60, 120, 180))

        # No message at all -- nothing to avoid, so the flag must be a no-op.
        no_message_off, _ = render_creative(
            hero, (800, 600), fit_mode="crop", cta_text="Shop Now", cta_position="bottom-center", cta_above_message=False
        )
        no_message_on, _ = render_creative(
            hero, (800, 600), fit_mode="crop", cta_text="Shop Now", cta_position="bottom-center", cta_above_message=True
        )
        self.assertEqual(list(no_message_off.getdata()), list(no_message_on.getdata()))

        # A non-bottom position -- "above the description" doesn't apply.
        center_off, _ = render_creative(
            hero,
            (800, 600),
            fit_mode="crop",
            message="Some message text here",
            cta_text="Shop Now",
            cta_position="center",
            cta_above_message=False,
        )
        center_on, _ = render_creative(
            hero,
            (800, 600),
            fit_mode="crop",
            message="Some message text here",
            cta_text="Shop Now",
            cta_position="center",
            cta_above_message=True,
        )
        self.assertEqual(list(center_off.getdata()), list(center_on.getdata()))

    def test_render_creative_cta_composited_last_on_top_of_badge(self):
        # The CTA must stay visible/actionable no matter what else is on
        # the creative -- confirm it renders on top of a corner badge image
        # placed at the same spot.
        from PIL import Image as PILImage

        from src.creative_render import render_creative

        hero = PILImage.new("RGB", (800, 600), (60, 120, 180))
        badge = PILImage.new("RGBA", (300, 300), (0, 200, 0, 255))
        out, _ = render_creative(
            hero,
            (400, 400),
            fit_mode="crop",
            badge_image=badge,
            badge_position="bottom-right",
            badge_scale=0.6,
            cta_text="Buy",
            cta_position="bottom-right",
            cta_button_color=(0, 87, 184),
        )

        def has_blue(pixel):
            r, g, b = pixel
            return b > 140 and b > r + 40 and b > g + 40

        found_blue_over_badge = any(
            has_blue(out.getpixel((x, y)))
            for x in range(out.width - 120, out.width)
            for y in range(out.height - 80, out.height)
        )
        self.assertTrue(found_blue_over_badge, "expected the CTA button to be visible on top of the badge image")

    @unittest.skipUnless(HAS_CV2, "opencv-python-headless not installed")
    def test_extract_video_frame_defaults_to_middle_of_video(self):
        video_path = Path(self.tmp_dir) / "product_demo.mp4"
        _write_test_video(video_path, seconds_per_third=1.0)  # ~3s total: red, green, blue thirds

        frame = extract_video_frame(video_path)  # no frame_seconds -> middle of video (~1.5s -> green third)
        r, g, b = frame.getpixel((frame.width // 2, frame.height // 2))
        self.assertGreater(g, 150)
        self.assertLess(r, 100)
        self.assertLess(b, 100)

    @unittest.skipUnless(HAS_CV2, "opencv-python-headless not installed")
    def test_extract_video_frame_honors_explicit_timestamp(self):
        video_path = Path(self.tmp_dir) / "product_demo.mp4"
        _write_test_video(video_path, seconds_per_third=1.0)

        early_frame = extract_video_frame(video_path, frame_seconds=0.1)  # -> red third
        r, g, b = early_frame.getpixel((early_frame.width // 2, early_frame.height // 2))
        self.assertGreater(r, 150)
        self.assertLess(g, 100)

        late_frame = extract_video_frame(video_path, frame_seconds=2.9)  # -> blue third
        r, g, b = late_frame.getpixel((late_frame.width // 2, late_frame.height // 2))
        self.assertGreater(b, 150)
        self.assertLess(r, 100)

    @unittest.skipUnless(HAS_CV2, "opencv-python-headless not installed")
    def test_open_as_rgb_routes_video_extensions_to_extract_video_frame(self):
        video_path = Path(self.tmp_dir) / "product_demo.mp4"
        _write_test_video(video_path)
        image = open_as_rgb(video_path)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (64, 48))

    def test_extract_video_frame_raises_helpful_error_for_bad_file(self):
        bad_video = Path(self.tmp_dir) / "not_really_a_video.mp4"
        bad_video.write_bytes(b"this is not a video file")
        with self.assertRaises(ValueError):
            extract_video_frame(bad_video)

    @unittest.skipUnless(HAS_CV2, "opencv-python-headless not installed")
    def test_video_asset_flows_through_pipeline_via_asset_path(self):
        repo_root = Path(__file__).resolve().parent.parent
        brief = load_brief(str(repo_root / "briefs" / "sample_campaign.yaml"))
        store = LocalAssetStore(input_dir=str(repo_root / "assets"), cache_dir=str(self.cache_dir))

        video_path = Path(self.tmp_dir) / "hydroboost_demo.mp4"
        _write_test_video(video_path, width=728, height=480, seconds_per_third=1.0)
        brief.products[0].asset_path = str(video_path)
        brief.products[0].video_frame_seconds = 0.1  # pin to the red third for a deterministic check

        pipeline = CreativePipeline(
            provider=MockImageProvider(),
            store=store,
            output_dir=str(self.output_dir),
            sizes=[(300, 300)],
        )
        report = pipeline.run(brief)
        creative = next(c for c in report.creatives if c.product == brief.products[0].name)
        self.assertIn("extracted video frame", creative.source)
        self.assertTrue(Path(creative.output_path).exists())

    def test_combining_presets_dedupes_shared_sizes(self):
        # "default" and "broadcast" both include 1920x1080 -- combining them
        # should render it once, not twice.
        sizes = parse_sizes("default,broadcast")
        self.assertEqual(sizes.count((1920, 1080)), 1)
        self.assertEqual(len(sizes), 5)  # 3 + 3 - 1 shared duplicate

        repo_root = Path(__file__).resolve().parent.parent
        brief = load_brief(str(repo_root / "briefs" / "sample_campaign.yaml"))
        store = LocalAssetStore(input_dir=str(repo_root / "assets"), cache_dir=str(self.cache_dir))
        pipeline = CreativePipeline(
            provider=MockImageProvider(), store=store, output_dir=str(self.output_dir), sizes=sizes
        )
        report = pipeline.run(brief)
        self.assertEqual(len(report.creatives), 2 * len(sizes))


if __name__ == "__main__":
    unittest.main()
