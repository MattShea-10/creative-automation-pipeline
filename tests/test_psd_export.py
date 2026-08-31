"""Tests for src/psd_export.py -- turning a render_creative_layers() stack
into an actual multi-layer .psd file, renamed and cropped so it can be
re-uploaded straight back into the app as a PSD template.

Run with: python -m unittest discover tests
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageChops
from psd_tools import PSDImage

from src.creative_render import render_creative, render_creative_layers
from src.image_ops import get_psd_layer_boxes
from src.psd_export import REUPLOAD_LAYER_NAMES, _tight_bbox_crop, build_layered_psd, save_layered_psd


class PsdExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_build_layered_psd_rejects_empty_layer_list(self):
        with self.assertRaises(ValueError):
            build_layered_psd([], (100, 100))

    def test_build_layered_psd_renames_layers_for_reupload_by_default(self):
        size = (200, 150)
        bg = Image.new("RGBA", size, (10, 20, 30, 255))
        header = Image.new("RGBA", size, (0, 0, 0, 0))
        layers = [("Background", bg), ("Header", header)]
        psd = build_layered_psd(layers, size)
        # "Background" -> "product" (a recognized upload-flow name);
        # "Header" has no matching upload-flow role, so it's untouched.
        self.assertEqual([l.name for l in psd], ["product", "Header"])
        self.assertEqual(psd.size, size)

    def test_build_layered_psd_layer_names_can_be_kept_as_is(self):
        size = (200, 150)
        bg = Image.new("RGBA", size, (10, 20, 30, 255))
        layers = [("Background", bg)]
        psd = build_layered_psd(layers, size, layer_names={})
        self.assertEqual([l.name for l in psd], ["Background"])

    def test_tight_bbox_crop_shrinks_to_drawn_content_only(self):
        size = (400, 300)
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        img.paste(Image.new("RGBA", (50, 20), (255, 0, 0, 255)), (100, 60))
        cropped, left, top = _tight_bbox_crop(img)
        self.assertEqual((left, top), (100, 60))
        self.assertEqual(cropped.size, (50, 20))

    def test_tight_bbox_crop_leaves_a_fully_opaque_layer_at_full_size(self):
        size = (400, 300)
        img = Image.new("RGBA", size, (10, 20, 30, 255))
        cropped, left, top = _tight_bbox_crop(img)
        self.assertEqual((left, top), (0, 0))
        self.assertEqual(cropped.size, size)

    def test_save_layered_psd_round_trips_and_composites_correctly(self):
        # Build a real render_creative_layers() stack for a creative that
        # exercises header + logo + message + CTA, save it, reopen it with
        # psd_tools, and confirm both the (renamed) layer names and the
        # *composited* pixels match what render_creative() itself would
        # have flattened.
        hero = Image.new("RGB", (800, 600), (50, 100, 150))
        logo = Image.new("RGBA", (160, 80), (255, 209, 0, 255))
        kwargs = dict(
            headline="A Fairly Long Product Headline Here",
            message="Shop the new drop today",
            logo=logo,
            logo_position="top-right",
            cta_text="Shop Now",
            cta_position="bottom-right",
            fit_mode="crop",
        )
        expected, _ = render_creative(hero, (800, 600), **kwargs)
        layers = render_creative_layers(hero, (800, 600), **kwargs)
        original_names = [name for name, _ in layers]
        self.assertEqual(original_names, ["Background", "Header", "Logo", "Message", "CTA"])

        dest = self.tmp_dir / "creative.psd"
        save_layered_psd(layers, (800, 600), dest)
        self.assertTrue(dest.exists())
        self.assertGreater(dest.stat().st_size, 0)

        reopened = PSDImage.open(dest)
        expected_renamed = [REUPLOAD_LAYER_NAMES.get(name, name) for name in original_names]
        self.assertEqual(expected_renamed, ["product", "Header", "logo", "description", "cta"])
        self.assertEqual([l.name for l in reopened], expected_renamed)
        self.assertEqual(reopened.size, (800, 600))

        # The "logo" and "cta" layers should be cropped tight, not left
        # spanning the whole 800x600 canvas -- otherwise a replacement
        # image uploaded against their box would stretch to fill the
        # entire creative instead of landing where the logo/button is.
        logo_layer = next(l for l in reopened if l.name == "logo")
        cta_layer = next(l for l in reopened if l.name == "cta")
        product_layer = next(l for l in reopened if l.name == "product")
        self.assertLess(logo_layer.width, 800)
        self.assertLess(logo_layer.height, 600)
        self.assertLess(cta_layer.width, 800)
        self.assertLess(cta_layer.height, 600)
        # The background/"product" layer is fully opaque, so it's still
        # (correctly) the full canvas.
        self.assertEqual((product_layer.width, product_layer.height), (800, 600))

        composite = reopened.composite().convert("RGB")
        # Compare with a small per-channel tolerance rather than exact
        # equality -- psd_tools' own compositor and Pillow's
        # alpha_composite() are two independent implementations of the
        # same alpha-blending math, and can legitimately round a
        # semi-transparent pixel's blended value +/-1 differently from one
        # another. ImageChops.difference().getextrema() gets the worst-case
        # per-channel difference in C, in one call -- comparing the raw
        # pixel lists with assertEqual would also work logically, but
        # unittest's failure-diff machinery is unusably slow on two
        # 480,000-tuple lists that partially disagree.
        diff = ImageChops.difference(composite, expected)
        per_channel_max = [hi for _lo, hi in diff.getextrema()]
        self.assertLessEqual(
            max(per_channel_max), 2,
            f"composited PSD differs from render_creative()'s own output by more than rounding noise: {per_channel_max}",
        )

    def test_save_layered_psd_background_only_still_produces_a_valid_file(self):
        # No headline/message/logo/CTA/badge -- just the Background layer,
        # renamed to "product".
        hero = Image.new("RGB", (400, 400), (5, 5, 5))
        layers = render_creative_layers(hero, (400, 400), fit_mode="crop")
        self.assertEqual([name for name, _ in layers], ["Background"])

        dest = self.tmp_dir / "background_only.psd"
        save_layered_psd(layers, (400, 400), dest)
        reopened = PSDImage.open(dest)
        self.assertEqual(len(reopened), 1)
        self.assertEqual([l.name for l in reopened], ["product"])
        self.assertEqual(reopened.composite().convert("RGB").getpixel((0, 0)), (5, 5, 5))

    def test_exported_psd_is_accepted_by_the_apps_own_template_upload_check(self):
        # The actual point of the rename/crop: get_psd_layer_boxes() --
        # the same function webapp.py's PSD-template upload flow uses to
        # find "logo"/"description"/"product"/"cta" -- should find all
        # three required layers (plus the optional "cta") in a PSD this
        # module just exported, with no manual renaming in between.
        hero = Image.new("RGB", (728, 480), (40, 90, 140))
        logo = Image.new("RGBA", (160, 80), (255, 209, 0, 255))
        kwargs = dict(
            headline="Header Text",
            message="Some description text",
            logo=logo,
            logo_position="top-right",
            cta_text="Shop Now",
            cta_position="bottom-right",
            fit_mode="crop",
        )
        layers = render_creative_layers(hero, (728, 480), **kwargs)
        dest = self.tmp_dir / "reuploadable.psd"
        save_layered_psd(layers, (728, 480), dest)

        boxes = get_psd_layer_boxes(dest)
        self.assertIn("logo", boxes)
        self.assertIn("description", boxes)
        self.assertIn("product", boxes)
        self.assertIn("cta", boxes)
        # "product" (the background) should be the full canvas; "logo"
        # and "cta" should be small, specific regions within it.
        self.assertEqual(boxes["product"], (0, 0, 728, 480))
        logo_box = boxes["logo"]
        self.assertLess(logo_box[2] - logo_box[0], 728)
        self.assertLess(logo_box[3] - logo_box[1], 480)


if __name__ == "__main__":
    unittest.main()
