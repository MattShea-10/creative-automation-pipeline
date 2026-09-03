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
from src.image_ops import get_psd_backdrop, get_psd_layer_background, get_psd_layer_boxes
from src.psd_export import (
    REUPLOAD_LAYER_NAMES,
    _tight_bbox_crop,
    build_layered_psd,
    save_layered_psd,
    save_layered_psd_preserving_type,
    set_type_layer_colors,
    set_type_layer_text,
)


class PsdExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _overlapping_template_psd(self):
        """A 200x150 PSD shaped like a real ad template: a full-canvas
        red background, a blue "logo" band across the top, and a green
        "header" text box sitting on top of that same band. The overlap
        is the point -- it's what separates "hide one layer" from "clear
        the whole box".
        """
        size = (200, 150)
        background = Image.new("RGBA", size, (200, 0, 0, 255))
        logo = Image.new("RGBA", size, (0, 0, 0, 0))
        logo.paste((0, 0, 200, 255), (20, 10, 180, 60))
        header = Image.new("RGBA", size, (0, 0, 0, 0))
        header.paste((0, 200, 0, 255), (30, 20, 170, 50))
        dest = self.tmp_dir / "overlapping.psd"
        save_layered_psd(
            [("background", background), ("logo", logo), ("header", header)],
            size,
            dest,
            layer_names={},
        )
        return dest

    def test_get_psd_backdrop_clears_everything_but_the_background(self):
        # The header box is the region a text override wipes. Inside it
        # the backdrop must be pure background -- no logo, no old header
        # -- so replacement text lands on the ad's own artwork instead of
        # on top of whatever was sharing its box.
        dest = self._overlapping_template_psd()
        backdrop = get_psd_backdrop(dest)
        self.assertIsNotNone(backdrop)
        colors = {backdrop.getpixel((x, y)) for x in range(30, 170, 20) for y in range(20, 50, 10)}
        self.assertEqual(colors, {(200, 0, 0)})

    def test_get_psd_layer_background_alone_leaves_the_overlapping_logo(self):
        # The contrast case, and the reason get_psd_backdrop() exists:
        # hiding just "header" still leaves the logo band inside the
        # header's own box, so text drawn there would sit on the logo.
        dest = self._overlapping_template_psd()
        clean = get_psd_layer_background(dest, "header")
        self.assertIsNotNone(clean)
        self.assertEqual(clean.getpixel((100, 35)), (0, 0, 200))

    def test_get_psd_backdrop_returns_none_without_a_background_layer(self):
        # Nothing to keep means no usable backdrop -- callers fall back to
        # the single-layer clean-up rather than wiping a box to nothing.
        size = (200, 150)
        logo = Image.new("RGBA", size, (0, 0, 200, 255))
        dest = self.tmp_dir / "no-background.psd"
        save_layered_psd([("logo", logo)], size, dest, layer_names={})
        self.assertIsNone(get_psd_backdrop(dest))

    REAL_TEMPLATE = Path(__file__).resolve().parent.parent / "default_templates" / "tester-1080x1080.psd"

    @unittest.skipUnless(
        REAL_TEMPLATE.is_file(),
        "needs a real template with a live type layer -- the synthetic fixtures have none",
    )
    def test_set_type_layer_colors_recolours_live_text_in_place(self):
        # Recolouring a type layer has to leave it editable text. The
        # colour lives in the text engine data, so this checks it both
        # survives a save and doesn't cost the layer its text.
        dest = self.tmp_dir / "recoloured.psd"
        shutil.copy(self.REAL_TEMPLATE, dest)

        before = PSDImage.open(dest)
        original = [l for l in before if l.kind == "type" and l.name == "description"][0]
        original_text = original.text

        recoloured = set_type_layer_colors(dest, {"description": (255, 0, 0)})
        self.assertEqual(recoloured, ["description"])

        after = PSDImage.open(dest)
        layer = [l for l in after if l.name == "description"][0]
        self.assertEqual(layer.kind, "type", "recolouring must not rasterize the layer")
        self.assertEqual(layer.text, original_text, "the words must survive untouched")
        fill = layer.engine_dict["StyleRun"]["RunArray"][0]["StyleSheet"]["StyleSheetData"]["FillColor"]
        # [alpha, r, g, b] as 0..1 floats.
        self.assertEqual([round(float(v), 3) for v in fill["Values"]], [1.0, 1.0, 0.0, 0.0])

    @unittest.skipUnless(REAL_TEMPLATE.is_file(), "needs a real template")
    def test_set_type_layer_text_rewrites_the_copy_in_place(self):
        # The source-template PSD shipped beside every render was a
        # straight copy of the template, so it opened showing the
        # template's placeholder words whatever had been typed into the
        # form -- the one download made to be edited was the one that
        # didn't have the edits.
        dest = self.tmp_dir / "retyped.psd"
        shutil.copy(self.REAL_TEMPLATE, dest)

        rewritten = set_type_layer_text(dest, {"description": "Half price this week"})
        self.assertEqual(rewritten, ["description"])

        after = PSDImage.open(dest)
        layer = [l for l in after if l.name == "description"][0]
        self.assertEqual(layer.kind, "type", "rewriting must not rasterize the layer")
        self.assertEqual(layer.text.rstrip("\x00"), "Half price this week")

    @unittest.skipUnless(REAL_TEMPLATE.is_file(), "needs a real template")
    def test_set_type_layer_text_reaches_a_label_inside_a_group(self):
        # A designer's CTA is a group -- a vector rectangle with its
        # label on top -- so the layer holding the words is a child
        # called something like "Click", not the "cta" the form and the
        # render both address it by. A flat scan of the document's top
        # level never sees it.
        dest = self.tmp_dir / "group-label.psd"
        shutil.copy(self.REAL_TEMPLATE, dest)

        def type_layers(node):
            for layer in node:
                if layer.kind == "type":
                    yield layer
                elif layer.is_group():
                    yield from type_layers(layer)

        groups = [l for l in PSDImage.open(dest) if l.is_group() and l.name.strip().lower() == "cta"]
        if not groups or not list(type_layers(groups[0])):
            self.skipTest("this template's CTA is not a group with a live label")

        rewritten = set_type_layer_text(dest, {"cta": "Shop now"})
        self.assertTrue(rewritten, "the group's label was never found")

        labels = [l for l in type_layers(PSDImage.open(dest)) if l.name in rewritten]
        self.assertTrue(labels)
        self.assertEqual(labels[0].text.rstrip("\x00"), "Shop now")

    @unittest.skipUnless(REAL_TEMPLATE.is_file(), "needs a real template")
    def test_retyped_layers_keep_their_run_lengths_consistent(self):
        # A style/paragraph run whose character count doesn't match the
        # string is exactly what makes Photoshop call a file damaged, so
        # this is the assertion that the download still opens.
        dest = self.tmp_dir / "runs.psd"
        shutil.copy(self.REAL_TEMPLATE, dest)
        set_type_layer_text(dest, {"description": "A much, much longer line than before"})

        layer = [l for l in PSDImage.open(dest) if l.name == "description"][0]
        engine = layer.engine_dict
        length = len(engine["Editor"]["Text"].value)
        for key in ("StyleRun", "ParagraphRun"):
            lengths = [int(v) for v in engine[key]["RunLengthArray"]]
            self.assertEqual(
                sum(lengths), length, f"{key} lengths must sum to the string's length"
            )
            self.assertEqual(
                len(lengths), len(engine[key]["RunArray"]), f"{key} array lengths differ"
            )

    @unittest.skipUnless(REAL_TEMPLATE.is_file(), "needs a real template")
    def test_set_type_layer_text_leaves_blank_fields_alone(self):
        # Nothing typed means "keep the template's own copy", not
        # "empty this layer".
        dest = self.tmp_dir / "blank.psd"
        shutil.copy(self.REAL_TEMPLATE, dest)
        before = [l for l in PSDImage.open(dest) if l.name == "description"][0].text
        self.assertEqual(set_type_layer_text(dest, {"description": None, "header": ""}), [])
        after = [l for l in PSDImage.open(dest) if l.name == "description"][0].text
        self.assertEqual(after, before)

    def test_set_type_layer_text_on_an_unreadable_file_is_a_no_op(self):
        junk = self.tmp_dir / "not-a-text.psd"
        junk.write_bytes(b"nope")
        self.assertEqual(set_type_layer_text(junk, {"description": "hi"}), [])

    @unittest.skipUnless(REAL_TEMPLATE.is_file(), "needs a real template")
    def test_set_type_layer_colors_ignores_layers_it_was_not_asked_about(self):
        dest = self.tmp_dir / "untouched.psd"
        shutil.copy(self.REAL_TEMPLATE, dest)
        self.assertEqual(set_type_layer_colors(dest, {"logo": (255, 0, 0)}), [])

    def test_set_type_layer_colors_on_an_unreadable_file_is_a_no_op(self):
        # Best-effort by design: this only decorates a download that is
        # already correct, so a bad file returns [] rather than raising.
        junk = self.tmp_dir / "not-a.psd"
        junk.write_bytes(b"nope")
        self.assertEqual(set_type_layer_colors(junk, {"description": (1, 2, 3)}), [])

    def _template_shaped_layers(self, size):
        """A render stack named to match the real template's own layers,
        so the preserved type layer has somewhere to slot back into."""
        width, height = size
        def block(color):
            image = Image.new("RGBA", size, (0, 0, 0, 0))
            image.paste(color, (5, 5, width // 2, height // 2))
            return image

        return [
            ("Background", Image.new("RGBA", size, (20, 30, 60, 255))),
            ("background", block((40, 80, 140, 255))),
            ("product", block((200, 120, 40, 255))),
            ("cta", block((240, 60, 60, 255))),
            ("description", block((0, 255, 0, 255))),
            ("logo", block((255, 255, 255, 255))),
        ]

    @unittest.skipUnless(
        REAL_TEMPLATE.is_file(),
        "needs a real template with a live type layer -- the synthetic fixtures have none",
    )
    def test_preserving_type_keeps_the_description_editable(self):
        # The whole point of the preserving export: the downloaded PSD's
        # description has to open in Photoshop as characters somebody can
        # retype, not as a picture of characters.
        template = PSDImage.open(self.REAL_TEMPLATE)
        size = (template.width, template.height)
        dest = self.tmp_dir / "live.psd"

        kept = save_layered_psd_preserving_type(
            self._template_shaped_layers(size),
            size,
            dest,
            template_path=self.REAL_TEMPLATE,
            preserve_text={"description": "Limited drop: 500 free cans"},
            layer_names={},
        )
        self.assertEqual(kept, ["description"])

        after = PSDImage.open(dest)
        layers = {l.name: l for l in after}
        self.assertEqual(
            layers["description"].kind,
            "type",
            "the description must survive as live type, not pixels",
        )
        self.assertEqual(layers["description"].text, "Limited drop: 500 free cans")
        # Everything else is still ordinary pixel art, and the stack is
        # intact rather than partially rebuilt.
        self.assertEqual(layers["logo"].kind, "pixel")
        self.assertEqual([l.name for l in after], [n for n, _ in self._template_shaped_layers(size)])

    @unittest.skipUnless(REAL_TEMPLATE.is_file(), "needs a real template")
    def test_preserving_type_without_an_override_keeps_the_templates_own_words(self):
        template = PSDImage.open(self.REAL_TEMPLATE)
        size = (template.width, template.height)
        original = [l for l in template if l.kind == "type" and l.name == "description"][0].text
        dest = self.tmp_dir / "unchanged.psd"

        kept = save_layered_psd_preserving_type(
            self._template_shaped_layers(size),
            size,
            dest,
            template_path=self.REAL_TEMPLATE,
            preserve_text={"description": None},
            layer_names={},
        )
        self.assertEqual(kept, ["description"])
        layer = [l for l in PSDImage.open(dest) if l.name == "description"][0]
        self.assertEqual(layer.kind, "type")
        self.assertEqual(layer.text, original)

    def test_preserving_type_falls_back_to_a_plain_export_without_a_type_layer(self):
        # A template whose text is already flattened art (every CTA in
        # default_templates today) can't give the export a type layer
        # back. That has to degrade to the ordinary pixel export rather
        # than losing the PSD download altogether.
        size = (120, 90)
        flat = self.tmp_dir / "flat-template.psd"
        save_layered_psd([("description", Image.new("RGBA", size, (10, 10, 10, 255)))], size, flat, layer_names={})
        dest = self.tmp_dir / "fallback.psd"

        kept = save_layered_psd_preserving_type(
            [("Background", Image.new("RGBA", size, (1, 2, 3, 255)))],
            size,
            dest,
            template_path=flat,
            preserve_text={"description": "nope"},
            layer_names={},
        )
        self.assertEqual(kept, [])
        self.assertTrue(dest.is_file(), "the PSD download must still be written")
        self.assertEqual([l.name for l in PSDImage.open(dest)], ["Background"])

    def test_preserving_type_falls_back_when_the_template_is_unreadable(self):
        size = (80, 60)
        junk = self.tmp_dir / "not-a.psd"
        junk.write_bytes(b"nope")
        dest = self.tmp_dir / "fallback2.psd"
        kept = save_layered_psd_preserving_type(
            [("Background", Image.new("RGBA", size, (9, 9, 9, 255)))],
            size,
            dest,
            template_path=junk,
            preserve_text={"description": "x"},
            layer_names={},
        )
        self.assertEqual(kept, [])
        self.assertTrue(dest.is_file())

    HEADER_TEMPLATE = Path(__file__).resolve().parent.parent / "default_templates" / "tester-1080x1080.psd"

    @unittest.skipUnless(HEADER_TEMPLATE.is_file(), "needs a template with a live header type layer")
    def test_preserving_type_keeps_header_and_description_editable_together(self):
        # Six of the seven shipped templates carry both a header and a
        # description as live type; preserving one must not cost the
        # other, and each has to end up saying its own override.
        template = PSDImage.open(self.HEADER_TEMPLATE)
        size = (template.width, template.height)
        names = [l.name for l in template]
        layers = [
            (name, Image.new("RGBA", size, (30 + i * 20, 40, 60, 255)))
            for i, name in enumerate(names)
        ]
        dest = self.tmp_dir / "both.psd"

        kept = save_layered_psd_preserving_type(
            layers,
            size,
            dest,
            template_path=self.HEADER_TEMPLATE,
            preserve_text={"description": "Body copy here", "header": "Big Headline"},
            layer_names={},
        )
        self.assertEqual(sorted(kept), ["description", "header"])

        after = {l.name: l for l in PSDImage.open(dest)}
        self.assertEqual(after["header"].kind, "type")
        self.assertEqual(after["header"].text, "Big Headline")
        self.assertEqual(after["description"].kind, "type")
        self.assertEqual(after["description"].text, "Body copy here")
        self.assertEqual(after["logo"].kind, "pixel")

    @unittest.skipUnless(REAL_TEMPLATE.is_file(), "needs a real template")
    def test_preserving_type_skips_a_name_that_is_not_a_type_layer(self):
        # "logo" is a pixel layer in every template -- asking to preserve
        # it as live text has to be quietly ignored rather than dropping
        # the layer or failing the export, so a caller can name the whole
        # set without checking each template first.
        template = PSDImage.open(self.REAL_TEMPLATE)
        size = (template.width, template.height)
        names = [l.name for l in template]
        self.assertIn("logo", [n.lower() for n in names])
        layers = [(n, Image.new("RGBA", size, (5, 5, 5, 255))) for n in names]
        dest = self.tmp_dir / "skips-non-type.psd"

        kept = save_layered_psd_preserving_type(
            layers, size, dest,
            template_path=self.REAL_TEMPLATE,
            preserve_text={"description": "Still works", "logo": "Ignored"},
            layer_names={},
        )
        self.assertEqual(kept, ["description"])
        after = {l.name.lower(): l for l in PSDImage.open(dest)}
        self.assertEqual(after["description"].text, "Still works")
        self.assertEqual(after["logo"].kind, "pixel", "logo must stay ordinary art")

    def test_a_vector_shape_layer_does_not_cost_us_the_other_layers(self):
        # A CTA built as a group holding a rectangle puts a VECTOR SHAPE
        # layer in the template. psd-tools can only draw one with aggdraw
        # installed, and it raises the moment a composite has to be
        # re-rendered -- which is every per-layer operation here, since
        # isolating a layer means toggling visibility. Both call sites
        # caught that silently: the foreground came back as None so a
        # replaced background restored nothing over itself, and the stack
        # simply lost the layer. The creative arrived as a bare backdrop.
        from psd_tools import PSDImage

        from src.image_ops import get_psd_layer_foreground, get_psd_layer_stack

        psd = PSDImage.open(self.REAL_TEMPLATE)
        top_level = [l.name.strip().lower() for l in psd if (l.name or "").strip()]
        has_shape = any(
            child.kind == "shape"
            for layer in psd
            if layer.is_group()
            for child in layer
        )
        if not has_shape:
            self.skipTest("template has no vector shape layer to exercise this")

        self.assertIsNotNone(
            get_psd_layer_foreground(self.REAL_TEMPLATE, "background"),
            "no foreground to restore -- a replaced background will wipe every "
            "other layer (is aggdraw installed?)",
        )
        stack = get_psd_layer_stack(self.REAL_TEMPLATE)
        self.assertIsNotNone(stack)
        self.assertEqual(
            [name.strip().lower() for name, _img in stack],
            top_level,
            "every top-level layer must survive isolation, groups included",
        )

    def test_background_override_can_fit_instead_of_crop(self):
        # Cropping to fill is right for a texture and wrong for artwork
        # the model laid out: it takes the end off a generated headline.
        # "contain" has to keep the whole image, top and bottom included.
        from src.image_ops import apply_layer_background_override

        art = Image.new("RGB", (300, 300), (10, 10, 10))
        art.paste((255, 0, 0), (0, 0, 300, 20))       # a band at the very top
        art.paste((0, 0, 255), (0, 280, 300, 300))    # and one at the very bottom
        base = Image.new("RGB", (400, 200), (9, 9, 9))

        cropped = apply_layer_background_override(base, (0, 0, 400, 200), art)
        contained = apply_layer_background_override(
            base, (0, 0, 400, 200), art, fit="contain"
        )
        self.assertEqual(contained.size, (400, 200))

        def has(image, colour):
            return any(
                abs(p[0] - colour[0]) + abs(p[1] - colour[1]) + abs(p[2] - colour[2]) < 40
                for p in image.convert("RGB").getdata()
            )

        # A square fitted into a wide box keeps both edge bands...
        self.assertTrue(has(contained, (255, 0, 0)), "top of the artwork was lost")
        self.assertTrue(has(contained, (0, 0, 255)), "bottom of the artwork was lost")
        # ...where filling the same box crops them away.
        self.assertFalse(has(cropped, (255, 0, 0)))
        self.assertFalse(has(cropped, (0, 0, 255)))

    def test_text_and_cta_overrides_take_a_stroke(self):
        # The outline is a percentage of the type size, not a pixel
        # width: one setting runs at every output size, and 3px that
        # frames 176px type at 1920x1080 swallows the 30px it becomes at
        # 160x600.
        from src.image_ops import apply_layer_cta_override, apply_layer_text_override

        base = Image.new("RGB", (400, 200), (120, 120, 120))
        plain = apply_layer_text_override(base, (20, 20, 380, 120), "Hello")
        outlined = apply_layer_text_override(
            base, (20, 20, 380, 120), "Hello", stroke_size=8, stroke_color=(255, 0, 0)
        )
        self.assertNotEqual(list(plain.getdata()), list(outlined.getdata()))
        # 0 is the default and has to be a true no-op.
        self.assertEqual(
            list(plain.getdata()),
            list(apply_layer_text_override(base, (20, 20, 380, 120), "Hello", stroke_size=0).getdata()),
        )

        cta_plain = apply_layer_cta_override(base, (20, 140, 380, 190), "Go")
        cta_outlined = apply_layer_cta_override(
            base, (20, 140, 380, 190), "Go", stroke_size=10, stroke_color=(255, 0, 0)
        )
        self.assertNotEqual(list(cta_plain.getdata()), list(cta_outlined.getdata()))

    def test_a_stroke_scales_with_the_size_it_lands_on(self):
        # Same percentage, bigger box: the outline has to grow with the
        # type rather than stay a fixed number of pixels.
        from src.image_ops import apply_layer_text_override

        def outlined_pixels(size):
            base = Image.new("RGB", size, (255, 255, 255))
            out = apply_layer_text_override(
                base, (0, 0, size[0], size[1]), "Hi",
                text_color=(255, 255, 255), stroke_size=10, stroke_color=(255, 0, 0),
            )
            return sum(1 for p in out.convert("RGB").getdata() if p[0] > 180 and p[1] < 80)

        small = outlined_pixels((160, 60))
        large = outlined_pixels((640, 240))
        self.assertGreater(large, small * 4, f"stroke did not scale: {small} -> {large}")

    def test_the_cta_button_takes_a_border_and_a_corner_radius(self):
        # The button is the background rectangle of a CTA group, and it
        # gets its own styling: fill, outline and how round the corners
        # are. An empty label is legitimate here -- the words are set
        # separately from the group's own text layer.
        from src.image_ops import apply_layer_cta_override

        base = Image.new("RGB", (300, 120), (255, 255, 255))

        def corner(radius):
            out = apply_layer_cta_override(
                base, (20, 20, 280, 100), "", button_color=(0, 87, 184), corner_radius=radius
            )
            return out.getpixel((22, 22))

        # 0 squares the corners off, so the corner pixel is the button.
        self.assertEqual(corner(0), (0, 87, 184))
        # Left alone it is a pill, and the corner is whatever was behind.
        self.assertEqual(corner(None), (255, 255, 255))
        self.assertEqual(corner(25), (255, 255, 255))

        plain = apply_layer_cta_override(base, (20, 20, 280, 100), "", button_color=(0, 87, 184))
        bordered = apply_layer_cta_override(
            base, (20, 20, 280, 100), "", button_color=(0, 87, 184),
            border_size=10, border_color=(255, 255, 255),
        )
        self.assertNotEqual(list(plain.getdata()), list(bordered.getdata()))

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
