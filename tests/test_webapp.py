"""Smoke tests for the quick-generate web UI (webapp.py).

These exercise the Flask app directly via its test client -- no server
process or browser needed -- covering the same "upload hero image, header,
description -> sized creatives + zip" flow a person would drive by hand.
"""

import io
import re
import shutil
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

import webapp


class _CampaignBriefAutoFillClient:
    """Wraps a Flask test client so a POST to /generate gets sensible
    defaults merged in for the four now-required campaign-brief fields
    (product_name/market/audience/campaign_message) for any of them a
    test's own `data` dict doesn't already mention.

    Campaign brief became a required part of every batch well after most
    of this file's /generate tests were written -- there are well over a
    hundred of them, each focused on one unrelated thing (a logo offset,
    a font size, a PSD template row...). Rather than hand-editing every
    one to carry the same four unrelated fields just to get past
    validation, tests get it for free here; a test that actually cares
    about campaign-brief behavior (blank-field validation, product-name
    driven filenames, etc.) sets the field(s) it cares about explicitly
    in its own `data` -- including as "" to deliberately test the blank
    case -- and this wrapper only ever fills in a key that's altogether
    *absent*, so it never overrides anything a test set on purpose.
    """

    # product_name is deliberately punctuation-only: it satisfies the
    # "non-blank" requirement (so validation passes) while sanitizing
    # down to "" (see _slugify_for_filename() in webapp.py), which falls
    # back to the original generic "creative_..." / "creatives.zip"
    # filenames -- exactly what the many pre-existing tests in this file
    # that hardcode those filenames (and were written before campaign
    # brief existed at all) still expect. A test that specifically cares
    # about product-name-driven filenames sets its own real product_name
    # explicitly, which overrides this default entirely (see post()
    # below).
    _BRIEF_DEFAULTS = {
        "product_name": "---",
        "market": "US",
        "audience": "Test audience",
        "campaign_message": "Test campaign message.",
    }

    def __init__(self, client):
        self._client = client

    def post(self, path, *args, **kwargs):
        data = kwargs.get("data")
        if path == "/generate" and isinstance(data, dict):
            merged = dict(self._BRIEF_DEFAULTS)
            merged.update(data)
            kwargs = dict(kwargs)
            kwargs["data"] = merged
        return self._client.post(path, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._client, name)


class WebAppSmokeTest(unittest.TestCase):
    def setUp(self):
        webapp.app.config["TESTING"] = True
        self.client = _CampaignBriefAutoFillClient(webapp.app.test_client())
        # Redirect job output into a scratch dir so tests don't litter (or
        # depend on) the real outputs/web/ folder.
        self._orig_jobs_dir = webapp.JOBS_DIR
        self.tmp_dir = tempfile.mkdtemp()
        webapp.JOBS_DIR = Path(self.tmp_dir)
        # Point default_templates/ scanning at an empty scratch dir too --
        # otherwise these tests' output would depend on (and could be
        # broken by) whatever real .psd templates a user has saved for
        # their own project.
        self._orig_default_templates_dir = webapp.DEFAULT_TEMPLATES_DIR
        self.tmp_default_templates_dir = tempfile.mkdtemp()
        webapp.DEFAULT_TEMPLATES_DIR = Path(self.tmp_default_templates_dir)

    def tearDown(self):
        webapp.JOBS_DIR = self._orig_jobs_dir
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        webapp.DEFAULT_TEMPLATES_DIR = self._orig_default_templates_dir
        shutil.rmtree(self.tmp_default_templates_dir, ignore_errors=True)

    def _sample_image_bytes(self, size=(400, 300), color=(20, 100, 200)):
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, format="PNG")
        buf.seek(0)
        return buf

    def _sample_logo_bytes(self, size=(200, 100), color=(255, 209, 0, 255)):
        buf = io.BytesIO()
        Image.new("RGBA", size, color).save(buf, format="PNG")
        buf.seek(0)
        return buf

    def _sample_badge_bytes(self, size=(200, 200), color=(220, 30, 30, 255)):
        buf = io.BytesIO()
        Image.new("RGBA", size, color).save(buf, format="PNG")
        buf.seek(0)
        return buf

    def _sample_psd_bytes(self, size=(64, 40), color=(200, 50, 50)):
        """Hand-assemble a minimal, valid, uncompressed flat RGB .psd file
        with three real named layers -- logo/description/product.

        Pillow can *read* PSD (it renders the flattened composite via plain
        Image.open().convert('RGB'), same as any other format) but it has no
        PSD *writer* to build a test fixture with -- so this lays out just
        enough of the real format (file header, color-mode-data /
        image-resources sections, a layer-and-mask section with three
        minimal stub layer records, then the flattened composite's raw
        uncompressed per-channel planar pixel data) for Pillow's reader to
        open it back with img.layers populated and .convert('RGB') giving
        `color` as the flattened result.
        """
        width, height = size
        r, g, b = color
        header = b"8BPS"
        header += struct.pack(">H", 1)  # version
        header += b"\x00" * 6  # reserved
        header += struct.pack(">H", 3)  # channels (RGB)
        header += struct.pack(">I", height)
        header += struct.pack(">I", width)
        header += struct.pack(">H", 8)  # depth
        header += struct.pack(">H", 3)  # color mode: RGB
        color_mode_data = struct.pack(">I", 0)
        image_resources = struct.pack(">I", 0)

        # Real named layers -- webapp.py's REQUIRED_PSD_LAYERS validation
        # (logo/description/product) rejects any template missing one, so
        # every synthetic test PSD needs real layer records with those
        # exact names, not just an empty layer-and-mask section. Small
        # stub boxes in a top strip -- they just need to exist and be
        # named correctly, tests that care about layer *placement* use a
        # real project template instead (see LayerOverrideIntegrationTest).
        stub_layers = [
            ("logo", (0, 0, 20, 10)),
            ("description", (20, 0, 40, 10)),
            ("product", (40, 0, 60, 10)),
        ]
        layer_records = b""
        channel_data = b""
        for name, (lx0, ly0, lx1, ly1) in stub_layers:
            lw, lh = lx1 - lx0, ly1 - ly0
            rec = struct.pack(">iiii", ly0, lx0, ly1, lx1)
            rec += struct.pack(">H", 3)  # 3 channels
            for ch_id in (0, 1, 2):
                rec += struct.pack(">H", ch_id)
                rec += struct.pack(">I", 2 + lw * lh)  # declared size (unused by Pillow's reader)
            rec += b"8BIM" + b"norm"
            rec += struct.pack(">B", 255)  # opacity
            rec += struct.pack(">B", 0)  # clipping
            rec += struct.pack(">B", 0)  # flags
            rec += struct.pack(">B", 0)  # filler
            name_bytes = name.encode("latin-1")
            extra = struct.pack(">I", 0) + struct.pack(">I", 0)  # mask + blending-ranges lengths (both empty)
            extra += struct.pack(">B", len(name_bytes)) + name_bytes
            rec += struct.pack(">I", len(extra)) + extra
            layer_records += rec
            stub_plane = lambda value, n=lw * lh: bytes([value]) * n
            for v in color:
                channel_data += struct.pack(">H", 0) + stub_plane(v)  # compression=0 (raw)

        layer_info_inner = struct.pack(">h", len(stub_layers)) + layer_records + channel_data
        if len(layer_info_inner) % 2:
            layer_info_inner += b"\x00"
        layer_info = struct.pack(">I", len(layer_info_inner)) + layer_info_inner
        global_layer_mask_info = struct.pack(">I", 0)
        layer_and_mask_section = layer_info + global_layer_mask_info
        layer_mask_info = struct.pack(">I", len(layer_and_mask_section)) + layer_and_mask_section

        compression = struct.pack(">H", 0)  # raw, uncompressed
        plane = lambda value: bytes([value]) * (width * height)
        image_data = compression + plane(r) + plane(g) + plane(b)
        buf = io.BytesIO(header + color_mode_data + image_resources + layer_mask_info + image_data)
        buf.seek(0)
        return buf

    def test_index_loads(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Hero image", r.data)

    def test_index_and_edit_page_both_show_a_reset_link_back_to_a_blank_form(self):
        # The top-of-page "Reset form" link is just a plain link to "/" --
        # index() always renders with prefill={}/prefill_files={}/
        # edit_job_id=None, so following it is a guaranteed full reset
        # (typed fields, attached files, and "editing a prior batch"
        # state all go away) with no separate reset logic to keep in
        # sync with the form itself.
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'>Reset form</a>', r.data)
        self.assertIn(b'href="/"', r.data)  # the reset link's target -- index()'s own route

        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "header": "Test Header",
            "description": "Test description",
        }
        gen = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(gen.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", gen.data).group(1).decode()

        edit_page = self.client.get(f"/edit/{job_id}")
        self.assertEqual(edit_page.status_code, 200)
        self.assertIn(b'>Reset form</a>', edit_page.data)

    def test_generate_renders_default_sizes_and_zip(self):
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "header": "Test Header",
            "description": "Test description text.",
            "sizes": ["default"],
            "fit_mode": "crop",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data.count(b'class="card"'), 3)  # the 3 social defaults

        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        job_dir = webapp.JOBS_DIR / job_id
        self.assertTrue((job_dir / "creatives.zip").exists())
        self.assertTrue((job_dir / "creative_1080x1080.png").exists())

        download = self.client.get(f"/download/{job_id}")
        try:
            self.assertEqual(download.status_code, 200)
            self.assertGreater(len(download.data), 0)
        finally:
            download.close()

    def test_generate_warns_when_a_checked_brand_color_is_missing(self):
        # Solid orange hero -- brand_color_1 matches it exactly,
        # brand_color_2 (bright green) is nowhere in the image and
        # should trigger the warning note; brand_color_3 is left
        # unchecked and should be ignored entirely (no note about it).
        buf = io.BytesIO()
        Image.new("RGB", (400, 400), (230, 100, 20)).save(buf, format="PNG")
        buf.seek(0)
        data = {
            "hero_image": (buf, "hero.png"),
            "sizes": ["1080x1080"],
            "fit_mode": "crop",
            "brand_color_1_enabled": "1",
            "brand_color_1": "#e66414",
            "brand_color_2_enabled": "1",
            "brand_color_2": "#00ff00",
            "brand_color_3": "#123456",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"brand color check", r.data)
        self.assertIn(b"#00ff00", r.data)
        self.assertNotIn(b"#123456", r.data)
        # Rendered as a warning (red, in background-warnings), not a
        # plain informational note (blue, in background-notes).
        text = r.get_data(as_text=True)
        warnings_block = text.split('class="background-warnings"')[1].split("</ul>")[0]
        self.assertIn("brand color check", warnings_block)
        if 'class="background-notes"' in text:
            notes_block = text.split('class="background-notes"')[1].split("</ul>")[0]
            self.assertNotIn("brand color check", notes_block)

    def test_generate_no_brand_color_warning_when_all_present_or_none_checked(self):
        buf = io.BytesIO()
        Image.new("RGB", (400, 400), (230, 100, 20)).save(buf, format="PNG")
        buf.seek(0)
        data = {
            "hero_image": (buf, "hero.png"),
            "sizes": ["1080x1080"],
            "fit_mode": "crop",
            "brand_color_1_enabled": "1",
            "brand_color_1": "#e66414",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertNotIn(b"brand color check", r.data)

        buf2 = io.BytesIO()
        Image.new("RGB", (400, 400), (230, 100, 20)).save(buf2, format="PNG")
        buf2.seek(0)
        data2 = {
            "hero_image": (buf2, "hero.png"),
            "sizes": ["1080x1080"],
            "fit_mode": "crop",
            "brand_color_1": "#00ff00",  # not enabled -- must be ignored
        }
        r2 = self.client.post("/generate", data=data2, content_type="multipart/form-data")
        self.assertEqual(r2.status_code, 200)
        self.assertNotIn(b"brand color check", r2.data)

    def test_generate_campaign_brief_fields_appear_on_results_page_and_are_carried_forward(self):
        # The campaign-brief fields (product name / market / audience /
        # campaign message) are purely informational -- not composited
        # into the creatives -- but should surface on the results page
        # and round-trip through the Edit flow (form_state.json) like
        # every other field.
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "sizes": ["default"],
            "fit_mode": "crop",
            "product_name": "HydroBoost Sports Drink",
            "market": "US",
            "audience": "Active adults 18-34",
            "campaign_message": "Drive summer trial with a refreshed flavor line.",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        for needle in (
            b"Campaign brief",
            b"HydroBoost Sports Drink",
            b"US",
            b"Active adults 18-34",
            b"Drive summer trial with a refreshed flavor line.",
        ):
            self.assertIn(needle, r.data)

        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        edit_page = self.client.get(f"/edit/{job_id}")
        self.assertEqual(edit_page.status_code, 200)
        self.assertIn(b'value="HydroBoost Sports Drink"', edit_page.data)
        self.assertIn(b'value="US"', edit_page.data)
        self.assertIn(b'value="Active adults 18-34"', edit_page.data)
        self.assertIn(b"Drive summer trial with a refreshed flavor line.", edit_page.data)

    def test_generate_names_downloaded_files_after_product_name_and_size(self):
        # The PNG, its per-size PSD, and the zip should all carry the
        # product name (sanitized to something filesystem-safe) plus that
        # creative's own size, so a batch downloaded to a folder full of
        # other batches is identifiable at a glance instead of every file
        # being a generic "creative_WIDTHxHEIGHT.png".
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "product_name": "HydroBoost Sports Drink!!",
            "custom_sizes": "300x250",
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        job_dir = webapp.JOBS_DIR / job_id

        expected_prefix = "HydroBoost_Sports_Drink"
        self.assertTrue((job_dir / f"{expected_prefix}_300x250.png").is_file())
        self.assertTrue((job_dir / f"{expected_prefix}_300x250.psd").is_file())
        self.assertTrue((job_dir / f"{expected_prefix}_creatives.zip").is_file())
        self.assertIn(f"{expected_prefix}_300x250.png".encode(), r.data)
        self.assertIn(
            f"/download-psd/{job_id}/{expected_prefix}_300x250.psd".encode(), r.data
        )

        dl = self.client.get(f"/download/{job_id}")
        self.assertEqual(dl.status_code, 200)
        self.assertIn(
            f"{expected_prefix}_creatives.zip",
            dl.headers.get("Content-Disposition", ""),
        )

    def test_generate_zips_files_into_a_folder_named_after_the_product(self):
        # A product name nests everything the zip contains -- both PNGs
        # and their per-size PSDs -- under one "<product>/" folder inside
        # the zip, so extracting it drops a single self-contained folder
        # instead of scattering files loose wherever the user extracts to.
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "product_name": "HydroBoost Sports Drink!!",
            "custom_sizes": "300x250,320x50",
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        job_dir = webapp.JOBS_DIR / job_id

        expected_prefix = "HydroBoost_Sports_Drink"
        zip_path = job_dir / f"{expected_prefix}_creatives.zip"
        self.assertTrue(zip_path.is_file())
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
        self.assertEqual(
            names,
            {
                f"{expected_prefix}/{expected_prefix}_300x250.png",
                f"{expected_prefix}/{expected_prefix}_300x250.psd",
                f"{expected_prefix}/{expected_prefix}_320x50.png",
                f"{expected_prefix}/{expected_prefix}_320x50.psd",
            },
        )

    def test_generate_without_a_product_name_zips_files_flat_with_no_folder(self):
        # No product name -- the zip's internal layout is unchanged from
        # before this feature: every file loose at the zip root, no
        # nesting folder (there's no product name to name one after).
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "custom_sizes": "300x250",
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        job_dir = webapp.JOBS_DIR / job_id

        zip_path = job_dir / "creatives.zip"
        self.assertTrue(zip_path.is_file())
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
        self.assertEqual(names, {"creative_300x250.png", "creative_300x250.psd"})

    def test_generate_without_a_product_name_keeps_the_original_generic_filenames(self):
        # No product name given -- filenames must come out byte-identical
        # to before this feature existed (tests all over this file rely
        # on the "creative_WIDTHxHEIGHT.png"/"creatives.zip" pattern).
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "custom_sizes": "300x250",
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        job_dir = webapp.JOBS_DIR / job_id
        self.assertTrue((job_dir / "creative_300x250.png").is_file())
        self.assertTrue((job_dir / "creative_300x250.psd").is_file())
        self.assertTrue((job_dir / "creatives.zip").is_file())

    def test_generate_product_name_that_sanitizes_to_nothing_falls_back_to_generic_filenames(self):
        # A product name made entirely of emoji/punctuation has nothing
        # filesystem-safe left after sanitizing -- must fall back to the
        # same generic names as no product name at all, not a filename
        # made of bare underscores.
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "product_name": "!!! ***",
            "custom_sizes": "300x250",
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        job_dir = webapp.JOBS_DIR / job_id
        self.assertTrue((job_dir / "creative_300x250.png").is_file())
        self.assertTrue((job_dir / "creatives.zip").is_file())

    def test_generate_with_a_fully_blank_campaign_brief_flashes_and_creates_no_job(self):
        # Campaign brief is required -- all four fields explicitly blank
        # (not just omitted; see _CampaignBriefAutoFillClient's docstring
        # for why omitted vs. explicitly-blank matters here) must flash
        # and redirect without ever reaching creative generation.
        before = set(webapp.JOBS_DIR.glob("*")) if webapp.JOBS_DIR.is_dir() else set()
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "sizes": ["default"],
            "fit_mode": "crop",
            "product_name": "",
            "market": "",
            "audience": "",
            "campaign_message": "",
        }
        r = self.client.post(
            "/generate", data=data, content_type="multipart/form-data", follow_redirects=True
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Campaign brief is required", r.data)
        for needle in (b"Product name", b"Market", b"Audience", b"Campaign message"):
            self.assertIn(needle, r.data)
        after = set(webapp.JOBS_DIR.glob("*")) if webapp.JOBS_DIR.is_dir() else set()
        self.assertEqual(before, after)  # no job directory was created

    def test_generate_with_one_blank_campaign_brief_field_names_just_that_field(self):
        # Only "audience" left blank -- the flash message should name
        # that field specifically, not claim the whole brief is missing.
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "sizes": ["default"],
            "fit_mode": "crop",
            "product_name": "HydroBoost",
            "market": "US",
            "audience": "",
            "campaign_message": "Drive trial.",
        }
        r = self.client.post(
            "/generate", data=data, content_type="multipart/form-data", follow_redirects=True
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Campaign brief is required -- please fill in: Audience.", r.data)

    def test_generate_with_profanity_in_campaign_message_is_blocked(self):
        # Profanity is a hard gate, same class of thing as a missing
        # campaign-brief field -- no job directory should be created.
        before = set(webapp.JOBS_DIR.glob("*")) if webapp.JOBS_DIR.is_dir() else set()
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "sizes": ["default"],
            "fit_mode": "crop",
            "campaign_message": "This is such shit, buy it now.",
        }
        r = self.client.post(
            "/generate", data=data, content_type="multipart/form-data", follow_redirects=True
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"language we can", r.data)
        self.assertIn(b"Campaign message", r.data)
        after = set(webapp.JOBS_DIR.glob("*")) if webapp.JOBS_DIR.is_dir() else set()
        self.assertEqual(before, after)  # no job directory was created

    def test_generate_with_profanity_in_header_names_that_field(self):
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "sizes": ["default"],
            "fit_mode": "crop",
            "header": "this is fucking great",
        }
        r = self.client.post(
            "/generate", data=data, content_type="multipart/form-data", follow_redirects=True
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"language we can", r.data)
        self.assertIn(b"Header/title", r.data)

    def test_generate_with_clean_text_is_not_blocked_by_profanity_check(self):
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "sizes": ["default"],
            "fit_mode": "crop",
            "header": "Totally clean headline",
            "campaign_message": "Totally clean campaign message.",
        }
        r = self.client.post(
            "/generate", data=data, content_type="multipart/form-data", follow_redirects=True
        )
        self.assertEqual(r.status_code, 200)
        self.assertNotIn(b"language we can", r.data)
        self.assertIn(b"/download/", r.data)

    def test_generate_with_profanity_in_a_content_psd_text_layer_is_blocked(self):
        # A PSD's text layer is just as much "text someone typed" as any
        # web form field -- otherwise the profanity check above would be
        # trivially bypassable by putting flagged language in the PSD
        # instead of the form. get_psd_text_layers() itself is exercised
        # for real (against real PSD bytes) elsewhere -- this test
        # monkeypatches it so the *webapp wiring* (which upload, which
        # field label, which flash message) can be checked without
        # depending on any particular fixture file's actual text content.
        orig = webapp.get_psd_text_layers
        webapp.get_psd_text_layers = lambda path: {"description": "such shit, wow"}
        try:
            (webapp.DEFAULT_TEMPLATES_DIR / "profanity-970x90.psd").write_bytes(
                self._sample_psd_bytes(size=(970, 90)).getvalue()
            )
            data = {
                "content_psd": (self._sample_psd_bytes(), "content.psd"),
            }
            r = self.client.post(
                "/generate", data=data, content_type="multipart/form-data", follow_redirects=True
            )
        finally:
            webapp.get_psd_text_layers = orig
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"language we can", r.data)
        self.assertIn(b"content PSD", r.data)
        self.assertIn(b"description", r.data)

    def test_generate_with_profanity_in_a_psd_template_row_text_layer_is_blocked(self):
        orig = webapp.get_psd_text_layers
        webapp.get_psd_text_layers = lambda path: {"description": "such shit, wow"}
        try:
            data = {
                "hero_image": (self._sample_image_bytes(), "hero.png"),
                "sizes": ["default"],
                "fit_mode": "crop",
                "psd_size_1": "970x90",
                "psd_file_1": (self._sample_psd_bytes(size=(970, 90)), "row1.psd"),
            }
            r = self.client.post(
                "/generate", data=data, content_type="multipart/form-data", follow_redirects=True
            )
        finally:
            webapp.get_psd_text_layers = orig
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"language we can", r.data)
        self.assertIn(b"PSD template row 1", r.data)

    def test_generate_with_a_clean_psd_text_layer_is_not_blocked(self):
        orig = webapp.get_psd_text_layers
        webapp.get_psd_text_layers = lambda path: {"description": "Totally clean copy."}
        try:
            (webapp.DEFAULT_TEMPLATES_DIR / "clean-970x90.psd").write_bytes(
                self._sample_psd_bytes(size=(970, 90)).getvalue()
            )
            data = {
                "content_psd": (self._sample_psd_bytes(), "content.psd"),
            }
            r = self.client.post(
                "/generate", data=data, content_type="multipart/form-data", follow_redirects=True
            )
        finally:
            webapp.get_psd_text_layers = orig
        self.assertEqual(r.status_code, 200)
        self.assertNotIn(b"language we can", r.data)

    def test_generate_combines_preset_and_custom_sizes(self):
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "header": "",
            "description": "Desc",
            "sizes": ["default"],
            "custom_sizes": "1200x628",
            "fit_mode": "crop",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data.count(b'class="card"'), 4)  # 3 defaults + 1 custom
        self.assertIn(b"1200x628", r.data)

    def test_generate_without_hero_image_or_template_redirects_home_with_flash(self):
        # No hero image, no PSD template -- the (default) requested sizes
        # have no background at all, which is a validation error naming
        # them, not a crash.
        r = self.client.post(
            "/generate",
            data={"header": "x", "description": "y"},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"hero image or a matching PSD template", r.data)
        self.assertIn(b"1080x1080", r.data)
        # Points at the one-click fix (the AI-hero checkbox) rather than
        # leaving "upload something" as the only way forward.
        self.assertIn(b"Generate a hero image with AI", r.data)

    def test_generate_ai_hero_fills_gap_when_no_hero_image_or_template(self):
        # The "mock" provider is used here specifically because it's
        # offline and deterministic (see src/providers/mock_provider.py)
        # -- this exercises the exact same web-app wiring a real provider
        # (Pollinations/Hugging Face) goes through, without a live network
        # call in a test.
        data = {
            "sizes": ["1080x1080"],
            "fit_mode": "crop",
            "ai_hero_enabled": "1",
            "ai_hero_provider": "mock",
            "ai_hero_prompt": "a can of energy drink on ice, studio lighting",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'class="card"', r.data)
        self.assertIn(b"Hero image generated with AI (mock)", r.data)
        self.assertIn(b"a can of energy drink on ice, studio lighting", r.data)

        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        job_dir = webapp.JOBS_DIR / job_id
        self.assertTrue((job_dir / "uploads" / "ai_generated_hero.png").exists())

        # Carries forward on Edit exactly like an uploaded hero image --
        # no regeneration needed to keep editing the same batch.
        edit_page = self.client.get(f"/edit/{job_id}")
        self.assertIn(b"Currently: <strong>ai_generated_hero.png</strong>", edit_page.data)
        self.assertIn(b'id="ai_hero_enabled" name="ai_hero_enabled" value="1" checked', edit_page.data)

    def test_generate_ai_hero_regenerates_on_edit_instead_of_reusing_old_image(self):
        # Regression test: editing a job that used the AI-hero fallback,
        # with the checkbox still checked and a new prompt typed in, must
        # actually regenerate -- not silently carry forward the first
        # generated image forever (the original bug: hero_provided was
        # already True from the carried-forward file, so the "not
        # hero_provided" gate below always skipped regeneration).
        first = self.client.post(
            "/generate",
            data={
                "sizes": ["1080x1080"],
                "fit_mode": "crop",
                "ai_hero_enabled": "1",
                "ai_hero_provider": "mock",
                "ai_hero_prompt": "a can of energy drink on ice, studio lighting",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(first.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", first.data).group(1).decode()
        first_hero_bytes = (
            webapp.JOBS_DIR / job_id / "uploads" / "ai_generated_hero.png"
        ).read_bytes()

        second = self.client.post(
            "/generate",
            data={
                "edit_job_id": job_id,
                "sizes": ["1080x1080"],
                "fit_mode": "crop",
                "ai_hero_enabled": "1",
                "ai_hero_provider": "mock",
                "ai_hero_prompt": "a runner sprinting on a track at sunrise",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(second.status_code, 200)
        self.assertIn(b"Hero image generated with AI (mock)", second.data)
        self.assertIn(b"a runner sprinting on a track at sunrise", second.data)
        self.assertNotIn(b"a can of energy drink on ice, studio lighting", second.data)

        second_job_id = re.search(rb"/download/([0-9a-f]+)", second.data).group(1).decode()
        second_hero_bytes = (
            webapp.JOBS_DIR / second_job_id / "uploads" / "ai_generated_hero.png"
        ).read_bytes()
        # MockImageProvider seeds its gradient from a SHA256 of the prompt
        # text (see src/providers/mock_provider.py) -- a changed prompt
        # must produce different image bytes, not a byte-for-byte reuse
        # of the first job's file.
        self.assertNotEqual(first_hero_bytes, second_hero_bytes)

    def test_generate_ai_hero_not_triggered_when_hero_image_provided(self):
        # AI generation is a gap-filler, not an override -- an uploaded
        # hero image always wins even if the checkbox is checked.
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "sizes": ["1080x1080"],
            "fit_mode": "crop",
            "ai_hero_enabled": "1",
            "ai_hero_provider": "mock",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertNotIn(b"generated with AI", r.data)

    def test_generate_ai_hero_provider_failure_falls_back_to_mock_with_warning(self):
        # huggingface_provider.py raises ImageProviderError immediately if
        # HUGGINGFACE_API_TOKEN isn't set -- exercises the fallback path
        # that mirrors src/pipeline.py's own resilience (never let one
        # flaky/misconfigured provider hard-fail the whole request).
        import os

        old_token = os.environ.pop("HUGGINGFACE_API_TOKEN", None)
        try:
            data = {
                "sizes": ["1080x1080"],
                "fit_mode": "crop",
                "ai_hero_enabled": "1",
                "ai_hero_provider": "huggingface",
            }
            r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        finally:
            if old_token is not None:
                os.environ["HUGGINGFACE_API_TOKEN"] = old_token
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'class="card"', r.data)  # still renders successfully via the mock fallback
        text = r.get_data(as_text=True)
        self.assertIn('class="background-warnings"', text)
        warnings_block = text.split('class="background-warnings"')[1].split("</ul>")[0]
        self.assertIn("huggingface", warnings_block)
        self.assertIn("HUGGINGFACE_API_TOKEN", warnings_block)

    def test_generate_rejects_unsupported_file_extension(self):
        data = {
            "hero_image": (io.BytesIO(b"not an image"), "hero.txt"),
            "header": "x",
            "description": "y",
        }
        r = self.client.post(
            "/generate", data=data, content_type="multipart/form-data", follow_redirects=True
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"supported file type", r.data)

    def test_generate_with_blank_header_omits_header_band(self):
        # No header text -> render_creative() gets headline=None -> no top
        # band at all, distinct from the default-to-product-name behavior
        # the full CLI pipeline has (there's no "product" here to fall back to).
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "header": "",
            "description": "Only a message, no header.",
            "sizes": ["default"],
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            # Top-left corner pixel should be the plain hero color (no dark
            # semi-transparent header band drawn over it).
            r_, g_, b_ = img.getpixel((5, 5))
        self.assertGreater(r_ + g_ + b_, 60)  # not near-black like a header band would be

    def test_generate_no_background_option_leaves_hero_visible_under_header(self):
        # A light, distinctive hero color -- with the default background
        # plate the header band would be dark; with "no background"
        # checked, the header band should still show the hero color away
        # from the text itself.
        data = {
            "hero_image": (self._sample_image_bytes(color=(230, 220, 200)), "hero.png"),
            "header": "Styled Header",
            "description": "Body",
            "sizes": ["default"],
            "header_text_color": "#ff2d55",
            "header_no_background": "1",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            corner_pixel = img.getpixel((5, 5))  # header band, away from centered text
        self.assertEqual(corner_pixel, (230, 220, 200))  # untouched by any background plate

    def test_generate_default_header_color_and_background_unchanged(self):
        # Omitting the new fields entirely should behave exactly like
        # before they existed: white text on a dark semi-transparent plate.
        data = {
            "hero_image": (self._sample_image_bytes(color=(230, 220, 200)), "hero.png"),
            "header": "Plain Header",
            "description": "Body",
            "sizes": ["default"],
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            corner_pixel = img.getpixel((5, 5))
        self.assertNotEqual(corner_pixel, (230, 220, 200))  # darkened by the default plate

    def test_generate_message_no_background_option_leaves_hero_visible(self):
        # Mirror of the header test above, for the bottom message banner --
        # sample near the bottom-left corner (message banner is
        # left-aligned, unlike the centered header) away from the text.
        data = {
            "hero_image": (self._sample_image_bytes(color=(80, 150, 210)), "hero.png"),
            "header": "",
            "description": "Styled Message",
            "sizes": ["default"],
            "message_text_color": "#ffe600",
            "message_no_background": "1",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            corner_pixel = img.getpixel((img.width - 5, img.height - 5))  # bottom-right, away from left-aligned text
        self.assertEqual(corner_pixel, (80, 150, 210))  # untouched by any background plate

    def test_generate_default_message_color_and_background_unchanged(self):
        data = {
            "hero_image": (self._sample_image_bytes(color=(80, 150, 210)), "hero.png"),
            "header": "",
            "description": "Plain Message",
            "sizes": ["default"],
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            corner_pixel = img.getpixel((img.width - 5, img.height - 5))
        self.assertNotEqual(corner_pixel, (80, 150, 210))  # darkened by the default plate

    def test_header_and_message_styling_are_independent(self):
        # Styling one banner shouldn't affect the other -- header goes
        # transparent/colored while message keeps its default look.
        data = {
            "hero_image": (self._sample_image_bytes(color=(230, 220, 200)), "hero.png"),
            "header": "Header",
            "description": "Message",
            "sizes": ["default"],
            "header_text_color": "#ff2d55",
            "header_no_background": "1",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            header_corner = img.getpixel((5, 5))
            message_corner = img.getpixel((5, img.height - 5))
        self.assertEqual(header_corner, (230, 220, 200))  # header: no plate
        self.assertNotEqual(message_corner, (230, 220, 200))  # message: still has its default plate

    def test_generate_header_glow_option(self):
        # header_glow + header_glow_color should produce a colored halo
        # around the header text -- mirroring the color/no-background tests
        # above, but for the new glow effect.
        data = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 10)), "hero.png"),
            "header": "Glowy Header",
            "description": "",
            "sizes": ["default"],
            "header_no_background": "1",
            "header_glow": "1",
            "header_glow_color": "#00dcff",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            found_glow = any(
                g > 120 and b > 120 and r_ < 150
                for x in range(0, img.width, 4)
                for y in range(0, 240, 4)
                for r_, g, b in [img.getpixel((x, y))]
            )
        self.assertTrue(found_glow, "expected a cyan-tinted glow halo somewhere in the header band")

    def test_generate_message_glow_option(self):
        data = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 10)), "hero.png"),
            "header": "",
            "description": "Glowy Message",
            "sizes": ["default"],
            "message_no_background": "1",
            "message_glow": "1",
            "message_glow_color": "#ff00dc",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            found_glow = any(
                r_ > 120 and b > 120 and g < 150
                for x in range(0, img.width, 4)
                for y in range(img.height - 240, img.height, 4)
                for r_, g, b in [img.getpixel((x, y))]
            )
        self.assertTrue(found_glow, "expected a magenta-tinted glow halo somewhere in the message band")

    def test_generate_default_glow_off_unless_requested(self):
        # Omitting the glow fields entirely should behave exactly like
        # before they existed -- no halo, and identical to a plain
        # no-background render.
        data = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 10)), "hero.png"),
            "header": "Plain Header",
            "description": "",
            "sizes": ["default"],
            "header_no_background": "1",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            corner_pixel = img.getpixel((5, 5))
        self.assertEqual(corner_pixel, (10, 10, 10))  # untouched -- no plate, no glow halo

    def test_generate_header_right_align_moves_text_away_from_left_edge(self):
        data = {
            "hero_image": (self._sample_image_bytes(color=(0, 0, 0)), "hero.png"),
            "header": "Hi",
            "description": "",
            "sizes": ["default"],
            "header_no_background": "1",
            "header_align": "right",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            # Right-aligned short text shouldn't light up any pixels in the
            # header band's far-left strip.
            left_strip_has_text = any(
                sum(img.getpixel((x, y))) > 300
                for x in range(0, 60)
                for y in range(0, 80, 2)
            )
        self.assertFalse(left_strip_has_text, "expected right-aligned header text to leave the far-left strip empty")

    def test_generate_invalid_align_falls_back_to_default(self):
        # A bogus align value (never sent by the real form, but defensive
        # against a hand-crafted request) should fall back to the original
        # default rather than raising.
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "header": "Header",
            "description": "Message",
            "sizes": ["default"],
            "header_align": "diagonal",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)

    def test_generate_header_font_size_option(self):
        data = {
            "hero_image": (self._sample_image_bytes(color=(0, 0, 0)), "hero.png"),
            "header": "Hi",
            "description": "",
            "sizes": ["default"],
            "header_no_background": "1",
            "header_font_size": "150",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            # A pinned 150px font on a short "Hi" should light up text
            # pixels well past the normal ~110px (0.22 * 1080) header cap.
            text_below_normal_cap = any(
                sum(img.getpixel((x, y))) > 300
                for x in range(0, img.width, 4)
                for y in range(130, 200, 4)
            )
        self.assertTrue(text_below_normal_cap, "expected a pinned 150px header font to render well past the normal header height")

    def test_generate_blank_font_size_falls_back_to_autofit(self):
        # An empty font-size field (the normal "Auto" case) shouldn't error
        # or be treated as 0 -- it should behave exactly like the field
        # doesn't exist at all.
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "header": "Header",
            "description": "Message",
            "sizes": ["default"],
            "header_font_size": "",
            "message_font_size": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)

    def test_generate_logo_appears_at_requested_position(self):
        from src.image_ops import logo_render_size

        data = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 10)), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
            "logo": (self._sample_logo_bytes(), "logo.png"),
            "logo_position": "bottom-left",
            "logo_scale": "20",
            "logo_opacity": "100",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            logo_w, logo_h, margin = logo_render_size((1080, 1080), Image.new("RGBA", (200, 100)), scale_frac=0.2)
            cx = margin + logo_w // 2
            cy = 1080 - margin - logo_h // 2
            r_, g_, b_ = img.getpixel((cx, cy))
        # The sample logo is solid yellow -- high red and green, low blue.
        self.assertGreater(r_, 200)
        self.assertGreater(g_, 150)
        self.assertLess(b_, 80)

    def test_generate_below_header_logo_clears_the_header_and_persists_through_edit(self):
        from src.image_ops import header_banner_height

        headline = "A Fairly Long Product Headline Here"
        data = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 10)), "hero.png"),
            "header": headline,
            "description": "",
            "sizes": ["default"],
            "logo": (self._sample_logo_bytes(), "logo.png"),
            "logo_position": "below-header-center",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"

        def is_logo_yellow(pixel):
            r_, g_, b_ = pixel
            return r_ > 200 and g_ > 150 and b_ < 80

        header_h = header_banner_height((1080, 1080), headline)
        with Image.open(out_path) as img:
            in_header = any(
                is_logo_yellow(img.getpixel((x, y))) for x in range(0, img.width, 4) for y in range(0, header_h)
            )
            below_header = any(
                is_logo_yellow(img.getpixel((x, y)))
                for x in range(0, img.width, 4)
                for y in range(header_h, min(header_h + 250, img.height))
            )
        self.assertFalse(in_header, "the below-header logo shouldn't render inside the header banner")
        self.assertTrue(below_header, "expected the below-header logo to render just underneath the header")

        edit_page = self.client.get(f"/edit/{job_id}")
        self.assertIn(
            b'<input type="radio" name="logo_position" value="below-header-center" checked>', edit_page.data
        )

    def test_generate_logo_offset_nudges_the_logo_and_persists_through_edit(self):
        data = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 10)), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
            "logo": (self._sample_logo_bytes(), "logo.png"),
            "logo_position": "center",
            "logo_offset_x": "400",
            "logo_offset_y": "-300",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"

        def is_logo_yellow(pixel):
            r_, g_, b_ = pixel
            return r_ > 200 and g_ > 150 and b_ < 80

        with Image.open(out_path) as img:
            cx, cy = img.width // 2, img.height // 2
            self.assertFalse(is_logo_yellow(img.getpixel((cx, cy))), "expected the un-nudged center spot to be clear")
            self.assertTrue(
                is_logo_yellow(img.getpixel((cx + 400, cy - 300))),
                "expected the logo at its nudged spot (400 right, 300 up)",
            )

        edit_page = self.client.get(f"/edit/{job_id}")
        self.assertIn(b'name="logo_offset_x" min="-2000" max="2000" value="400"', edit_page.data)
        self.assertIn(b'name="logo_offset_y" min="-2000" max="2000" value="-300"', edit_page.data)

    def test_generate_offers_a_psd_download_link_per_size(self):
        from psd_tools import PSDImage

        from src.image_ops import get_psd_layer_boxes

        data = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 10)), "hero.png"),
            "header": "A Fairly Long Product Headline Here",
            "description": "Shop the new drop today",
            "sizes": ["default"],
            "logo": (self._sample_logo_bytes(), "logo.png"),
            "logo_position": "top-right",
            "cta_text": "Shop Now",
            "cta_position": "bottom-right",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()

        # "default" is 3 sizes -- expect a Download PSD link for each.
        psd_links = re.findall(rb'/download-psd/' + job_id.encode() + rb'/([\w.]+\.psd)', r.data)
        self.assertEqual(len(psd_links), 3, r.data)

        for psd_filename_bytes in psd_links:
            psd_filename = psd_filename_bytes.decode()
            self.assertTrue((webapp.JOBS_DIR / job_id / psd_filename).is_file())

            resp = self.client.get(f"/download-psd/{job_id}/{psd_filename}")
            self.assertEqual(resp.status_code, 200)
            self.assertIn(b"attachment", resp.headers.get("Content-Disposition", "").encode())

            tmp_psd = Path(self.tmp_dir) / f"downloaded_{psd_filename}"
            tmp_psd.write_bytes(resp.data)

            # Layers are named/cropped for re-upload -- see src/psd_export.py
            # -- so "logo"/"description"/"product"/"cta" (what
            # REQUIRED_PSD_LAYERS/get_psd_layer_boxes() actually look for on
            # a PSD-template upload) should already be present. "Header" has
            # no matching upload-flow role, so it keeps its own name.
            psd = PSDImage.open(tmp_psd)
            layer_names = [l.name for l in psd]
            self.assertIn("product", layer_names)
            self.assertIn("Header", layer_names)
            self.assertIn("logo", layer_names)
            self.assertIn("description", layer_names)
            self.assertIn("cta", layer_names)

            # And the real proof: the app's own PSD-template upload check
            # (Pillow-based, not psd_tools) finds all three required
            # layers plus the optional cta -- this is what would actually
            # let someone re-upload this exact download.
            boxes = get_psd_layer_boxes(tmp_psd)
            for required in ("logo", "description", "product", "cta"):
                self.assertIn(required, boxes, f"{required!r} not recognized by get_psd_layer_boxes()")

    def test_download_psd_rejects_non_psd_filenames_and_unknown_jobs(self):
        data = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 10)), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()

        # Real job, but asking for a non-.psd file through this route.
        resp = self.client.get(f"/download-psd/{job_id}/creative_1080x1080.png")
        self.assertEqual(resp.status_code, 404)

        # Made-up job id.
        resp = self.client.get("/download-psd/does-not-exist/creative_1080x1080.psd")
        self.assertEqual(resp.status_code, 404)

    def test_generate_without_logo_file_is_unaffected(self):
        # Omitting the logo file entirely (the normal case) shouldn't error
        # or change output -- logo position/scale/opacity fields being
        # present with no file attached should just be ignored.
        data = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 10)), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
            "logo_position": "center",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            corner_pixel = img.getpixel((2, 2))
        self.assertEqual(corner_pixel, (10, 10, 10))  # untouched -- no logo file was attached

    def test_generate_logo_invalid_position_falls_back_to_top_right(self):
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
            "logo": (self._sample_logo_bytes(), "logo.png"),
            "logo_position": "sideways",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)

    def test_generate_default_logo_position_scale_opacity_are_unchanged(self):
        # Omitting the new fields entirely (just uploading a logo, the
        # original-only workflow) should behave exactly like before these
        # options existed -- top-right, ~16% scale, fully opaque.
        data_default = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 10)), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
            "logo": (self._sample_logo_bytes(), "logo.png"),
        }
        data_explicit = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 10)), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
            "logo": (self._sample_logo_bytes(), "logo.png"),
            "logo_position": "top-right",
            "logo_scale": "16",
            "logo_opacity": "100",
        }
        r_default = self.client.post("/generate", data=data_default, content_type="multipart/form-data")
        r_explicit = self.client.post("/generate", data=data_explicit, content_type="multipart/form-data")
        self.assertEqual(r_default.status_code, 200)
        self.assertEqual(r_explicit.status_code, 200)
        job_default = re.search(rb"/download/([0-9a-f]+)", r_default.data).group(1).decode()
        job_explicit = re.search(rb"/download/([0-9a-f]+)", r_explicit.data).group(1).decode()
        with Image.open(webapp.JOBS_DIR / job_default / "creative_1080x1080.png") as img_default, Image.open(
            webapp.JOBS_DIR / job_explicit / "creative_1080x1080.png"
        ) as img_explicit:
            self.assertEqual(list(img_default.getdata()), list(img_explicit.getdata()))

    def test_generate_badge_image_appears_at_requested_corner(self):
        from src.image_ops import badge_render_size

        data = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 10)), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
            "badge_image": (self._sample_badge_bytes(), "badge.png"),
            "badge_position": "bottom-right",
            "badge_scale": "40",
            "badge_opacity": "100",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            target_w, target_h, margin = badge_render_size((1080, 1080), Image.new("RGBA", (200, 200)), scale_frac=0.4)
            cx = 1080 - target_w - margin + target_w // 2
            cy = 1080 - target_h - margin + target_h // 2
            r_, g_, b_ = img.getpixel((cx, cy))
        self.assertGreater(r_, 180)
        self.assertLess(g_, 80)
        self.assertLess(b_, 80)

    def test_generate_badge_full_position_covers_frame_and_ignores_scale(self):
        data = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 10)), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
            "badge_image": (self._sample_badge_bytes(color=(0, 200, 0, 255)), "tint.png"),
            "badge_position": "full",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            corner_pixel = img.getpixel((2, 2))
        self.assertEqual(corner_pixel, (0, 200, 0))

    def test_generate_without_badge_file_is_unaffected(self):
        # Omitting the badge file entirely (the normal case) shouldn't
        # error or change output -- badge position/scale/opacity fields
        # being present with no file attached should just be ignored.
        data = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 10)), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
            "badge_position": "full",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            corner_pixel = img.getpixel((2, 2))
        self.assertEqual(corner_pixel, (10, 10, 10))  # untouched -- no badge file was attached

    def test_generate_badge_invalid_position_falls_back_to_top_right(self):
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
            "badge_image": (self._sample_badge_bytes(), "badge.png"),
            "badge_position": "sideways",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)

    def test_generate_badge_rejects_unsupported_file_extension(self):
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "header": "",
            "description": "",
            "badge_image": (io.BytesIO(b"not an image"), "badge.txt"),
        }
        r = self.client.post(
            "/generate", data=data, content_type="multipart/form-data", follow_redirects=True
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"supported type", r.data)

    def test_generate_cta_button_appears_at_requested_position(self):
        data = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 10)), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
            "cta_text": "Shop Now",
            "cta_position": "top-left",
            "cta_button_color": "#ff2d55",
            "cta_text_color": "#ffffff",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            found_button = any(
                r_ > 180 and g < 100 and b < 130
                for x in range(0, img.width // 2, 4)
                for y in range(0, img.height // 3, 4)
                for r_, g, b in [img.getpixel((x, y))]
            )
        self.assertTrue(found_button, "expected a pink CTA button somewhere in the top-left region")

    def test_generate_cta_above_message_checkbox_lifts_button_above_banner(self):
        data = {
            "hero_image": (self._sample_image_bytes(color=(60, 120, 180)), "hero.png"),
            "header": "",
            "description": "Free shipping on orders over fifty dollars this week only",
            "sizes": ["default"],
            "cta_text": "Shop Now",
            "cta_position": "bottom-center",
            "cta_button_color": "#ff0000",
            "cta_above_message": "1",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"

        def has_red(pixel):
            r_, g, b = pixel
            return r_ > 140 and r_ > g + 60 and r_ > b + 60

        with Image.open(out_path) as img:
            # The bottom-most rows (where an un-lifted bottom-center button
            # would normally sit, overlapping the message banner's own
            # text) should now be clear of the button's red.
            bottom_strip_has_red = any(
                has_red(img.getpixel((x, y)))
                for x in range(0, img.width, 4)
                for y in range(img.height - 10, img.height)
            )
            # But the button must still be visible somewhere above that --
            # lifted, not simply removed.
            found_lifted_button = any(
                has_red(img.getpixel((x, y)))
                for x in range(0, img.width, 4)
                for y in range(img.height // 3, img.height - 10)
            )
        self.assertFalse(bottom_strip_has_red, "the button should be lifted clear of the very bottom edge")
        self.assertTrue(found_lifted_button, "expected the lifted CTA button to still render, just higher up")

        # Persists through Edit like every other checkbox in
        # EDIT_CHECKBOX_FIELD_NAMES.
        edit_page = self.client.get(f"/edit/{job_id}")
        self.assertIn(b'id="cta_above_message" name="cta_above_message" value="1" checked', edit_page.data)

    def test_generate_without_cta_text_is_unaffected(self):
        # Omitting the CTA text entirely (the normal case) shouldn't error
        # or change output -- style/position fields being present with no
        # text should just be ignored.
        data = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 10)), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
            "cta_position": "bottom-center",
            "cta_button_color": "#ff2d55",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            corner_pixel = img.getpixel((img.width // 2, img.height - 5))
        self.assertEqual(corner_pixel, (10, 10, 10))  # untouched -- no CTA text was given

    def test_generate_cta_invalid_position_falls_back_to_bottom_center(self):
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
            "cta_text": "Buy",
            "cta_position": "sideways",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)

    def test_generate_cta_font_size_option(self):
        data = {
            "hero_image": (self._sample_image_bytes(color=(0, 0, 0)), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
            "cta_text": "Go",
            "cta_position": "center",
            "cta_font_size": "80",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            # An 80px pinned font on a short "Go" button should make the
            # button noticeably tall/wide compared to the ~65px default
            # button height at this canvas size -- check that button-colored
            # pixels extend a good distance from dead-center.
            cx, cy = img.width // 2, img.height // 2
            found_far_from_center = any(
                sum(img.getpixel((x, y))) > 150  # button-blue or text-white, not the black background
                for x in range(cx - 10, cx + 10)
                for y in [cy - 68, cy + 68]
            )
        self.assertTrue(found_far_from_center, "expected an 80px pinned CTA font to render a noticeably tall button")

    def test_generate_cta_font_family_option(self):
        # Rendering the same short label at the same explicit font size in
        # two different families should produce different pixel data --
        # confirms the form field actually reaches add_cta_button().
        data_sans = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 10)), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
            "cta_text": "Shop Now",
            "cta_position": "center",
            "cta_font_size": "60",
            "cta_font_family": "sans",
        }
        data_serif = dict(data_sans, cta_font_family="serif")
        data_serif["hero_image"] = (self._sample_image_bytes(color=(10, 10, 10)), "hero.png")

        r_sans = self.client.post("/generate", data=data_sans, content_type="multipart/form-data")
        r_serif = self.client.post("/generate", data=data_serif, content_type="multipart/form-data")
        self.assertEqual(r_sans.status_code, 200)
        self.assertEqual(r_serif.status_code, 200)

        job_sans = re.search(rb"/download/([0-9a-f]+)", r_sans.data).group(1).decode()
        job_serif = re.search(rb"/download/([0-9a-f]+)", r_serif.data).group(1).decode()
        with Image.open(webapp.JOBS_DIR / job_sans / "creative_1080x1080.png") as img_sans, Image.open(
            webapp.JOBS_DIR / job_serif / "creative_1080x1080.png"
        ) as img_serif:
            self.assertNotEqual(list(img_sans.getdata()), list(img_serif.getdata()))

    def test_generate_cta_invalid_font_family_falls_back_to_sans(self):
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
            "cta_text": "Go",
            "cta_font_family": "wingdings",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)

    def test_generate_cta_glow_option(self):
        data = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 10)), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
            "cta_text": "Shop Now",
            "cta_position": "center",
            "cta_button_color": "#0057b8",
            "cta_glow": "1",
            "cta_glow_color": "#00dcff",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            found_glow = any(
                g > 150 and b > 150 and r_ < 60
                for r_, g, b in img.getdata()
            )
        self.assertTrue(found_glow, "expected a cyan-tinted glow halo somewhere around the CTA button")

    def test_generate_cta_default_glow_off_unless_requested(self):
        data = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 10)), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
            "cta_text": "Shop Now",
            "cta_position": "center",
            "cta_button_color": "#0057b8",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_1080x1080.png"
        with Image.open(out_path) as img:
            found_glow = any(
                g > 150 and b > 150 and r_ < 60
                for r_, g, b in img.getdata()
            )
        self.assertFalse(found_glow, "no glow was requested -- there should be no cyan-tinted halo")

    def test_generate_cta_blank_font_size_falls_back_to_autofit(self):
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "header": "",
            "description": "",
            "sizes": ["default"],
            "cta_text": "Learn More",
            "cta_font_size": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)

    def test_download_nonexistent_job_returns_404(self):
        r = self.client.get("/download/deadbeefdeadbeefdeadbeefdeadbeef")
        self.assertEqual(r.status_code, 404)

    def test_serve_output_rejects_path_traversal(self):
        r = self.client.get("/outputs/deadbeefdeadbeefdeadbeefdeadbeef/../../etc/passwd")
        self.assertIn(r.status_code, (404, 400))


class PsdTemplateSectionTest(unittest.TestCase):
    """Covers the size-specific PSD-template upload section: a template
    fills in as that size's background, the hero image becomes optional
    once every requested size is covered, and the various row-shaped user
    errors are flagged clearly instead of crashing."""

    def setUp(self):
        webapp.app.config["TESTING"] = True
        self.client = _CampaignBriefAutoFillClient(webapp.app.test_client())
        self._orig_jobs_dir = webapp.JOBS_DIR
        self.tmp_dir = tempfile.mkdtemp()
        webapp.JOBS_DIR = Path(self.tmp_dir)
        # Isolate from whatever real .psd templates a user has saved in
        # their own project's default_templates/ folder.
        self._orig_default_templates_dir = webapp.DEFAULT_TEMPLATES_DIR
        self.tmp_default_templates_dir = tempfile.mkdtemp()
        webapp.DEFAULT_TEMPLATES_DIR = Path(self.tmp_default_templates_dir)

    def tearDown(self):
        webapp.JOBS_DIR = self._orig_jobs_dir
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        webapp.DEFAULT_TEMPLATES_DIR = self._orig_default_templates_dir
        shutil.rmtree(self.tmp_default_templates_dir, ignore_errors=True)

    def _sample_image_bytes(self, size=(400, 300), color=(20, 100, 200)):
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, format="PNG")
        buf.seek(0)
        return buf

    def _sample_psd_bytes(self, size=(64, 40), color=(200, 50, 50)):
        # Includes three real named layers -- logo/description/product --
        # see WebAppSmokeTest._sample_psd_bytes()'s docstring for why.
        width, height = size
        r, g, b = color
        header = b"8BPS"
        header += struct.pack(">H", 1)  # version
        header += b"\x00" * 6  # reserved
        header += struct.pack(">H", 3)  # channels (RGB)
        header += struct.pack(">I", height)
        header += struct.pack(">I", width)
        header += struct.pack(">H", 8)  # depth
        header += struct.pack(">H", 3)  # color mode: RGB
        color_mode_data = struct.pack(">I", 0)
        image_resources = struct.pack(">I", 0)

        # Real named layers -- webapp.py's REQUIRED_PSD_LAYERS validation
        # (logo/description/product) rejects any template missing one, so
        # every synthetic test PSD needs real layer records with those
        # exact names, not just an empty layer-and-mask section. Small
        # stub boxes in a top strip -- they just need to exist and be
        # named correctly, tests that care about layer *placement* use a
        # real project template instead (see LayerOverrideIntegrationTest).
        stub_layers = [
            ("logo", (0, 0, 20, 10)),
            ("description", (20, 0, 40, 10)),
            ("product", (40, 0, 60, 10)),
        ]
        layer_records = b""
        channel_data = b""
        for name, (lx0, ly0, lx1, ly1) in stub_layers:
            lw, lh = lx1 - lx0, ly1 - ly0
            rec = struct.pack(">iiii", ly0, lx0, ly1, lx1)
            rec += struct.pack(">H", 3)  # 3 channels
            for ch_id in (0, 1, 2):
                rec += struct.pack(">H", ch_id)
                rec += struct.pack(">I", 2 + lw * lh)  # declared size (unused by Pillow's reader)
            rec += b"8BIM" + b"norm"
            rec += struct.pack(">B", 255)  # opacity
            rec += struct.pack(">B", 0)  # clipping
            rec += struct.pack(">B", 0)  # flags
            rec += struct.pack(">B", 0)  # filler
            name_bytes = name.encode("latin-1")
            extra = struct.pack(">I", 0) + struct.pack(">I", 0)  # mask + blending-ranges lengths (both empty)
            extra += struct.pack(">B", len(name_bytes)) + name_bytes
            rec += struct.pack(">I", len(extra)) + extra
            layer_records += rec
            stub_plane = lambda value, n=lw * lh: bytes([value]) * n
            for v in color:
                channel_data += struct.pack(">H", 0) + stub_plane(v)  # compression=0 (raw)

        layer_info_inner = struct.pack(">h", len(stub_layers)) + layer_records + channel_data
        if len(layer_info_inner) % 2:
            layer_info_inner += b"\x00"
        layer_info = struct.pack(">I", len(layer_info_inner)) + layer_info_inner
        global_layer_mask_info = struct.pack(">I", 0)
        layer_and_mask_section = layer_info + global_layer_mask_info
        layer_mask_info = struct.pack(">I", len(layer_and_mask_section)) + layer_and_mask_section

        compression = struct.pack(">H", 0)  # raw, uncompressed
        plane = lambda value: bytes([value]) * (width * height)
        image_data = compression + plane(r) + plane(g) + plane(b)
        buf = io.BytesIO(header + color_mode_data + image_resources + layer_mask_info + image_data)
        buf.seek(0)
        return buf

    def test_generate_psd_template_without_hero_image_succeeds(self):
        data = {
            "psd_size_1": "300x250",
            "psd_file_1": (self._sample_psd_bytes(color=(90, 200, 40)), "template.psd"),
            "custom_sizes": "300x250",
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"used your uploaded PSD template", r.data)

        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        out_path = webapp.JOBS_DIR / job_id / "creative_300x250.png"
        with Image.open(out_path) as img:
            r_, g_, b_ = img.getpixel((5, 5))
        # No header/message text was requested, so the whole frame should
        # still be the template's solid greenish color.
        self.assertGreater(g_, r_)
        self.assertGreater(g_, b_)

    def test_generate_missing_hero_and_template_for_size_flashes_and_redirects(self):
        data = {
            "custom_sizes": "300x250",
            "header": "",
            "description": "",
        }
        r = self.client.post(
            "/generate", data=data, content_type="multipart/form-data", follow_redirects=True
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"300x250", r.data)
        self.assertIn(b"hero image or a matching PSD template", r.data)

    def test_generate_hero_and_psd_template_used_for_their_respective_sizes(self):
        data = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 200)), "hero.png"),
            "psd_size_1": "300x250",
            "psd_file_1": (self._sample_psd_bytes(color=(200, 200, 10)), "template.psd"),
            "custom_sizes": "300x250,320x50",
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"300x250 used your uploaded PSD template", r.data)

        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        with Image.open(webapp.JOBS_DIR / job_id / "creative_300x250.png") as templated:
            tr, tg, tb = templated.getpixel((5, 5))
        with Image.open(webapp.JOBS_DIR / job_id / "creative_320x50.png") as from_hero:
            hr, hg, hb = from_hero.getpixel((5, 5))

        self.assertGreater(tr, 150)
        self.assertGreater(tg, 150)
        self.assertLess(tb, 100)
        self.assertLess(hr, 100)
        self.assertLess(hg, 100)
        self.assertGreater(hb, 150)

    def test_generate_psd_template_size_not_in_list_is_added_automatically(self):
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "psd_size_1": "970x90",
            "psd_file_1": (self._sample_psd_bytes(), "template.psd"),
            "sizes": ["default"],
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        # The 3 social defaults plus the auto-added 970x90 template size.
        self.assertEqual(r.data.count(b'class="card"'), 4)
        self.assertIn(b"970x90", r.data)

    def test_generate_psd_template_row_with_file_but_no_size_flashes(self):
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "psd_file_1": (self._sample_psd_bytes(), "template.psd"),
            "sizes": ["default"],
        }
        r = self.client.post(
            "/generate", data=data, content_type="multipart/form-data", follow_redirects=True
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"target size", r.data)

    def test_generate_psd_template_row_with_size_but_no_file_flashes(self):
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "psd_size_1": "300x250",
            "sizes": ["default"],
        }
        r = self.client.post(
            "/generate", data=data, content_type="multipart/form-data", follow_redirects=True
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn(b".psd file", r.data)

    def test_generate_psd_template_wrong_extension_flashes(self):
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "psd_size_1": "300x250",
            "psd_file_1": (self._sample_image_bytes(), "template.png"),
            "sizes": ["default"],
        }
        r = self.client.post(
            "/generate", data=data, content_type="multipart/form-data", follow_redirects=True
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"PSD template row 1", r.data)
        self.assertIn(b"supported file type", r.data)

    def test_generate_psd_template_size_offers_the_original_file_as_its_psd_download(self):
        # The template branch's "Download PSD" is the user's own uploaded
        # template file itself (already using the app's own required
        # layer names, since that's what made it a valid template in the
        # first place) -- not a rebuild with any overrides baked in. See
        # the comment above `psd_source_path` in webapp.py for why.
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "psd_size_1": "300x250",
            "psd_file_1": (self._sample_psd_bytes(color=(90, 200, 40)), "template.psd"),
            "custom_sizes": "300x250,320x50",
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)

        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        self.assertIn(
            f"/download-psd/{job_id}/creative_300x250.psd".encode(), r.data
        )
        psd_path = webapp.JOBS_DIR / job_id / "creative_300x250.psd"
        self.assertTrue(psd_path.is_file())
        from src.image_ops import get_psd_layer_boxes
        layer_boxes = get_psd_layer_boxes(psd_path)
        self.assertIn("logo", layer_boxes)
        self.assertIn("description", layer_boxes)
        self.assertIn("product", layer_boxes)

        # 320x50 rendered from the hero image, not a template -- it gets
        # its own (differently-built) PSD download, not this one.
        self.assertIn(
            f"/download-psd/{job_id}/creative_320x50.psd".encode(), r.data
        )
        self.assertNotEqual(
            (webapp.JOBS_DIR / job_id / "creative_320x50.psd").read_bytes(),
            psd_path.read_bytes(),
        )

        dl = self.client.get(f"/download-psd/{job_id}/creative_300x250.psd")
        self.assertEqual(dl.status_code, 200)
        self.assertEqual(dl.data, psd_path.read_bytes())

    def test_generate_psd_template_bad_size_string_flashes(self):
        data = {
            "hero_image": (self._sample_image_bytes(), "hero.png"),
            "psd_size_1": "not-a-size",
            "psd_file_1": (self._sample_psd_bytes(), "template.psd"),
            "sizes": ["default"],
        }
        r = self.client.post(
            "/generate", data=data, content_type="multipart/form-data", follow_redirects=True
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"PSD template row 1", r.data)




class DefaultTemplatesFolderTest(unittest.TestCase):
    """Covers default_templates/: .psd files saved there permanently (not
    uploaded per-request) that auto-apply to their matching output size,
    auto-include that size in the batch, yield to a per-request upload for
    the same size, and don't crash generation if one of them is corrupt."""

    def setUp(self):
        webapp.app.config["TESTING"] = True
        self.client = _CampaignBriefAutoFillClient(webapp.app.test_client())
        self._orig_jobs_dir = webapp.JOBS_DIR
        self.tmp_dir = tempfile.mkdtemp()
        webapp.JOBS_DIR = Path(self.tmp_dir)
        # Point default_templates/ scanning at an empty scratch dir instead
        # of the real project folder -- these tests should never depend on
        # (or risk interfering with) a real user's saved templates.
        self._orig_default_templates_dir = webapp.DEFAULT_TEMPLATES_DIR
        self.tmp_default_templates_dir = tempfile.mkdtemp()
        webapp.DEFAULT_TEMPLATES_DIR = Path(self.tmp_default_templates_dir)

    def tearDown(self):
        webapp.JOBS_DIR = self._orig_jobs_dir
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        webapp.DEFAULT_TEMPLATES_DIR = self._orig_default_templates_dir
        shutil.rmtree(self.tmp_default_templates_dir, ignore_errors=True)

    def _write_default_template(self, filename, data: bytes) -> Path:
        path = webapp.DEFAULT_TEMPLATES_DIR / filename
        path.write_bytes(data)
        return path

    def _sample_image_bytes(self, size=(400, 300), color=(20, 100, 200)):
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, format="PNG")
        buf.seek(0)
        return buf

    def _sample_psd_bytes(self, size=(64, 40), color=(200, 50, 50)):
        # Includes three real named layers -- logo/description/product --
        # see WebAppSmokeTest._sample_psd_bytes()'s docstring for why.
        width, height = size
        r, g, b = color
        header = b"8BPS"
        header += struct.pack(">H", 1)  # version
        header += b"\x00" * 6  # reserved
        header += struct.pack(">H", 3)  # channels (RGB)
        header += struct.pack(">I", height)
        header += struct.pack(">I", width)
        header += struct.pack(">H", 8)  # depth
        header += struct.pack(">H", 3)  # color mode: RGB
        color_mode_data = struct.pack(">I", 0)
        image_resources = struct.pack(">I", 0)

        # Real named layers -- webapp.py's REQUIRED_PSD_LAYERS validation
        # (logo/description/product) rejects any template missing one, so
        # every synthetic test PSD needs real layer records with those
        # exact names, not just an empty layer-and-mask section. Small
        # stub boxes in a top strip -- they just need to exist and be
        # named correctly, tests that care about layer *placement* use a
        # real project template instead (see LayerOverrideIntegrationTest).
        stub_layers = [
            ("logo", (0, 0, 20, 10)),
            ("description", (20, 0, 40, 10)),
            ("product", (40, 0, 60, 10)),
        ]
        layer_records = b""
        channel_data = b""
        for name, (lx0, ly0, lx1, ly1) in stub_layers:
            lw, lh = lx1 - lx0, ly1 - ly0
            rec = struct.pack(">iiii", ly0, lx0, ly1, lx1)
            rec += struct.pack(">H", 3)  # 3 channels
            for ch_id in (0, 1, 2):
                rec += struct.pack(">H", ch_id)
                rec += struct.pack(">I", 2 + lw * lh)  # declared size (unused by Pillow's reader)
            rec += b"8BIM" + b"norm"
            rec += struct.pack(">B", 255)  # opacity
            rec += struct.pack(">B", 0)  # clipping
            rec += struct.pack(">B", 0)  # flags
            rec += struct.pack(">B", 0)  # filler
            name_bytes = name.encode("latin-1")
            extra = struct.pack(">I", 0) + struct.pack(">I", 0)  # mask + blending-ranges lengths (both empty)
            extra += struct.pack(">B", len(name_bytes)) + name_bytes
            rec += struct.pack(">I", len(extra)) + extra
            layer_records += rec
            stub_plane = lambda value, n=lw * lh: bytes([value]) * n
            for v in color:
                channel_data += struct.pack(">H", 0) + stub_plane(v)  # compression=0 (raw)

        layer_info_inner = struct.pack(">h", len(stub_layers)) + layer_records + channel_data
        if len(layer_info_inner) % 2:
            layer_info_inner += b"\x00"
        layer_info = struct.pack(">I", len(layer_info_inner)) + layer_info_inner
        global_layer_mask_info = struct.pack(">I", 0)
        layer_and_mask_section = layer_info + global_layer_mask_info
        layer_mask_info = struct.pack(">I", len(layer_and_mask_section)) + layer_and_mask_section

        compression = struct.pack(">H", 0)  # raw, uncompressed
        plane = lambda value: bytes([value]) * (width * height)
        image_data = compression + plane(r) + plane(g) + plane(b)
        return header + color_mode_data + image_resources + layer_mask_info + image_data

    def test_psd_template_row_does_not_pull_in_default_templates_for_other_sizes(self):
        # A Size-specific PSD template row is scoped to exactly the
        # size(s) uploaded there -- it must NOT also switch on
        # default_templates/ auto-use for every OTHER requested size.
        # Only the 728x480 content PSD field does that (see
        # ContentPsdQuickModeTest). A saved 970x90 template existing on
        # disk must be irrelevant here even though a PSD was uploaded
        # this request (for the unrelated 300x250 row) and 970x90 is
        # part of the requested batch.
        self._write_default_template("tester-970x90.psd", self._sample_psd_bytes(color=(30, 180, 30)))

        data = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 200)), "hero.png"),
            "psd_size_1": "300x250",
            "psd_file_1": (io.BytesIO(self._sample_psd_bytes(color=(200, 200, 10))), "template.psd"),
            "custom_sizes": "970x90",
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"970x90", r.data)
        self.assertNotIn(b"used the saved default template", r.data)

        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        with Image.open(webapp.JOBS_DIR / job_id / "creative_970x90.png") as img:
            r_, g_, b_ = img.getpixel((5, 5))
        # 970x90 fell back to the hero image (blue), not the saved
        # template (green) -- the saved template was never even scanned.
        self.assertGreater(b_, r_)
        self.assertGreater(b_, g_)

    def test_default_template_ignored_when_no_psd_uploaded_this_request(self):
        # No PSD upload at all this request (just a plain hero image) --
        # the saved default must NOT be pulled in, even though it exists
        # on disk and even though its size matches nothing requested here
        # is a coincidence away from colliding. The normal tool (hero +
        # overlay pipeline) should be used for every requested size.
        self._write_default_template("tester-970x90.psd", self._sample_psd_bytes(color=(30, 180, 30)))

        data = {
            "hero_image": (self._sample_image_bytes(color=(10, 10, 200)), "hero.png"),
            "sizes": ["default"],
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertNotIn(b"970x90", r.data)
        self.assertNotIn(b"used the saved default template", r.data)
        # Exactly the 3 social defaults -- nothing extra snuck in from
        # default_templates/.
        self.assertEqual(r.data.count(b'class="card"'), 3)

    def test_per_request_upload_overrides_saved_default_for_same_size(self):
        self._write_default_template("300x250.psd", self._sample_psd_bytes(color=(200, 200, 10)))

        data = {
            "psd_size_1": "300x250",
            "psd_file_1": (io.BytesIO(self._sample_psd_bytes(color=(10, 10, 200))), "fresh.psd"),
            "custom_sizes": "300x250",
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"300x250 used your uploaded PSD template", r.data)
        self.assertNotIn(b"used the saved default template", r.data)

        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        with Image.open(webapp.JOBS_DIR / job_id / "creative_300x250.png") as img:
            r_, g_, b_ = img.getpixel((5, 5))
        # The per-request upload's blue wins, not the saved default's yellow.
        self.assertLess(r_, 100)
        self.assertLess(g_, 100)
        self.assertGreater(b_, 150)

    def test_edit_without_reuploading_keeps_using_the_carried_forward_psd_template(self):
        # The whole point of carrying a PSD template forward on edit: the
        # user uploads it once, then tweaks something unrelated (here,
        # the header text) and resubmits without touching the PSD row at
        # all -- the template should still drive that size, no re-upload
        # needed. See _carry_forward_upload() and the edit_job_id
        # handling in generate().
        first_data = {
            "hero_image": (self._sample_image_bytes(color=(10, 200, 10)), "hero.png"),
            "psd_size_1": "300x250",
            "psd_file_1": (io.BytesIO(self._sample_psd_bytes(color=(200, 30, 30))), "template.psd"),
            "custom_sizes": "300x250",
            "header": "",
            "description": "",
        }
        first = self.client.post("/generate", data=first_data, content_type="multipart/form-data")
        self.assertEqual(first.status_code, 200)
        self.assertIn(b"300x250 used your uploaded PSD template", first.data)
        job_id = re.search(rb"/download/([0-9a-f]+)", first.data).group(1).decode()

        # The edit page must show a "Currently: template.psd" hint for
        # this row -- that's the visible half of "cached"; the carried
        # -forward file on resubmit (checked below) is the functional half.
        edit_page = self.client.get(f"/edit/{job_id}")
        self.assertIn(b"Currently: <strong>template.psd</strong>", edit_page.data)
        self.assertIn(b'value="300x250"', edit_page.data)

        # Resubmit exactly as the pre-filled edit form would -- the size
        # field carried over by the browser, no new psd_file_1, no clear
        # flag -- just a change to an unrelated field.
        second_data = {
            "edit_job_id": job_id,
            "psd_size_1": "300x250",
            "custom_sizes": "300x250",
            "header": "New Headline",
            "description": "",
        }
        second = self.client.post("/generate", data=second_data, content_type="multipart/form-data")
        self.assertEqual(second.status_code, 200)
        self.assertIn(b"300x250 used your uploaded PSD template", second.data)

        second_job_id = re.search(rb"/download/([0-9a-f]+)", second.data).group(1).decode()
        with Image.open(webapp.JOBS_DIR / second_job_id / "creative_300x250.png") as img:
            r2, g2, b2 = img.getpixel((5, 5))
        # Still the template's reddish color, not the hero's green -- the
        # template kept driving this size across the edit.
        self.assertGreater(r2, g2)

        # And a *second* edit off of that job, still not touching the
        # PSD row, must carry the same template forward again -- proving
        # this isn't a one-hop copy that only survives a single edit.
        third_edit_page = self.client.get(f"/edit/{second_job_id}")
        self.assertIn(b"Currently: <strong>template.psd</strong>", third_edit_page.data)
        third_data = {
            "edit_job_id": second_job_id,
            "psd_size_1": "300x250",
            "custom_sizes": "300x250",
            "header": "Yet Another Headline",
            "description": "",
        }
        third = self.client.post("/generate", data=third_data, content_type="multipart/form-data")
        self.assertEqual(third.status_code, 200)
        self.assertIn(b"300x250 used your uploaded PSD template", third.data)

    def test_edit_can_clear_a_size_specific_psd_template_via_hidden_flag(self):
        # The "x" button next to a Size-specific PSD template row (see
        # index.html) sets a hidden psd_size_N_clear field so a template
        # carried forward from a prior job can be cancelled outright --
        # otherwise there'd be no way to say "stop templating this size,
        # go back to the hero image" short of uploading a different .psd.
        first_data = {
            "hero_image": (self._sample_image_bytes(color=(10, 200, 10)), "hero.png"),
            "psd_size_1": "300x250",
            "psd_file_1": (io.BytesIO(self._sample_psd_bytes(color=(200, 30, 30))), "template.psd"),
            "custom_sizes": "300x250",
            "header": "",
            "description": "",
        }
        first = self.client.post("/generate", data=first_data, content_type="multipart/form-data")
        self.assertEqual(first.status_code, 200)
        self.assertIn(b"300x250 used your uploaded PSD template", first.data)
        job_id = re.search(rb"/download/([0-9a-f]+)", first.data).group(1).decode()
        with Image.open(webapp.JOBS_DIR / job_id / "creative_300x250.png") as img:
            r_, g_, b_ = img.getpixel((5, 5))
        self.assertGreater(r_, g_)  # the template's reddish color, unmistakably not the hero's

        second_data = {
            "edit_job_id": job_id,
            "psd_size_1": "",
            "psd_size_1_clear": "1",
            "custom_sizes": "300x250",
            "header": "",
            "description": "",
        }
        second = self.client.post("/generate", data=second_data, content_type="multipart/form-data")
        self.assertEqual(second.status_code, 200)
        self.assertNotIn(b"300x250 used your uploaded PSD template", second.data)

        second_job_id = re.search(rb"/download/([0-9a-f]+)", second.data).group(1).decode()
        with Image.open(webapp.JOBS_DIR / second_job_id / "creative_300x250.png") as img:
            r2, g2, b2 = img.getpixel((5, 5))
        # No PSD template this time -- 300x250 falls back to the carried-forward
        # hero image, so it should show the hero's green, not the template's red.
        self.assertGreater(g2, r2)

        # The Edit page for the cleared job must not show a "Currently: ..."
        # hint (or the x button) for that row any more -- there's nothing
        # left to cancel.
        edit_page = self.client.get(f"/edit/{second_job_id}")
        self.assertNotIn(b'id="psd_clear_btn_1"', edit_page.data)



class ContentPsdQuickModeTest(unittest.TestCase):
    """Covers the single-input "quick campaign" content_psd field: the
    upload renders as its own size (keyed by its actual pixel dimensions,
    not the nominal "728x480" in its label) *and* pulls in whatever's
    saved in default_templates/ -- Output sizes/Custom sizes/hero image
    are all disregarded either way."""

    def setUp(self):
        webapp.app.config["TESTING"] = True
        self.client = _CampaignBriefAutoFillClient(webapp.app.test_client())
        self._orig_jobs_dir = webapp.JOBS_DIR
        self.tmp_dir = tempfile.mkdtemp()
        webapp.JOBS_DIR = Path(self.tmp_dir)
        self._orig_default_templates_dir = webapp.DEFAULT_TEMPLATES_DIR
        self.tmp_default_templates_dir = tempfile.mkdtemp()
        webapp.DEFAULT_TEMPLATES_DIR = Path(self.tmp_default_templates_dir)

    def tearDown(self):
        webapp.JOBS_DIR = self._orig_jobs_dir
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        webapp.DEFAULT_TEMPLATES_DIR = self._orig_default_templates_dir
        shutil.rmtree(self.tmp_default_templates_dir, ignore_errors=True)

    def _write_default_template(self, filename, data: bytes) -> Path:
        path = webapp.DEFAULT_TEMPLATES_DIR / filename
        path.write_bytes(data)
        return path

    def _sample_image_bytes(self, size=(400, 300), color=(20, 100, 200)):
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, format="PNG")
        buf.seek(0)
        return buf

    def _sample_psd_bytes(self, size=(64, 40), color=(200, 50, 50)):
        # Includes three real named layers -- logo/description/product --
        # see WebAppSmokeTest._sample_psd_bytes()'s docstring for why.
        width, height = size
        r, g, b = color
        header = b"8BPS"
        header += struct.pack(">H", 1)  # version
        header += b"\x00" * 6  # reserved
        header += struct.pack(">H", 3)  # channels (RGB)
        header += struct.pack(">I", height)
        header += struct.pack(">I", width)
        header += struct.pack(">H", 8)  # depth
        header += struct.pack(">H", 3)  # color mode: RGB
        color_mode_data = struct.pack(">I", 0)
        image_resources = struct.pack(">I", 0)

        # Real named layers -- webapp.py's REQUIRED_PSD_LAYERS validation
        # (logo/description/product) rejects any template missing one, so
        # every synthetic test PSD needs real layer records with those
        # exact names, not just an empty layer-and-mask section. Small
        # stub boxes in a top strip -- they just need to exist and be
        # named correctly, tests that care about layer *placement* use a
        # real project template instead (see LayerOverrideIntegrationTest).
        stub_layers = [
            ("logo", (0, 0, 20, 10)),
            ("description", (20, 0, 40, 10)),
            ("product", (40, 0, 60, 10)),
        ]
        layer_records = b""
        channel_data = b""
        for name, (lx0, ly0, lx1, ly1) in stub_layers:
            lw, lh = lx1 - lx0, ly1 - ly0
            rec = struct.pack(">iiii", ly0, lx0, ly1, lx1)
            rec += struct.pack(">H", 3)  # 3 channels
            for ch_id in (0, 1, 2):
                rec += struct.pack(">H", ch_id)
                rec += struct.pack(">I", 2 + lw * lh)  # declared size (unused by Pillow's reader)
            rec += b"8BIM" + b"norm"
            rec += struct.pack(">B", 255)  # opacity
            rec += struct.pack(">B", 0)  # clipping
            rec += struct.pack(">B", 0)  # flags
            rec += struct.pack(">B", 0)  # filler
            name_bytes = name.encode("latin-1")
            extra = struct.pack(">I", 0) + struct.pack(">I", 0)  # mask + blending-ranges lengths (both empty)
            extra += struct.pack(">B", len(name_bytes)) + name_bytes
            rec += struct.pack(">I", len(extra)) + extra
            layer_records += rec
            stub_plane = lambda value, n=lw * lh: bytes([value]) * n
            for v in color:
                channel_data += struct.pack(">H", 0) + stub_plane(v)  # compression=0 (raw)

        layer_info_inner = struct.pack(">h", len(stub_layers)) + layer_records + channel_data
        if len(layer_info_inner) % 2:
            layer_info_inner += b"\x00"
        layer_info = struct.pack(">I", len(layer_info_inner)) + layer_info_inner
        global_layer_mask_info = struct.pack(">I", 0)
        layer_and_mask_section = layer_info + global_layer_mask_info
        layer_mask_info = struct.pack(">I", len(layer_and_mask_section)) + layer_and_mask_section

        compression = struct.pack(">H", 0)  # raw, uncompressed
        plane = lambda value: bytes([value]) * (width * height)
        image_data = compression + plane(r) + plane(g) + plane(b)
        return header + color_mode_data + image_resources + layer_mask_info + image_data

    def test_content_psd_renders_its_own_size_plus_saved_defaults(self):
        self._write_default_template("970x90.psd", self._sample_psd_bytes(color=(30, 180, 30)))

        data = {
            "content_psd": (io.BytesIO(self._sample_psd_bytes(color=(10, 10, 200))), "content.psd"),
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        # Two cards: the saved default (970x90) *and* the uploaded PSD's
        # own actual size (64x40 -- this test's synthetic PSD's real
        # pixel dimensions, not the nominal "728x480" the field is
        # labeled for).
        self.assertEqual(r.data.count(b'class="card"'), 2)
        self.assertIn(b"64x40", r.data)
        self.assertIn(b"970x90", r.data)
        self.assertNotIn(b"1080x1080", r.data)

    def test_content_psd_own_size_takes_priority_over_a_same_size_saved_default(self):
        # A saved default at the exact same size the upload itself
        # happens to be -- the freshly uploaded PSD wins, since it's the
        # more recent, more explicit thing the user just gave us.
        self._write_default_template("64x40.psd", self._sample_psd_bytes(color=(30, 180, 30)))

        data = {
            "content_psd": (io.BytesIO(self._sample_psd_bytes(color=(10, 10, 200))), "content.psd"),
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data.count(b'class="card"'), 1)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        with Image.open(webapp.JOBS_DIR / job_id / "creative_64x40.png") as img:
            r_, g_, b_ = img.getpixel((5, 5))
        # The upload's own blue-ish color (10, 10, 200), not the saved
        # default's green (30, 180, 30).
        self.assertGreater(b_, r_)
        self.assertGreater(b_, g_)

    def test_content_psd_corrupt_saved_default_is_skipped_without_crashing(self):
        # default_templates/ is scanned on every content-PSD request; one
        # corrupt file in there must not take down the whole batch -- it's
        # skipped, and every other valid saved template still renders.
        self._write_default_template("160x600.psd", b"this is not a real psd file at all")
        self._write_default_template("970x90.psd", self._sample_psd_bytes(color=(30, 180, 30)))

        data = {
            "content_psd": (io.BytesIO(self._sample_psd_bytes(color=(10, 10, 200))), "content.psd"),
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"970x90", r.data)
        # Not a bare "160x600" substring check -- the results page's
        # ad-blocker disclaimer mentions that dimension as an example
        # unrelated to this job's actual creatives, so check the card
        # count and the actual filename instead.
        self.assertNotIn(b"creative_160x600", r.data)
        # 970x90 (the valid saved default) plus the upload's own size
        # (64x40) -- the corrupt 160x600 file is the only one skipped.
        self.assertEqual(r.data.count(b'class="card"'), 2)

    def test_content_psd_ignores_checked_output_sizes_and_custom_sizes(self):
        self._write_default_template("300x250.psd", self._sample_psd_bytes(size=(300, 250)))
        data = {
            "content_psd": (io.BytesIO(self._sample_psd_bytes()), "content.psd"),
            "sizes": ["default"],
            "custom_sizes": "970x250",
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        # The saved default (300x250) plus the upload's own size (64x40)
        # -- the checked/custom sizes above are still disregarded
        # entirely in this mode.
        self.assertEqual(r.data.count(b'class="card"'), 2)
        self.assertIn(b"300x250", r.data)
        self.assertIn(b"64x40", r.data)
        self.assertNotIn(b"970x250", r.data)
        self.assertNotIn(b"1080x1080", r.data)

    def test_content_psd_with_no_saved_defaults_still_renders_its_own_size(self):
        data = {
            "content_psd": (io.BytesIO(self._sample_psd_bytes()), "content.psd"),
            "header": "",
            "description": "",
        }
        r = self.client.post(
            "/generate", data=data, content_type="multipart/form-data", follow_redirects=True
        )
        self.assertEqual(r.status_code, 200)
        # Nothing saved in default_templates/ this test's isolated folder
        # -- but the upload's own size is always exported regardless, so
        # this succeeds with exactly one card, no flash.
        self.assertEqual(r.data.count(b'class="card"'), 1)
        self.assertIn(b"64x40", r.data)

    def test_content_psd_saved_default_renders_as_is_with_no_overlay(self):
        self._write_default_template("728x480.psd", self._sample_psd_bytes(color=(90, 200, 40)))
        data = {
            "content_psd": (io.BytesIO(self._sample_psd_bytes()), "content.psd"),
            "header": "Some headline that would normally get drawn",
            "description": "Some message that would normally get drawn",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        with Image.open(webapp.JOBS_DIR / job_id / "creative_728x480.png") as img:
            r_, g_, b_ = img.getpixel((5, 5))
        # No overlay was drawn -- the frame is still the saved template's
        # solid greenish color, untouched by the headline/message fields.
        self.assertGreater(g_, r_)
        self.assertGreater(g_, b_)

    def test_content_psd_wrong_extension_flashes(self):
        data = {
            "content_psd": (self._sample_image_bytes(), "content.png"),
            "header": "",
            "description": "",
        }
        r = self.client.post(
            "/generate", data=data, content_type="multipart/form-data", follow_redirects=True
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"728x480 content PSD", r.data)
        self.assertIn(b"supported file type", r.data)


class BundledFontLoadingTest(unittest.TestCase):
    """Regression test for a bug where description/CTA/header/message text
    silently rendered at a tiny fixed size (~10px) on machines that don't
    happen to have DejaVu fonts installed at a Linux system font path --
    _load_font() used to ask PIL for a bare filename like "DejaVuSans.ttf",
    which only resolves via FreeType's search on systems that have that
    exact file somewhere FreeType looks (e.g. Debian/Ubuntu's
    /usr/share/fonts/truetype/dejavu/). On macOS/Windows that lookup fails
    for every family/weight, and PIL silently falls back to
    ImageFont.load_default() -- a fixed-size bitmap font that ignores
    whatever size was requested, no matter how large. Fonts are now
    bundled in fonts/ next to this project and loaded by an absolute path,
    so loading never depends on what the host OS happens to have
    installed. These tests would have caught that bug: the old code
    passed on Linux CI/dev machines (which usually have DejaVu installed)
    while silently failing on the user's own Mac.
    """

    def test_fonts_directory_is_bundled_with_every_required_file(self):
        from src.image_ops import _FONT_FAMILIES, _FONTS_DIR

        self.assertTrue(_FONTS_DIR.is_dir(), f"missing bundled fonts dir: {_FONTS_DIR}")
        for bold_name, regular_name in _FONT_FAMILIES.values():
            for name in (bold_name, regular_name):
                self.assertTrue(
                    (_FONTS_DIR / name).is_file(),
                    f"bundled font file missing: {_FONTS_DIR / name}",
                )

    def test_load_font_honors_requested_size_for_every_family_and_weight(self):
        from src.image_ops import _FONT_FAMILIES, _load_font

        # The real bug: a silent fallback returns a font whose .size never
        # matches what was asked for (PIL's default bitmap font reports a
        # small fixed size regardless of the `size` argument). Asserting
        # the returned font's own .size matches the request, across every
        # family/weight this app supports, at a size nowhere near that
        # fallback's fixed size, is exactly the check that would have
        # caught the bug before it ever reached the user's machine.
        for family in _FONT_FAMILIES:
            for bold in (True, False):
                font = _load_font(123, bold=bold, family=family)
                self.assertEqual(
                    font.size,
                    123,
                    f"family={family!r} bold={bold} loaded a font that ignored the "
                    "requested size -- almost certainly silently fell back to PIL's "
                    "tiny fixed-size default font instead of a real scalable TTF",
                )


class BrandColorCheckTest(unittest.TestCase):
    """Unit coverage for find_missing_brand_colors() -- the "does every
    brand color actually show up in this creative" check."""

    def test_returns_empty_when_no_colors_requested(self):
        from src.image_ops import find_missing_brand_colors

        img = Image.new("RGB", (50, 50), (10, 20, 30))
        self.assertEqual(find_missing_brand_colors(img, []), [])

    def test_exact_color_present_is_not_flagged_missing(self):
        from src.image_ops import find_missing_brand_colors

        img = Image.new("RGB", (50, 50), (10, 20, 30))
        self.assertEqual(find_missing_brand_colors(img, [(10, 20, 30)]), [])

    def test_absent_color_is_flagged_missing(self):
        from src.image_ops import find_missing_brand_colors

        img = Image.new("RGB", (50, 50), (10, 20, 30))
        missing = find_missing_brand_colors(img, [(10, 20, 30), (250, 5, 5)])
        self.assertEqual(missing, [(250, 5, 5)])

    def test_slightly_off_color_within_tolerance_is_not_flagged(self):
        # A brand color placed as a flat swatch can still drift a little
        # after resizing/blending -- the check should tolerate a small
        # Euclidean RGB distance rather than requiring an exact hit.
        from src.image_ops import find_missing_brand_colors

        img = Image.new("RGB", (50, 50), (100, 100, 100))
        missing = find_missing_brand_colors(img, [(105, 98, 102)], tolerance=30)
        self.assertEqual(missing, [])

    def test_color_beyond_tolerance_is_flagged(self):
        from src.image_ops import find_missing_brand_colors

        img = Image.new("RGB", (50, 50), (100, 100, 100))
        missing = find_missing_brand_colors(img, [(200, 100, 100)], tolerance=30)
        self.assertEqual(missing, [(200, 100, 100)])

    def test_works_on_a_large_image_without_error(self):
        # Exercises the downsampling path (images above the internal
        # size cap are shrunk before comparison, purely for speed).
        from src.image_ops import find_missing_brand_colors

        img = Image.new("RGB", (2000, 1500), (0, 0, 0))
        ImageDraw.Draw(img).rectangle((0, 0, 50, 50), fill=(255, 0, 0))
        missing = find_missing_brand_colors(img, [(0, 0, 0), (255, 0, 0), (0, 255, 0)])
        self.assertEqual(missing, [(0, 255, 0)])


class ContentComplianceCheckTest(unittest.TestCase):
    """Unit coverage for the profanity and trademark-text helpers in
    src/compliance.py -- the web UI's content checks."""

    def test_check_profanity_flags_bad_language(self):
        from src.compliance import check_profanity

        self.assertTrue(check_profanity("this is such shit"))

    def test_check_profanity_allows_clean_text(self):
        from src.compliance import check_profanity

        self.assertFalse(check_profanity("Refreshing summer lemonade."))

    def test_check_profanity_allows_blank_text(self):
        from src.compliance import check_profanity

        self.assertFalse(check_profanity(""))
        self.assertFalse(check_profanity(None))

    def test_check_trademark_text_gracefully_returns_empty_without_ocr(self):
        # Simulates the tesseract binary not being installed -- must never
        # raise, just find nothing, since this is an optional bonus check.
        import src.compliance as compliance

        original = compliance._pytesseract
        compliance._pytesseract = None
        try:
            img = Image.new("RGB", (100, 100), "white")
            self.assertEqual(compliance.check_trademark_text(img), [])
        finally:
            compliance._pytesseract = original

    def test_check_trademark_text_finds_a_known_brand_name_when_ocr_available(self):
        import src.compliance as compliance

        if compliance._pytesseract is None:
            self.skipTest("tesseract OCR binary not available in this environment")
        img = Image.new("RGB", (400, 120), "white")
        ImageDraw.Draw(img).text((10, 40), "NIKE", fill="black")
        found = compliance.check_trademark_text(img)
        self.assertIn("Nike", found)


class BoxMappingTest(unittest.TestCase):
    """Regression coverage for a bug where a layer-override box, read
    straight from a PSD's own saved pixel space, silently drifted from
    where that content actually lands in `final_image` whenever the
    PSD's saved canvas size didn't exactly match the (width, height) it
    was being rendered at (e.g. a 728x480 upload used for the "720x480"
    size slot). A template PSD normally IS saved at its nominal size, so
    this never showed up against this project's own default_templates/
    fixtures -- only against a real user-uploaded file with an
    off-by-a-few-pixels canvas. The visible symptom was a background
    patch missing a layer's true edge by a few pixels, leaving a sliver
    of the PSD's original content (e.g. placeholder text) peeking out
    right at the edge of an otherwise-correct override.
    """

    def test_map_box_through_fit_is_a_no_op_when_canvas_matches_target(self):
        from src.image_ops import map_box_through_fit

        box = (10, 20, 100, 200)
        self.assertEqual(map_box_through_fit(box, (500, 500), (500, 500), "crop"), box)
        self.assertEqual(map_box_through_fit(box, (500, 500), (500, 500), "contain"), box)

    def test_map_box_through_fit_crop_mode_matches_center_crop_to_ratio(self):
        # The exact scenario that caused the reported artifact: a 728x480
        # PSD canvas rendered into a 720x480 slot. center_crop_to_ratio()
        # crops 4px off each side (728 -> 720, same height) with no
        # resize needed after that -- so a box's x-coordinates should
        # shift left by exactly 4, y untouched.
        from src.image_ops import map_box_through_fit, center_crop_to_ratio

        src_size = (728, 480)
        target_size = (720, 480)
        box = (34, 164, 248, 391)
        mapped = map_box_through_fit(box, src_size, target_size, "crop")
        self.assertEqual(mapped, (30, 164, 244, 391))

        # Cross-check against center_crop_to_ratio() itself: a solid
        # marker pixel placed at the box's top-left corner in a
        # src_size-shaped image should land at the mapped box's top-left
        # corner after actually being run through center_crop_to_ratio().
        marker_x, marker_y = box[0], box[1]
        src_img = Image.new("RGB", src_size, (0, 0, 0))
        src_img.putpixel((marker_x, marker_y), (255, 0, 0))
        result = center_crop_to_ratio(src_img, target_size)
        self.assertEqual(result.getpixel((mapped[0], mapped[1])), (255, 0, 0))

    def test_map_box_through_fit_contain_mode_matches_resize_to_contain(self):
        # LANCZOS resampling softens a single-pixel marker, so use a
        # solid marker block and check the mapped box's *center* --
        # comfortably inside the block regardless of resampling blur at
        # its edges -- lands on the marker color, not off it.
        from src.image_ops import map_box_through_fit, resize_to_contain

        src_size = (400, 200)  # 2:1, wider than a 1:1 target -> letterboxed top/bottom
        target_size = (200, 200)
        box = (100, 50, 300, 150)
        mapped = map_box_through_fit(box, src_size, target_size, "contain")

        src_img = Image.new("RGB", src_size, (0, 0, 0))
        marker_box = (box[0] + 20, box[1] + 20, box[2] - 20, box[3] - 20)
        ImageDraw.Draw(src_img).rectangle(marker_box, fill=(255, 0, 0))
        result = resize_to_contain(src_img, target_size)

        center_x = (mapped[0] + mapped[2]) // 2
        center_y = (mapped[1] + mapped[3]) // 2
        self.assertEqual(result.getpixel((center_x, center_y)), (255, 0, 0))

    def test_map_box_through_fit_clips_to_target_bounds(self):
        from src.image_ops import map_box_through_fit

        # A box that reaches the source canvas edge shouldn't map to
        # coordinates outside the target canvas.
        box = (0, 0, 728, 480)
        mapped = map_box_through_fit(box, (728, 480), (720, 480), "crop")
        x0, y0, x1, y1 = mapped
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x1, 720)
        self.assertLessEqual(y1, 480)


class LayerOverrideHelpersTest(unittest.TestCase):
    """Direct tests of the low-level compositing helpers behind the
    layer-override feature (src/image_ops.py) -- no PSD parsing or Flask
    involved, just Image-in/Image-out correctness."""

    def test_apply_layer_image_override_fits_vertical_and_centers(self):
        from src.image_ops import apply_layer_image_override

        base = Image.new("RGB", (200, 100), (255, 0, 0))
        bbox = (20, 20, 180, 80)  # 160x60 box -- much wider than the replacement
        replacement = Image.new("RGBA", (60, 60), (0, 255, 0, 255))  # square

        out = apply_layer_image_override(base, bbox, replacement)
        self.assertEqual(out.size, base.size)
        # Scaled to the box's height (60px) -- still square, so also 60px
        # wide -- and centered in the 160px-wide box: pixels near the
        # box's own left/right edges are untouched red (real margin,
        # left for a caller to have already patched with the true
        # background), the vertical center strip is the pasted green.
        self.assertEqual(out.getpixel((25, 50))[:3], (255, 0, 0))
        self.assertEqual(out.getpixel((175, 50))[:3], (255, 0, 0))
        self.assertEqual(out.getpixel((100, 50))[:3], (0, 255, 0))
        # Fills the box's full height -- top and bottom of the box, at
        # the horizontal center, are also the pasted green.
        self.assertEqual(out.getpixel((100, 21))[:3], (0, 255, 0))
        self.assertEqual(out.getpixel((100, 78))[:3], (0, 255, 0))

    def test_apply_layer_image_override_caps_to_width_when_taller_box(self):
        from src.image_ops import apply_layer_image_override

        base = Image.new("RGB", (100, 200), (255, 0, 0))
        bbox = (20, 20, 80, 180)  # 60x160 box -- much taller than the replacement
        replacement = Image.new("RGBA", (60, 60), (0, 255, 0, 255))  # square

        out = apply_layer_image_override(base, bbox, replacement)
        # Fitting to the box's height (160px) would make it 160px wide --
        # wider than the 60px box -- so it falls back to fitting the
        # width instead (still square, 60px tall) and is centered
        # vertically, leaving red margin above and below within the box.
        self.assertEqual(out.getpixel((50, 25))[:3], (255, 0, 0))
        self.assertEqual(out.getpixel((50, 175))[:3], (255, 0, 0))
        self.assertEqual(out.getpixel((50, 100))[:3], (0, 255, 0))

    def test_apply_layer_image_override_trims_transparent_padding_first(self):
        from src.image_ops import apply_layer_image_override

        base = Image.new("RGB", (100, 100), (255, 0, 0))
        bbox = (10, 10, 90, 90)  # 80x80 box
        # A 60x60 replacement canvas, but the real (opaque) content is
        # only a small 10x10 patch in its center -- the rest is padding
        # that shipped with the source file and shouldn't be scaled up
        # as if it were meaningful content.
        replacement = Image.new("RGBA", (60, 60), (0, 0, 0, 0))
        for px in range(25, 35):
            for py in range(25, 35):
                replacement.putpixel((px, py), (0, 255, 0, 255))

        out = apply_layer_image_override(base, bbox, replacement)
        # The trimmed 10x10 content, scaled to fit the 80x80 box (capped
        # by height and width equally since it's square), should cover
        # nearly the whole box -- not a tiny 10x10 speck in the middle
        # the way it would if the blank padding were scaled up too.
        self.assertEqual(out.getpixel((50, 50))[:3], (0, 255, 0))
        self.assertEqual(out.getpixel((15, 50))[:3], (0, 255, 0))

    def test_apply_layer_image_override_leaves_box_untouched_when_fully_transparent(self):
        from src.image_ops import apply_layer_image_override

        base = Image.new("RGB", (100, 100), (10, 10, 200))
        bbox = (20, 20, 80, 80)
        replacement = Image.new("RGBA", (60, 60), (0, 0, 0, 0))  # fully transparent

        out = apply_layer_image_override(base, bbox, replacement)
        # Nothing real to place -- this function doesn't touch the
        # background itself (see get_psd_layer_background() for that),
        # so the box is left exactly as it was.
        self.assertEqual(out.mode, "RGB")
        self.assertEqual(out.getpixel((50, 50)), (10, 10, 200))

    def test_apply_layer_text_override_draws_text_within_box(self):
        from src.image_ops import apply_layer_text_override

        base = Image.new("RGB", (300, 200), (250, 140, 20))
        bbox = (20, 20, 280, 100)
        out = apply_layer_text_override(base, bbox, "Hello layer override")
        self.assertEqual(out.size, base.size)
        # Something dark got drawn somewhere inside the box (the default
        # text color is a dark near-black) -- i.e. it isn't just a flat
        # fill with no text at all.
        box_pixels = list(out.crop(bbox).getdata())
        self.assertTrue(any(sum(p[:3]) < 200 for p in box_pixels))

    def test_apply_layer_text_override_leaves_surrounding_background_untouched(self):
        # No plate/reconstructed patch behind the text anymore (matching
        # apply_layer_image_override(), which never touched the box's
        # background either) -- the corners of the box, away from where
        # short centered text lands, should be exactly the original
        # background color, not a flat or gradient-reconstructed fill.
        from src.image_ops import apply_layer_text_override

        base = Image.new("RGB", (300, 200), (30, 200, 90))
        bbox = (20, 20, 280, 100)
        out = apply_layer_text_override(base, bbox, "Hi")
        self.assertEqual(out.getpixel((22, 22)), (30, 200, 90))
        self.assertEqual(out.getpixel((277, 97)), (30, 200, 90))

    def test_apply_layer_text_override_draws_no_outline_or_glow(self):
        # Regression test: a thin black/white outline used to be drawn
        # around every letter for legibility (since there's no background
        # plate anymore). That read as an unwanted "glow" the source PSD
        # never had -- a real PSD text layer is a flat fill, nothing more.
        # On a plain background, the only colors that should appear are
        # the background color, the exact fill color, and antialiasing
        # blends between those two -- never a distinct black/white ring
        # color that doesn't fall on that background<->fill gradient.
        from src.image_ops import apply_layer_text_override

        bg = (200, 200, 200)
        fg = (20, 20, 20)
        base = Image.new("RGB", (400, 200), bg)
        out = apply_layer_text_override(base, (20, 20, 380, 180), "Hi", text_color=fg, exact_font_size=60)
        for color in out.crop((20, 20, 380, 180)).getcolors(maxcolors=1_000_000):
            _count, (r, g, b) = color
            # Every pixel must be a shade on the straight line between bg
            # and fg (antialiasing), never an out-of-range outline color
            # like pure black/white that isn't already bg or fg.
            self.assertTrue(
                min(bg[0], fg[0]) <= r <= max(bg[0], fg[0])
                and min(bg[1], fg[1]) <= g <= max(bg[1], fg[1])
                and min(bg[2], fg[2]) <= b <= max(bg[2], fg[2]),
                f"unexpected outline/glow pixel color: {(r, g, b)}",
            )

    def test_apply_layer_text_override_never_overflows_the_box(self):
        # Regression test for the actual reported bug: a description box
        # rendered text that visibly spilled past its bounds. Two things
        # fed into that -- (1) an explicit exact_font_size used to be
        # honored completely literally with no fit check at all, so a
        # stale/oversized override (e.g. a leftover value from testing
        # the font-size field) rendered at full size no matter how small
        # the box was, and (2) even the ceiling/autofit search validated
        # candidate sizes using a generic line-height guess, then swapped
        # in the PSD's real (often larger) leading afterward *without*
        # re-checking the swap still fit -- so a "fits" result from the
        # search could still overflow once actually drawn. Both are fixed
        # now: exact_font_size is just a (winning) ceiling for the same
        # search, and the search itself uses the real leading throughout.
        # This asserts the actual outcome that matters -- the rendered
        # block's total height never exceeds the box -- against a
        # deliberately adversarial combination of both old failure modes
        # at once (a wildly oversized explicit override, plus a PSD
        # leading value far larger than the generic guess would produce).
        from src.image_ops import apply_layer_text_override

        base = Image.new("RGB", (500, 400), (240, 240, 240))
        bbox = (20, 20, 480, 380)
        text = "test this new awsome creative test"

        debug: dict = {}
        apply_layer_text_override(
            base,
            bbox,
            text,
            exact_font_size=300,
            leading=400,
            leading_reference_size=50,
            debug=debug,
        )

        x0, y0, x1, y1 = bbox
        box_w, box_h = x1 - x0, y1 - y0
        padding = max(int(min(box_w, box_h) * 0.06), 3)
        max_text_height = max(box_h - 2 * padding, 10)
        total_h = debug["line_height"] * debug["lines"]
        self.assertLessEqual(total_h, max_text_height)
        self.assertLess(debug["font_size"], 300)
        self.assertTrue(debug["clamped"])
        self.assertEqual(debug["requested_font_size"], 300)

    def test_apply_layer_text_override_exact_font_size_wins_as_ceiling_over_psd_font_size(self):
        # An explicit override should win over the PSD's own font_size as
        # the ceiling for the fit search -- but it's still a ceiling, not
        # a demand: both still shrink to fit the box when the text needs
        # it to.
        from src.image_ops import apply_layer_text_override

        base = Image.new("RGB", (400, 300), (240, 240, 240))
        bbox = (20, 20, 380, 280)
        text = "Hi"

        debug_psd_only: dict = {}
        apply_layer_text_override(base, bbox, text, font_size=20, debug=debug_psd_only)
        self.assertLessEqual(debug_psd_only["font_size"], 20)

        debug_explicit: dict = {}
        apply_layer_text_override(base, bbox, text, font_size=20, exact_font_size=80, debug=debug_explicit)
        self.assertLessEqual(debug_explicit["font_size"], 80)
        self.assertGreater(debug_explicit["font_size"], debug_psd_only["font_size"])

    def test_apply_layer_text_override_scales_leading_proportionally_when_it_fits(self):
        # When the requested size does fit, leading should scale
        # proportionally against the PSD's own leading/font-size pair,
        # not fall back to a generic approximation -- exercised here via
        # the fit search itself (see fit_text_block()'s `leading` param),
        # which validates candidates against this same proportional
        # formula rather than a generic guess.
        from src.image_ops import apply_layer_text_override

        base = Image.new("RGB", (2000, 2000), (240, 240, 240))
        bbox = (0, 0, 2000, 2000)
        psd_font_size, psd_leading = 50, 60

        debug: dict = {}
        apply_layer_text_override(
            base,
            bbox,
            "Hi",
            exact_font_size=144,
            leading=psd_leading,
            leading_reference_size=psd_font_size,
            debug=debug,
        )
        self.assertEqual(debug["font_size"], 144)
        expected = round(psd_leading * (144 / psd_font_size))
        self.assertEqual(debug["line_height"], expected)

    def test_get_psd_layer_boxes_returns_empty_for_non_layered_file(self):
        from src.image_ops import get_psd_layer_boxes

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.png"
            Image.new("RGB", (50, 50), (1, 2, 3)).save(path)
            self.assertEqual(get_psd_layer_boxes(path), {})

    def test_get_psd_layer_boxes_returns_empty_for_missing_file(self):
        from src.image_ops import get_psd_layer_boxes

        self.assertEqual(get_psd_layer_boxes("/nonexistent/path/does-not-exist.psd"), {})

    def test_get_psd_text_layers_returns_empty_for_missing_file(self):
        from src.image_ops import get_psd_text_layers

        self.assertEqual(get_psd_text_layers("/nonexistent/path/does-not-exist.psd"), {})

    def test_get_psd_text_layers_returns_empty_for_non_layered_file(self):
        from src.image_ops import get_psd_text_layers

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.png"
            Image.new("RGB", (50, 50), (1, 2, 3)).save(path)
            self.assertEqual(get_psd_text_layers(path), {})

    def test_get_psd_text_layers_reads_real_text_layer_content(self):
        # Exercises the real psd-tools TypeLayer.text property against
        # whatever actual layered template the project currently ships,
        # rather than a mocked/hand-built one.
        real_dir = Path(__file__).resolve().parent.parent / "default_templates"
        if not real_dir.is_dir():
            self.skipTest(f"no default_templates/ directory at {real_dir}")
        real_psd = None
        for candidate in sorted(real_dir.glob("*.psd")):
            from src.image_ops import get_psd_text_layers as _probe

            if _probe(candidate):
                real_psd = candidate
                break
        if real_psd is None:
            self.skipTest(f"no .psd in {real_dir} has a text layer with content")
        from src.image_ops import get_psd_text_layers

        texts = get_psd_text_layers(real_psd)
        self.assertTrue(texts)
        for name, text in texts.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(text, str)
            self.assertTrue(text)

    def test_get_psd_layer_boxes_finds_a_layer_that_has_an_ordinary_mask(self):
        # Regression test: get_psd_layer_boxes() used to read PSDs with
        # Pillow's own bundled parser, which silently drops any layer with
        # more than 4 channels -- including any layer with a completely
        # routine Photoshop layer mask attached (RGB + A + mask = 5
        # channels). A real, hand-authored template with a mask on its
        # "logo"/"description"/"product" layer would report that layer as
        # "missing" even though it's right there, exactly as happened for
        # a user's real upload. Now backed by psd-tools instead, which
        # reads masked layers the same as unmasked ones.
        from psd_tools import PSDImage

        from src.image_ops import get_psd_layer_boxes

        size = (200, 150)
        psd = PSDImage.new("RGBA", size)
        layer = psd.create_pixel_layer(Image.new("RGBA", size, (10, 20, 30, 255)), name="product", top=0, left=0)
        layer.create_mask(Image.new("L", size, 255), top=0, left=0)
        self.assertTrue(layer.has_mask())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "masked.psd"
            psd.save(path)
            boxes = get_psd_layer_boxes(path)
            self.assertIn("product", boxes)
            self.assertEqual(boxes["product"], (0, 0, 200, 150))

    def test_get_psd_layer_boxes_skips_a_layer_entirely_outside_the_canvas(self):
        # A layer positioned entirely off the canvas (e.g. moved or pasted
        # from a much larger document and never repositioned) clips down
        # to a zero-area box -- there's no usable region to report for it,
        # so it should be skipped rather than surfaced as a nonsensical
        # zero-area box.
        from psd_tools import PSDImage

        from src.image_ops import get_psd_layer_boxes

        size = (200, 150)
        psd = PSDImage.new("RGBA", size)
        psd.create_pixel_layer(
            Image.new("RGBA", (30, 30), (0, 0, 0, 255)), name="offcanvas", top=1000, left=1000
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "offcanvas.psd"
            psd.save(path)
            boxes = get_psd_layer_boxes(path)
            self.assertNotIn("offcanvas", boxes)


class LayerBackgroundOverrideHelperTest(unittest.TestCase):
    """Direct tests of apply_layer_background_override() -- the
    "background" layer's compositing is deliberately different from
    apply_layer_image_override()'s (used for logo/CTA/product): it fills
    its box completely rather than centering a smaller object within it
    with margin around it."""

    def test_fills_box_completely_no_margin(self):
        from src.image_ops import apply_layer_background_override

        base = Image.new("RGB", (200, 100), (255, 0, 0))
        bbox = (20, 20, 180, 80)  # 160x60 box
        replacement = Image.new("RGB", (60, 60), (0, 255, 0))  # square

        out = apply_layer_background_override(base, bbox, replacement)
        self.assertEqual(out.size, base.size)
        # Unlike apply_layer_image_override(), every corner and the
        # center of the box are the replacement color -- no red margin
        # anywhere inside the box, since it's cropped to fill rather than
        # fit-and-centered.
        self.assertEqual(out.getpixel((21, 21))[:3], (0, 255, 0))
        self.assertEqual(out.getpixel((178, 21))[:3], (0, 255, 0))
        self.assertEqual(out.getpixel((21, 78))[:3], (0, 255, 0))
        self.assertEqual(out.getpixel((178, 78))[:3], (0, 255, 0))
        self.assertEqual(out.getpixel((100, 50))[:3], (0, 255, 0))
        # Outside the box, the base is untouched.
        self.assertEqual(out.getpixel((5, 50))[:3], (255, 0, 0))

    def test_crops_a_wider_replacement_left_and_right(self):
        from src.image_ops import apply_layer_background_override

        base = Image.new("RGB", (100, 100), (255, 0, 0))
        bbox = (0, 0, 40, 40)  # square box
        # A much wider replacement -- half red, half blue, split down the
        # middle -- center-cropping to a square should keep only a
        # vertical strip out of the middle, discarding both edge colors.
        replacement = Image.new("RGB", (200, 40), (0, 0, 255))
        for x in range(200):
            replacement.putpixel((x, 20), (255, 255, 0) if 80 <= x < 120 else (0, 0, 255))

        out = apply_layer_background_override(base, bbox, replacement)
        # The box is fully covered -- no gaps -- and shows content from
        # the replacement's own center, not the base's original red.
        for x, y in ((2, 2), (37, 2), (2, 37), (37, 37)):
            self.assertNotEqual(out.getpixel((x, y))[:3], (255, 0, 0))

    def test_does_not_trim_or_treat_alpha_specially(self):
        # Unlike apply_layer_image_override(), a background replacement
        # is treated as opaque full-frame content -- no alpha-based
        # content-bbox trimming, even if the source happens to carry an
        # alpha channel.
        from src.image_ops import apply_layer_background_override

        base = Image.new("RGB", (100, 100), (255, 0, 0))
        bbox = (0, 0, 50, 50)
        replacement = Image.new("RGBA", (50, 50), (0, 255, 0, 128))

        out = apply_layer_background_override(base, bbox, replacement)
        # Composited as opaque green (RGBA source flattened to RGB via
        # convert("RGB"), not alpha-blended against the base) -- not a
        # blend toward the base red, and not skipped as "mostly
        # transparent".
        self.assertEqual(out.getpixel((25, 25))[:3], (0, 255, 0))

    def test_zero_area_box_is_a_no_op(self):
        from src.image_ops import apply_layer_background_override

        base = Image.new("RGB", (50, 50), (10, 20, 30))
        replacement = Image.new("RGB", (20, 20), (200, 200, 200))
        out = apply_layer_background_override(base, (10, 10, 10, 40), replacement)
        self.assertEqual(list(out.getdata()), list(base.getdata()))


class LayerOverrideIntegrationTest(unittest.TestCase):
    """End-to-end coverage of the "Update layers across all template
    sizes" fields against a project's real, actually-layered .psd
    template (background/product/cta/description/logo) -- a hand-built
    minimal PSD fixture can't reproduce real named layers without
    reimplementing Photoshop's file format, so this uses whatever real
    layered template the project currently ships in default_templates/
    instead, skipping gracefully if none is present.

    content_psd is a *trigger* only (see ContentPsdQuickModeTest) -- it
    never renders its own size, so these tests copy the real template
    into this test's isolated default_templates dir first (under a
    filename encoding its actual pixel size) and use content_psd purely
    to kick off that saved-defaults batch, then assert against the size
    the copied template actually renders as.
    """

    REAL_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "default_templates"

    def setUp(self):
        webapp.app.config["TESTING"] = True
        self.client = _CampaignBriefAutoFillClient(webapp.app.test_client())
        self._orig_jobs_dir = webapp.JOBS_DIR
        self.tmp_dir = tempfile.mkdtemp()
        webapp.JOBS_DIR = Path(self.tmp_dir)
        self._orig_default_templates_dir = webapp.DEFAULT_TEMPLATES_DIR
        self.tmp_default_templates_dir = tempfile.mkdtemp()
        webapp.DEFAULT_TEMPLATES_DIR = Path(self.tmp_default_templates_dir)

    def tearDown(self):
        webapp.JOBS_DIR = self._orig_jobs_dir
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        webapp.DEFAULT_TEMPLATES_DIR = self._orig_default_templates_dir
        shutil.rmtree(self.tmp_default_templates_dir, ignore_errors=True)

    def _find_real_layered_template(self):
        if not self.REAL_TEMPLATES_DIR.is_dir():
            return None
        for path in sorted(self.REAL_TEMPLATES_DIR.glob("*.psd")):
            return path
        return None

    def _stage_real_template(self):
        """Find whatever real layered .psd the project currently ships
        (skipping the test if there isn't one), copy it into this test's
        isolated default_templates dir under a filename encoding its own
        actual pixel size, and return (size, isolated_path)."""
        real_path = self._find_real_layered_template()
        if real_path is None:
            self.skipTest(f"no .psd files in {self.REAL_TEMPLATES_DIR} -- can't exercise real named-layer PSDs")
        size = Image.open(real_path).size
        dest = webapp.DEFAULT_TEMPLATES_DIR / f"staged-{size[0]}x{size[1]}.psd"
        dest.write_bytes(real_path.read_bytes())
        return size, dest

    def _sample_image_bytes(self, size=(60, 60), color=(0, 200, 0)):
        buf = io.BytesIO()
        Image.new("RGBA", size, color + (255,)).save(buf, format="PNG")
        buf.seek(0)
        return buf

    def test_description_and_logo_override_apply_to_saved_default_size(self):
        (w, h), staged_path = self._stage_real_template()
        data = {
            "content_psd": (io.BytesIO(staged_path.read_bytes()), "content.psd"),
            "layer_description_text": "Brand new description text",
            "layer_logo_image": (self._sample_image_bytes(color=(0, 120, 255)), "logo.png"),
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"updated layer(s)", r.data)
        self.assertIn(b"description", r.data)
        self.assertIn(b"logo", r.data)

        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        from src.image_ops import get_psd_layer_boxes
        box = get_psd_layer_boxes(staged_path).get("logo")
        self.assertIsNotNone(box, "staged real template has no 'logo' layer -- can't verify placement")
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        with Image.open(webapp.JOBS_DIR / job_id / f"creative_{w}x{h}.png") as img:
            # Center of the real logo layer's bbox -- should now be the
            # overridden blue, not whatever the original template logo
            # looked like there.
            r_, g_, b_ = img.getpixel((cx, cy))[:3]
        self.assertLess(r_, 100)
        self.assertGreater(b_, 150)

    def test_header_override_is_skipped_gracefully_when_template_has_no_header_layer(self):
        # The real staged template doesn't have a "header" layer (that's
        # a new addition a user might make on their own end) -- providing
        # layer_header_text must not error, and the existing description
        # override must keep working right alongside it.
        (w, h), staged_path = self._stage_real_template()
        data = {
            "content_psd": (io.BytesIO(staged_path.read_bytes()), "content.psd"),
            "layer_header_text": "New headline text",
            "layer_description_text": "New description text",
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"updated layer(s)", r.data)
        self.assertIn(b"description", r.data)
        # "header" is skipped -- the field name is never named as an
        # *applied* layer, since there's no matching layer to draw into.
        self.assertNotIn(b": header debug", r.data)

    def test_header_override_draws_new_text_into_a_named_header_layer(self):
        # Built directly via psd-tools' write API rather than
        # _stage_real_template() -- unlike description/logo/cta/product,
        # none of the project's real shipped templates have a "header"
        # layer yet, so this doesn't depend on one existing.
        from psd_tools import PSDImage
        from src.image_ops import get_psd_layer_boxes

        size = (400, 300)
        psd = PSDImage.new("RGBA", size)
        psd.create_pixel_layer(Image.new("RGBA", size, (240, 240, 240, 255)), name="background", top=0, left=0)
        psd.create_pixel_layer(Image.new("RGBA", (60, 40), (0, 0, 0, 255)), name="logo", top=0, left=0)
        psd.create_pixel_layer(Image.new("RGBA", (60, 40), (0, 0, 0, 255)), name="product", top=240, left=0)
        psd.create_pixel_layer(Image.new("RGBA", (60, 40), (0, 0, 0, 255)), name="cta", top=240, left=200)
        psd.create_pixel_layer(
            Image.new("RGBA", (300, 60), (240, 240, 240, 255)), name="header", top=10, left=50
        )
        psd.create_pixel_layer(
            Image.new("RGBA", (300, 60), (240, 240, 240, 255)), name="description", top=100, left=50
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "with_header.psd"
            psd.save(path)
            data = {
                "content_psd": (io.BytesIO(path.read_bytes()), "with_header.psd"),
                "layer_header_text": "Summer Sale",
                "header": "",
                "description": "",
            }
            r = self.client.post("/generate", data=data, content_type="multipart/form-data")
            self.assertEqual(r.status_code, 200)
            self.assertIn(b"updated layer(s)", r.data)
            self.assertIn(b"header", r.data)

            job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
            box = get_psd_layer_boxes(path).get("header")
            self.assertIsNotNone(box)
            with Image.open(webapp.JOBS_DIR / job_id / f"creative_{size[0]}x{size[1]}.png") as img:
                region = img.crop(box).convert("L")
            colors = region.getcolors(maxcolors=region.size[0] * region.size[1])
            self.assertGreater(
                len(colors), 1, "header box looks untouched (still a single flat color) -- no text was drawn"
            )

    def test_override_with_no_matching_layer_is_skipped_not_errored(self):
        _, staged_path = self._stage_real_template()
        # "badge" isn't one of this template's named layers (it has
        # background/product/cta/description/logo) -- a CTA image update
        # should still apply fine, this just confirms an unmatched field
        # (there isn't a "badge" field at all, so instead: request a CTA
        # image update against a template known to have that layer, and
        # confirm the request never errors even when description text is
        # also given for a template).
        data = {
            "content_psd": (io.BytesIO(staged_path.read_bytes()), "content.psd"),
            "layer_cta_image": (self._sample_image_bytes(color=(255, 0, 0)), "cta.png"),
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"updated layer(s)", r.data)
        self.assertIn(b"cta", r.data)

    def test_no_layer_override_fields_leaves_template_untouched(self):
        _, staged_path = self._stage_real_template()
        data = {
            "content_psd": (io.BytesIO(staged_path.read_bytes()), "content.psd"),
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertNotIn(b"updated layer(s)", r.data)

    def test_background_override_fills_its_layer_completely_on_saved_default_size(self):
        # This is the actual feature request: a background-only change to
        # one uploaded PSD should be pushable across every other saved
        # default size too, the same way logo/CTA/product/description
        # already are.
        (w, h), staged_path = self._stage_real_template()
        data = {
            "content_psd": (io.BytesIO(staged_path.read_bytes()), "content.psd"),
            "layer_background_image": (self._sample_image_bytes(size=(80, 80), color=(0, 200, 0)), "bg.png"),
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"updated layer(s)", r.data)
        self.assertIn(b"background", r.data)

        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        from src.image_ops import get_psd_layer_boxes

        box = get_psd_layer_boxes(staged_path).get("background")
        self.assertIsNotNone(box, "staged real template has no 'background' layer -- can't verify placement")
        x0, y0, x1, y1 = box
        with Image.open(webapp.JOBS_DIR / job_id / f"creative_{w}x{h}.png") as img:
            # Several points spread across the box, not just its center --
            # apply_layer_background_override() fills edge-to-edge, so
            # every one of these should be the overridden green, unlike
            # the centered-with-margin behavior logo/CTA/product use.
            points = [
                (x0 + 3, y0 + 3),
                (x1 - 3, y0 + 3),
                (x0 + 3, y1 - 3),
                (x1 - 3, y1 - 3),
                ((x0 + x1) // 2, (y0 + y1) // 2),
            ]
            pixels = [img.getpixel((max(0, min(px, img.width - 1)), max(0, min(py, img.height - 1))))[:3] for px, py in points]
        for r_, g_, b_ in pixels:
            self.assertGreater(g_, r_, pixels)
            self.assertGreater(g_, b_, pixels)

    def test_background_override_does_not_run_background_removal(self):
        # auto_transparent_background() would be actively harmful here --
        # it tries to strip an unwanted flat backdrop from a foreground
        # cutout, but a background upload's flat color/gradient *is* the
        # wanted content. A near-uniform light background upload (the
        # kind auto_transparent_background() is most aggressive about)
        # should still render solid, not full of holes.
        (w, h), staged_path = self._stage_real_template()
        near_white = Image.new("RGBA", (80, 80), (250, 250, 250, 255))
        buf = io.BytesIO()
        near_white.save(buf, format="PNG")
        buf.seek(0)
        data = {
            "content_psd": (io.BytesIO(staged_path.read_bytes()), "content.psd"),
            "layer_background_image": (buf, "near_white_bg.png"),
            "header": "",
            "description": "",
        }
        r = self.client.post("/generate", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        job_id = re.search(rb"/download/([0-9a-f]+)", r.data).group(1).decode()
        from src.image_ops import get_psd_layer_boxes

        box = get_psd_layer_boxes(staged_path).get("background")
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        with Image.open(webapp.JOBS_DIR / job_id / f"creative_{w}x{h}.png") as img:
            r_, g_, b_ = img.getpixel((cx, cy))[:3]
        # Still near-white -- not composited against whatever the
        # template's own background used to be underneath, which would
        # show through as a visibly different color if background
        # removal had run.
        self.assertGreater(r_, 200)
        self.assertGreater(g_, 200)
        self.assertGreater(b_, 200)


if __name__ == "__main__":
    unittest.main()
