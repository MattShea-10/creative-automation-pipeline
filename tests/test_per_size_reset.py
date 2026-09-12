"""Putting one size's template back to the backup zip.

The card's Reset restores every size and clears the form. This is the
narrowed version: one size back, the rest untouched, the form left alone.
"""

import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import webapp


class PerSizeResetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._orig_templates = webapp.DEFAULT_TEMPLATES_DIR
        self._orig_backups = webapp.TEMPLATE_BACKUPS_DIR
        webapp.DEFAULT_TEMPLATES_DIR = self.tmp / "default_templates"
        webapp.TEMPLATE_BACKUPS_DIR = self.tmp / "_template_backups"
        webapp.DEFAULT_TEMPLATES_DIR.mkdir(parents=True)

        # The master set: what every restore goes back to.
        self.zip_path = webapp.DEFAULT_TEMPLATES_DIR / "template-backup.zip"
        with zipfile.ZipFile(self.zip_path, "w") as zf:
            zf.writestr("tester-720x480.psd", "ORIGINAL 720x480")
            zf.writestr("tester-1080x1080.psd", "ORIGINAL 1080x1080")
            # Finder adds these to a zip of a folder; they are not templates.
            zf.writestr("__MACOSX/._tester-720x480.psd", "junk")

        # A campaign folder where both sizes have been edited since.
        self.folder = webapp.DEFAULT_TEMPLATES_DIR / "Winter Glow 2026" / "HydroBoost"
        self.folder.mkdir(parents=True)
        (self.folder / "tester-720x480.psd").write_text("EDITED 720x480")
        (self.folder / "tester-1080x1080.psd").write_text("EDITED 1080x1080")

    def tearDown(self):
        webapp.DEFAULT_TEMPLATES_DIR = self._orig_templates
        webapp.TEMPLATE_BACKUPS_DIR = self._orig_backups
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_one_size_goes_back_and_the_others_do_not(self):
        restored, _untouched, error = webapp.restore_templates_from_backup(
            dest_dir=self.folder, only_sizes={"720x480"}
        )
        self.assertIsNone(error)
        self.assertEqual(restored, ["tester-720x480.psd"])
        self.assertEqual((self.folder / "tester-720x480.psd").read_text(), "ORIGINAL 720x480")
        # The whole point: the size you did not ask for is still yours.
        self.assertEqual((self.folder / "tester-1080x1080.psd").read_text(), "EDITED 1080x1080")

    def test_the_replaced_file_is_kept(self):
        webapp.restore_templates_from_backup(dest_dir=self.folder, only_sizes={"720x480"})
        kept = list(webapp.TEMPLATE_BACKUPS_DIR.glob("tester-720x480.*.psd"))
        self.assertEqual(len(kept), 1, "the edited file should be stamped and kept")
        self.assertEqual(kept[0].read_text(), "EDITED 720x480")

    def test_a_size_the_zip_does_not_carry_changes_nothing(self):
        restored, _untouched, error = webapp.restore_templates_from_backup(
            dest_dir=self.folder, only_sizes={"160x600"}
        )
        self.assertIsNone(error)
        self.assertEqual(restored, [])
        self.assertEqual((self.folder / "tester-720x480.psd").read_text(), "EDITED 720x480")
        self.assertEqual((self.folder / "tester-1080x1080.psd").read_text(), "EDITED 1080x1080")

    def test_no_filter_still_restores_everything(self):
        # The card's Reset must keep working exactly as it did.
        restored, _untouched, error = webapp.restore_templates_from_backup(dest_dir=self.folder)
        self.assertIsNone(error)
        self.assertEqual(sorted(restored), ["tester-1080x1080.psd", "tester-720x480.psd"])
        self.assertEqual((self.folder / "tester-1080x1080.psd").read_text(), "ORIGINAL 1080x1080")

    def test_the_picker_lists_the_zips_sizes_largest_first(self):
        self.assertEqual(webapp.backup_zip_sizes(), ["1080x1080", "720x480"])

    def test_the_picker_is_empty_without_a_zip(self):
        self.zip_path.unlink()
        self.assertEqual(webapp.backup_zip_sizes(), [])


if __name__ == "__main__":
    unittest.main()


class ForgetPsdRowTest(unittest.TestCase):
    """Putting a size back has to let go of a row still holding an upload
    for that size -- otherwise the row wins on the next run and the
    restore looks like it did nothing, which is how the bug was found.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._orig_jobs = webapp.JOBS_DIR
        webapp.JOBS_DIR = self.tmp / "jobs"
        self.job = "e9b1c91c81a6408ea574c4ce15159543"
        uploads = webapp.JOBS_DIR / self.job / "uploads"
        uploads.mkdir(parents=True)
        (uploads / "tester-720x1280_1.psd").write_text("an uploaded template")
        (webapp.JOBS_DIR / self.job / "form_state.json").write_text(
            '{"files": {"psd_file_1": "tester-720x1280_1.psd"}}'
        )
        self.key = "Winter Glow 2026/HydroBoost Sports Drink"
        self.prefs = {
            "last_job_id": self.job,
            "psd_size_1": "1080x1920",
            "products": {
                self.key: {"last_job_id": self.job, "psd_size_1": "1080x1920"},
                "Other Campaign/Another Product": {"last_job_id": self.job, "psd_size_1": "1080x1920"},
            },
        }
        webapp._save_preferences(self.prefs)

    def tearDown(self):
        webapp.JOBS_DIR = self._orig_jobs
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_row_set_to_that_size_is_let_go(self):
        rows = webapp.forget_psd_row_for_size(self.key, "1080x1920")
        self.assertEqual(rows, [1])
        _job, files = webapp._remembered_files(self.key)
        self.assertEqual(files, {}, "the kept upload should no longer be offered")
        saved = webapp._load_preferences()
        self.assertEqual(saved["products"][self.key]["psd_size_1"], "")

    def test_another_products_row_is_untouched(self):
        webapp.forget_psd_row_for_size(self.key, "1080x1920")
        other = webapp._load_preferences()["products"]["Other Campaign/Another Product"]
        self.assertEqual(other["psd_size_1"], "1080x1920")
        _job, files = webapp._remembered_files("Other Campaign/Another Product")
        self.assertEqual(files, {"psd_file_1": "tester-720x1280_1.psd"})

    def test_a_row_set_to_a_different_size_stays(self):
        rows = webapp.forget_psd_row_for_size(self.key, "720x480")
        self.assertEqual(rows, [])
        _job, files = webapp._remembered_files(self.key)
        self.assertEqual(files, {"psd_file_1": "tester-720x1280_1.psd"})

    def test_the_marker_dies_with_the_run_it_names(self):
        # A later run writes a new job id; the old marker must not hide
        # a file uploaded since.
        webapp.forget_psd_row_for_size(self.key, "1080x1920")
        newer = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        uploads = webapp.JOBS_DIR / newer / "uploads"
        uploads.mkdir(parents=True)
        (uploads / "fresh.psd").write_text("uploaded after the restore")
        (webapp.JOBS_DIR / newer / "form_state.json").write_text('{"files": {"psd_file_1": "fresh.psd"}}')
        prefs = webapp._load_preferences()
        prefs["products"][self.key]["last_job_id"] = newer
        webapp._save_preferences(prefs)
        _job, files = webapp._remembered_files(self.key)
        self.assertEqual(files, {"psd_file_1": "fresh.psd"})


class TemplateSizesStatusTest(unittest.TestCase):
    """Which of a product's templates still match the backup zip. Nine
    identical labels in the picker is how the wrong size got put back."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._orig_templates = webapp.DEFAULT_TEMPLATES_DIR
        webapp.DEFAULT_TEMPLATES_DIR = self.tmp / "default_templates"
        webapp.DEFAULT_TEMPLATES_DIR.mkdir(parents=True)
        webapp._template_status_cache.clear()
        with zipfile.ZipFile(webapp.DEFAULT_TEMPLATES_DIR / "template-backup.zip", "w") as zf:
            zf.writestr("tester-1080x1080.psd", "ORIGINAL 1080x1080")
            zf.writestr("tester-720x1280.psd", "ORIGINAL 720x1280")
            zf.writestr("tester-160x600.psd", "ORIGINAL 160x600")
        self.folder = webapp.DEFAULT_TEMPLATES_DIR / "Winter Glow 2026" / "HydroBoost Sports Drink"
        self.folder.mkdir(parents=True)
        (self.folder / "tester-1080x1080.psd").write_text("ORIGINAL 1080x1080")
        (self.folder / "tester-720x1280.psd").write_text("EDITED -- a photo baked in")
        # 160x600 left absent on purpose.

    def tearDown(self):
        webapp.DEFAULT_TEMPLATES_DIR = self._orig_templates
        webapp._template_status_cache.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_it_says_which_size_is_yours(self):
        status = webapp.template_sizes_status("HydroBoost Sports Drink", "Winter Glow 2026")
        self.assertEqual(
            {e["size"]: e["state"] for e in status},
            {"1080x1080": "same", "720x1280": "changed", "160x600": "missing"},
        )

    def test_largest_first(self):
        status = webapp.template_sizes_status("HydroBoost Sports Drink", "Winter Glow 2026")
        self.assertEqual([e["size"] for e in status], ["1080x1080", "720x1280", "160x600"])

    def test_it_notices_an_edit_after_the_first_look(self):
        webapp.template_sizes_status("HydroBoost Sports Drink", "Winter Glow 2026")
        (self.folder / "tester-1080x1080.psd").write_text("EDITED SINCE")
        status = webapp.template_sizes_status("HydroBoost Sports Drink", "Winter Glow 2026")
        self.assertEqual({e["size"]: e["state"] for e in status}["1080x1080"], "changed")

    def test_no_zip_means_no_list(self):
        (webapp.DEFAULT_TEMPLATES_DIR / "template-backup.zip").unlink()
        webapp._template_status_cache.clear()
        self.assertEqual(webapp.template_sizes_status("HydroBoost Sports Drink", "Winter Glow 2026"), [])

    def test_an_unseeded_folder_is_not_all_missing(self):
        # A card with no product yet points at a folder with no templates.
        # Nothing has been changed there, so nothing should be marked.
        webapp._template_status_cache.clear()
        status = webapp.template_sizes_status("A Product That Has Never Run")
        self.assertTrue(status)
        self.assertEqual({e["state"] for e in status}, {"same"})


class SameShapeGroupTest(unittest.TestCase):
    """Sizes with the same proportions share a template -- an upload for
    one is the template for the others. Putting one back therefore has to
    put the whole shape back, or the twin keeps rendering the old art and
    the restore reads as broken."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._orig_templates = webapp.DEFAULT_TEMPLATES_DIR
        self._orig_backups = webapp.TEMPLATE_BACKUPS_DIR
        webapp.DEFAULT_TEMPLATES_DIR = self.tmp / "default_templates"
        webapp.TEMPLATE_BACKUPS_DIR = self.tmp / "_template_backups"
        webapp.DEFAULT_TEMPLATES_DIR.mkdir(parents=True)
        webapp._template_status_cache.clear()
        with zipfile.ZipFile(webapp.DEFAULT_TEMPLATES_DIR / "template-backup.zip", "w") as zf:
            zf.writestr("tester-1080x1920.psd", "ORIGINAL 9:16 tall")
            zf.writestr("tester-720x1280.psd", "ORIGINAL 9:16 small")
            zf.writestr("tester-1080x1080.psd", "ORIGINAL square")
            zf.writestr("tester-720x480.psd", "ORIGINAL 3:2")
        self.folder = webapp.DEFAULT_TEMPLATES_DIR / "Winter Glow 2026" / "HydroBoost Sports Drink"
        self.folder.mkdir(parents=True)
        for name in ("tester-1080x1920.psd", "tester-720x1280.psd"):
            (self.folder / name).write_text("EDITED -- the bottle photo")
        (self.folder / "tester-1080x1080.psd").write_text("ORIGINAL square")
        (self.folder / "tester-720x480.psd").write_text("ORIGINAL 3:2")

    def tearDown(self):
        webapp.DEFAULT_TEMPLATES_DIR = self._orig_templates
        webapp.TEMPLATE_BACKUPS_DIR = self._orig_backups
        webapp._template_status_cache.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_group_is_every_size_of_that_shape(self):
        self.assertEqual(sorted(webapp.sizes_sharing_shape("1080x1920")), ["1080x1920", "720x1280"])
        self.assertEqual(webapp.sizes_sharing_shape("1080x1080"), ["1080x1080"])
        self.assertEqual(webapp.sizes_sharing_shape("720x480"), ["720x480"])

    def test_putting_one_back_puts_its_twin_back_too(self):
        restored, _untouched, error = webapp.restore_templates_from_backup(
            dest_dir=self.folder, only_sizes=set(webapp.sizes_sharing_shape("1080x1920"))
        )
        self.assertIsNone(error)
        self.assertEqual(sorted(restored), ["tester-1080x1920.psd", "tester-720x1280.psd"])
        self.assertEqual((self.folder / "tester-720x1280.psd").read_text(), "ORIGINAL 9:16 small")
        self.assertEqual((self.folder / "tester-1080x1920.psd").read_text(), "ORIGINAL 9:16 tall")

    def test_other_shapes_are_left_alone(self):
        (self.folder / "tester-1080x1080.psd").write_text("EDITED square")
        webapp.restore_templates_from_backup(
            dest_dir=self.folder, only_sizes=set(webapp.sizes_sharing_shape("1080x1920"))
        )
        self.assertEqual((self.folder / "tester-1080x1080.psd").read_text(), "EDITED square",
                         "a square template is nobody's 9:16 twin")

    def test_the_picker_names_the_twin(self):
        status = {e["size"]: e for e in webapp.template_sizes_status("HydroBoost Sports Drink", "Winter Glow 2026")}
        self.assertEqual(status["1080x1920"]["with"], ["720x1280"])
        self.assertEqual(status["720x1280"]["with"], ["1080x1920"])
        self.assertEqual(status["1080x1080"]["with"], [])

    def test_a_size_the_zip_never_heard_of_is_just_itself(self):
        self.assertEqual(webapp.sizes_sharing_shape("999x333"), ["999x333"])
