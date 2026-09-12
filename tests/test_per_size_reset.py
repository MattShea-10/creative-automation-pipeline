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
