"""Put back, pressed on an Edit page, has to stick.

The Edit page loads a job's files directly rather than through the
remembered form, and submitting an edit carries forward any file the user
did not replace. Both bypassed the "let go of" markers, so a size put back
came straight back: the row still held the upload, and the run promoted it
into default_templates/ again.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import webapp


class EditFlowDropTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._orig_jobs = webapp.JOBS_DIR
        webapp.JOBS_DIR = self.tmp / "jobs"
        self.job = "b" * 32
        uploads = webapp.JOBS_DIR / self.job / "uploads"
        uploads.mkdir(parents=True)
        (uploads / "tester-720x1280.psd").write_text("the template with the bottle in it")
        (webapp.JOBS_DIR / self.job / "form_state.json").write_text(json.dumps({
            "fields": {"product_name": "HydroBoost Sports Drink", "psd_size_1": "720x1280"},
            "files": {"psd_file_1": "tester-720x1280.psd", "upload_hero_image": "hero.png"},
        }))
        webapp._save_preferences({})

    def tearDown(self):
        webapp.JOBS_DIR = self._orig_jobs
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_edit_page_stops_offering_a_dropped_row(self):
        cards = webapp._load_session_campaigns(None, self.job)
        self.assertEqual(cards[0]["prefill_files"].get("psd_file_1"), "tester-720x1280.psd")

        webapp.drop_row_files_for_job(self.job, [1])

        cards = webapp._load_session_campaigns(None, self.job)
        self.assertNotIn("psd_file_1", cards[0]["prefill_files"], "the row should come up empty")
        self.assertEqual(cards[0]["prefill_files"].get("upload_hero_image"), "hero.png",
                         "only the row that was put back is let go of")

    def test_submitting_that_edit_does_not_carry_the_file_forward(self):
        prior_dir = webapp.JOBS_DIR / self.job
        prior_state = json.loads((prior_dir / "form_state.json").read_text())
        uploads = self.tmp / "new-job-uploads"
        uploads.mkdir()

        carried = webapp._carry_forward_upload("psd_file_1", uploads, prior_dir, prior_state)
        self.assertIsNotNone(carried, "without a drop it carries forward as before")

        webapp.drop_row_files_for_job(self.job, [1])
        carried = webapp._carry_forward_upload("psd_file_1", uploads, prior_dir, prior_state)
        self.assertIsNone(carried, "a dropped row must not come back on submit")

        # Everything else on that job still carries forward.
        (prior_dir / "uploads" / "hero.png").write_text("the hero")
        self.assertIsNotNone(
            webapp._carry_forward_upload("upload_hero_image", uploads, prior_dir, prior_state),
            "dropping one row must not stop other files carrying forward",
        )

    def test_a_marker_for_another_job_does_not_touch_this_one(self):
        webapp.drop_row_files_for_job("c" * 32, [1])
        cards = webapp._load_session_campaigns(None, self.job)
        self.assertEqual(cards[0]["prefill_files"].get("psd_file_1"), "tester-720x1280.psd")


if __name__ == "__main__":
    unittest.main()
