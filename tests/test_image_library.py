"""Keeping AI runs for later training.

Job folders are pruned on a timer, so a generated image is temporary
unless something copies it out. image_library/ is that copy: image and
caption side by side, plus an index, in the layout a fine-tune expects.

The split matters more than the copying. backdrops/ is the bare artwork
the provider returned and the only thing safe to train on; creatives/ is
the finished ad, whose headline and logo would teach a model to paint
lettering.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import webapp


class ImageLibraryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._orig_library = webapp.IMAGE_LIBRARY_DIR
        webapp.IMAGE_LIBRARY_DIR = self.tmp / "image_library"
        self.job_dir = self.tmp / "job"
        self.job_dir.mkdir()
        self.creatives = []
        for label, ratio in (("1200x1200", "1:1"), ("1200x627", "1.91:1"), ("1080x1920", "9:16")):
            name = f"hydroboost_campaign1_{label}.png"
            (self.job_dir / name).write_bytes(b"PNG-" + label.encode())
            self.creatives.append({"filename": name, "label": label, "ratio": ratio, "name": label})
        self.backdrop = self.job_dir / "ai-campaign.png"
        self.backdrop.write_bytes(b"PNG-the-bare-backdrop")
        self.record = {
            "stamp": "20260912-041530",
            "slug": "hydroboost",
            "job_id": "abcdef0123456789",
            "generated_at": "2026-09-12T04:15:30",
            "product": "HydroBoost Sports Drink",
            "campaign": "Winter Glow 2026",
            "market": "France",
            "provider": "pollinations",
            "prompt": "Professional studio photo of a chilled blue sports drink bottle",
            "prompt_typed": "chilled blue sports drink",
            "copy_language": "fr",
        }

    def tearDown(self):
        webapp.IMAGE_LIBRARY_DIR = self._orig_library
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _backdrops(self):
        return sorted(f.name for f in (webapp.IMAGE_LIBRARY_DIR / "backdrops").glob("*.png"))

    def _creatives(self):
        return sorted(f.name for f in (webapp.IMAGE_LIBRARY_DIR / "creatives").glob("*.png"))

    def test_the_backdrop_is_kept_apart_from_the_finished_ads(self):
        webapp.save_to_image_library(self.job_dir, self.creatives, self.record, backdrop=self.backdrop)
        self.assertEqual(self._backdrops(), ["20260912-041530_hydroboost_backdrop_abcdef01.png"])
        self.assertEqual(self._creatives(), [
            "20260912-041530_hydroboost_1200x1200_abcdef01.png",
            "20260912-041530_hydroboost_1200x627_abcdef01.png",
        ])
        self.assertEqual(
            (webapp.IMAGE_LIBRARY_DIR / "backdrops" / "20260912-041530_hydroboost_backdrop_abcdef01.png").read_bytes(),
            b"PNG-the-bare-backdrop",
            "the training image must be what the provider returned, not a composited ad",
        )

    def test_only_the_two_library_sizes_are_kept(self):
        webapp.save_to_image_library(self.job_dir, self.creatives, self.record, backdrop=self.backdrop)
        self.assertFalse(list((webapp.IMAGE_LIBRARY_DIR / "creatives").glob("*1080x1920*")))

    def test_the_backdrop_caption_does_not_describe_an_ad(self):
        webapp.save_to_image_library(self.job_dir, self.creatives, self.record, backdrop=self.backdrop)
        caption = (webapp.IMAGE_LIBRARY_DIR / "backdrops" / "20260912-041530_hydroboost_backdrop_abcdef01.txt").read_text()
        self.assertIn("chilled blue sports drink bottle", caption)
        self.assertNotIn("advertising creative", caption, "a bare backdrop is not a creative")
        creative_caption = (webapp.IMAGE_LIBRARY_DIR / "creatives" / "20260912-041530_hydroboost_1200x1200_abcdef01.txt").read_text()
        self.assertIn("1:1 advertising creative", creative_caption)

    def test_the_index_says_which_is_which(self):
        webapp.save_to_image_library(self.job_dir, self.creatives, self.record, backdrop=self.backdrop)
        entries = [json.loads(l) for l in (webapp.IMAGE_LIBRARY_DIR / "index.jsonl").read_text().strip().splitlines()]
        self.assertEqual([e["kind"] for e in entries], ["backdrop", "creative", "creative"])
        backdrop = entries[0]
        self.assertEqual(backdrop["file"], "backdrops/20260912-041530_hydroboost_backdrop_abcdef01.png")
        self.assertEqual(backdrop["provider"], "pollinations")
        self.assertEqual(backdrop["job_id"], "abcdef0123456789")
        self.assertNotIn("stamp", backdrop, "bookkeeping fields stay out of the dataset record")
        # Same job id on both, so a backdrop can be traced to what was built from it.
        self.assertEqual({e["job_id"] for e in entries}, {"abcdef0123456789"})

    def test_the_same_run_is_not_kept_twice(self):
        webapp.save_to_image_library(self.job_dir, self.creatives, self.record, backdrop=self.backdrop)
        again = webapp.save_to_image_library(self.job_dir, self.creatives, self.record, backdrop=self.backdrop)
        self.assertEqual(again, [])
        self.assertEqual(len(self._backdrops()) + len(self._creatives()), 3)
        self.assertEqual(len((webapp.IMAGE_LIBRARY_DIR / "index.jsonl").read_text().strip().splitlines()), 3)

    def test_a_run_with_no_backdrop_still_keeps_its_creatives(self):
        written = webapp.save_to_image_library(self.job_dir, self.creatives, self.record, backdrop=None)
        self.assertEqual(self._backdrops(), [])
        self.assertEqual(len(written), 2)

    def test_a_missing_render_is_skipped_not_fatal(self):
        (self.job_dir / self.creatives[0]["filename"]).unlink()
        written = webapp.save_to_image_library(self.job_dir, self.creatives, self.record, backdrop=self.backdrop)
        self.assertEqual(len(written), 2)   # the backdrop and the one render that survived

    def test_an_unwritable_library_never_breaks_a_run(self):
        webapp.IMAGE_LIBRARY_DIR = Path("/proc/nope/image_library")
        self.assertEqual(
            webapp.save_to_image_library(self.job_dir, self.creatives, self.record, backdrop=self.backdrop), []
        )


if __name__ == "__main__":
    unittest.main()
