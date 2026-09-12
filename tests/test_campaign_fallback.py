"""A blank Campaign field must not open a second set of templates.

Editing default_templates/<product>/ while restoring
default_templates/<campaign>/<product>/ looks like the restore doing
nothing, over and over. The folder is decided in one place now, and it
asks the briefs when the field is empty; the card fills the field in so
it stops being empty in the first place.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import webapp


class CampaignFallbackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._orig_templates = webapp.DEFAULT_TEMPLATES_DIR
        self._orig_briefs = webapp.BRIEFS_DIR
        webapp.DEFAULT_TEMPLATES_DIR = self.tmp / "default_templates"
        webapp.DEFAULT_TEMPLATES_DIR.mkdir(parents=True)
        webapp.BRIEFS_DIR = self.tmp / "briefs"
        webapp.BRIEFS_DIR.mkdir()
        (webapp.BRIEFS_DIR / "sample.json").write_text("{}")
        webapp._brief_campaign_cache.clear()
        self.choices = [
            {"product_name": "HydroBoost Sports Drink", "campaign": "Winter Glow 2026"},
            {"product_name": "FreshGlow Body Wash", "campaign": "Winter Glow 2026"},
        ]

    def tearDown(self):
        webapp.DEFAULT_TEMPLATES_DIR = self._orig_templates
        webapp.BRIEFS_DIR = self._orig_briefs
        webapp._brief_campaign_cache.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_blank_campaign_uses_the_briefs_one(self):
        with mock.patch.object(webapp, "_brief_choices", return_value=self.choices):
            parts = webapp._campaign_folder_parts("HydroBoost Sports Drink", "")
        self.assertEqual(parts, ("Winter Glow 2026", "HydroBoost Sports Drink"))

    def test_the_folder_and_the_memory_key_agree(self):
        with mock.patch.object(webapp, "_brief_choices", return_value=self.choices):
            folder = webapp.product_templates_dir("HydroBoost Sports Drink")
            key = webapp._product_memory_key("HydroBoost Sports Drink")
        self.assertEqual(folder, webapp.DEFAULT_TEMPLATES_DIR / "Winter Glow 2026" / "HydroBoost Sports Drink")
        self.assertEqual(key, "Winter Glow 2026/HydroBoost Sports Drink")

    def test_a_typed_campaign_still_wins(self):
        with mock.patch.object(webapp, "_brief_choices", return_value=self.choices):
            parts = webapp._campaign_folder_parts("HydroBoost Sports Drink", "Summer Refresh 2026")
        self.assertEqual(parts, ("Summer Refresh 2026", "HydroBoost Sports Drink"))

    def test_a_product_no_brief_knows_keeps_its_own_folder(self):
        with mock.patch.object(webapp, "_brief_choices", return_value=self.choices):
            parts = webapp._campaign_folder_parts("Something Invented", "")
        self.assertEqual(parts, ("Something Invented",))

    def test_no_product_is_still_the_shared_folder(self):
        with mock.patch.object(webapp, "_brief_choices", return_value=self.choices):
            self.assertEqual(webapp._campaign_folder_parts("", ""), ())

    def test_the_lookup_is_cached_not_reparsed_every_call(self):
        with mock.patch.object(webapp, "_brief_choices", return_value=self.choices) as choices:
            for _ in range(5):
                webapp.campaign_for_product("HydroBoost Sports Drink")
        self.assertEqual(choices.call_count, 1)


class CardPrefillTest(unittest.TestCase):
    """The card has to open with Campaign filled. A memory saved before the
    field was required holds "", and that empty value used to overwrite what
    the brief supplied -- leaving a blank field pointing at a second set of
    templates."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._orig_templates = webapp.DEFAULT_TEMPLATES_DIR
        self._orig_jobs = webapp.JOBS_DIR
        webapp.DEFAULT_TEMPLATES_DIR = self.tmp / "default_templates"
        webapp.DEFAULT_TEMPLATES_DIR.mkdir(parents=True)
        webapp.JOBS_DIR = self.tmp / "jobs"
        webapp.JOBS_DIR.mkdir()
        webapp._brief_campaign_cache.clear()
        self.choices = [{"product_name": "HydroBoost Sports Drink", "campaign": "Winter Glow 2026"}]

    def tearDown(self):
        webapp.DEFAULT_TEMPLATES_DIR = self._orig_templates
        webapp.JOBS_DIR = self._orig_jobs
        webapp._brief_campaign_cache.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _card_for(self, cards, product):
        return next(c for c in cards if c["prefill"].get("product_name") == product)

    def test_an_empty_remembered_campaign_does_not_blank_the_card(self):
        webapp._save_preferences({
            "products": {
                "Winter Glow 2026/HydroBoost Sports Drink": {
                    "product_name": "HydroBoost Sports Drink",
                    "campaign_name": "",           # saved before the field was required
                    "campaign_message": "Glow Through Winter.",
                }
            }
        })
        with mock.patch.object(webapp, "_brief_choices", return_value=self.choices):
            cards = webapp._remembered_campaign_cards()
        card = self._card_for(cards, "HydroBoost Sports Drink")
        self.assertEqual(card["prefill"]["campaign_name"], "Winter Glow 2026")
        self.assertEqual(card["prefill"]["campaign_message"], "Glow Through Winter.",
                         "other remembered fields still apply")

    def test_a_remembered_campaign_still_wins(self):
        webapp._save_preferences({
            "products": {
                "Winter Glow 2026/HydroBoost Sports Drink": {
                    "product_name": "HydroBoost Sports Drink",
                    "campaign_name": "Summer Refresh 2026",
                }
            }
        })
        with mock.patch.object(webapp, "_brief_choices", return_value=self.choices):
            cards = webapp._remembered_campaign_cards()
        card = self._card_for(cards, "HydroBoost Sports Drink")
        self.assertEqual(card["prefill"]["campaign_name"], "Summer Refresh 2026")

    def test_a_product_only_memory_gets_its_campaign_filled(self):
        webapp._save_preferences({
            "products": {"HydroBoost Sports Drink": {"product_name": "HydroBoost Sports Drink"}}
        })
        with mock.patch.object(webapp, "_brief_choices", return_value=[]):
            with mock.patch.object(webapp, "campaign_for_product", return_value="Winter Glow 2026"):
                cards = webapp._remembered_campaign_cards()
        card = self._card_for(cards, "HydroBoost Sports Drink")
        self.assertEqual(card["prefill"]["campaign_name"], "Winter Glow 2026")


if __name__ == "__main__":
    unittest.main()


class NewCardDefaultsTest(unittest.TestCase):
    """A card created by "Create Campaign" must not open with Campaign
    blank -- that is where a stray campaign-less template folder comes
    from, and it is invisible until a restore appears to do nothing."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._orig_templates = webapp.DEFAULT_TEMPLATES_DIR
        self._orig_jobs = webapp.JOBS_DIR
        webapp.DEFAULT_TEMPLATES_DIR = self.tmp / "default_templates"
        webapp.DEFAULT_TEMPLATES_DIR.mkdir(parents=True)
        webapp.JOBS_DIR = self.tmp / "jobs"
        webapp.JOBS_DIR.mkdir()
        webapp._brief_campaign_cache.clear()
        self.choices = [
            {"product_name": "HydroBoost Sports Drink", "campaign": "Winter Glow 2026"},
            {"product_name": "Sunburst Lemonade", "campaign": "Summer Refresh 2026"},
        ]

    def tearDown(self):
        webapp.DEFAULT_TEMPLATES_DIR = self._orig_templates
        webapp.JOBS_DIR = self._orig_jobs
        webapp._brief_campaign_cache.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_new_card_opens_on_the_campaign_last_used(self):
        webapp._save_preferences({"campaign_name": "Winter Glow 2026"})
        with mock.patch.object(webapp, "_brief_choices", return_value=self.choices):
            self.assertEqual(webapp.default_campaign_name(), "Winter Glow 2026")

    def test_with_nothing_remembered_it_takes_the_first_brief(self):
        webapp._save_preferences({})
        with mock.patch.object(webapp, "_brief_choices", return_value=self.choices):
            self.assertEqual(webapp.default_campaign_name(), "Winter Glow 2026")

    def test_no_briefs_and_nothing_remembered_is_empty_not_a_guess(self):
        webapp._save_preferences({})
        with mock.patch.object(webapp, "_brief_choices", return_value=[]):
            self.assertEqual(webapp.default_campaign_name(), "")

    def test_the_page_gets_a_product_to_campaign_map(self):
        with mock.patch.object(webapp, "_brief_choices", return_value=self.choices):
            table = webapp.brief_campaign_by_product()
        self.assertEqual(table["hydroboost sports drink"], "Winter Glow 2026")
        self.assertEqual(table["sunburst lemonade"], "Summer Refresh 2026")
        self.assertNotIn("", table, "a brief with no campaign contributes nothing")
