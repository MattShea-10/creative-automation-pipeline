"""A blank Campaign field must not open a second set of templates.

Editing default_templates/<product>/ while restoring
default_templates/<campaign>/<product>/ looks like the restore doing
nothing, over and over. The folder is decided in one place now, and it
asks the briefs when the field is empty; the card fills the field in so
it stops being empty in the first place.
"""

import json
import os
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


class RemoveCampaignTest(unittest.TestCase):
    """Remove deletes one product-in-campaign, recoverably.

    A card is one product inside one campaign, so removing it must not
    take out the other products sharing that campaign -- and the
    templates are the one thing here a person makes by hand, so they move
    to _to_delete/ rather than being erased.
    """

    def setUp(self):
        webapp.app.config["TESTING"] = True
        self.tmp = Path(tempfile.mkdtemp())
        self._orig = (webapp.BASE_DIR, webapp.DEFAULT_TEMPLATES_DIR, webapp.JOBS_DIR, webapp.BRIEFS_DIR)
        webapp.BASE_DIR = self.tmp
        webapp.DEFAULT_TEMPLATES_DIR = self.tmp / "default_templates"
        webapp.JOBS_DIR = self.tmp / "jobs"
        webapp.BRIEFS_DIR = self.tmp / "briefs"
        for d in (webapp.DEFAULT_TEMPLATES_DIR, webapp.JOBS_DIR, webapp.BRIEFS_DIR):
            d.mkdir(parents=True)
        webapp._brief_campaign_cache.clear()
        webapp._save_preferences({})

    def tearDown(self):
        (webapp.BASE_DIR, webapp.DEFAULT_TEMPLATES_DIR,
         webapp.JOBS_DIR, webapp.BRIEFS_DIR) = self._orig
        webapp._brief_campaign_cache.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _campaign_with_two_products(self):
        for product in ("HydroBoost Sports Drink", "FreshGlow Body Wash"):
            folder = webapp.DEFAULT_TEMPLATES_DIR / "Winter Glow 2026" / product
            folder.mkdir(parents=True)
            (folder / "tester-1080x1080.psd").write_bytes(b"8BPS not really")
        (webapp.BRIEFS_DIR / "winter.json").write_text(json.dumps({"campaign": {
            "name": "Winter Glow 2026", "target_region": "France",
            "target_audience": "adults", "message": "Glow",
            "brand": {"colors": []},
            "products": [{"name": "HydroBoost Sports Drink", "slug": "hydroboost"},
                         {"name": "FreshGlow Body Wash", "slug": "freshglow"}],
        }}))

    def test_removing_one_product_leaves_the_others_alone(self):
        self._campaign_with_two_products()
        r = webapp.app.test_client().post("/delete-campaign", json={
            "product_name": "HydroBoost Sports Drink", "campaign_name": "Winter Glow 2026"})
        self.assertEqual(r.status_code, 200, r.data)

        gone = webapp.DEFAULT_TEMPLATES_DIR / "Winter Glow 2026" / "HydroBoost Sports Drink"
        kept = webapp.DEFAULT_TEMPLATES_DIR / "Winter Glow 2026" / "FreshGlow Body Wash"
        self.assertFalse(gone.exists(), "the removed product's templates are gone from default_templates")
        self.assertTrue(kept.is_dir(), "the campaign's other product must survive")

        # Moved, not erased.
        graves = list((self.tmp / "_to_delete").iterdir())
        self.assertEqual(len(graves), 1, graves)
        self.assertTrue((graves[0] / "tester-1080x1080.psd").is_file(), "the PSD is recoverable")

        brief = json.loads((webapp.BRIEFS_DIR / "winter.json").read_text())
        self.assertEqual([p["name"] for p in brief["campaign"]["products"]],
                         ["FreshGlow Body Wash"])

    def test_removing_the_last_product_takes_the_brief_with_it(self):
        self._campaign_with_two_products()
        client = webapp.app.test_client()
        for product in ("HydroBoost Sports Drink", "FreshGlow Body Wash"):
            client.post("/delete-campaign", json={
                "product_name": product, "campaign_name": "Winter Glow 2026"})
        self.assertFalse((webapp.BRIEFS_DIR / "winter.json").exists(),
                         "a brief describing nothing should not stay behind")
        self.assertFalse((webapp.DEFAULT_TEMPLATES_DIR / "Winter Glow 2026").exists(),
                         "the emptied campaign folder is clutter, not data")

    def test_removing_a_product_clears_its_session_slots(self):
        """A removed card must not come back as a second, identical one.

        The session index only ever gained slots: generate() files a job
        under whatever campaign_slot its card posted, and removing a card
        renumbered the page without touching what was recorded. A real
        session ended up with slots 1, 2 and 4 where 2 and 4 were the
        same product in the same campaign, rendered as two cards nobody
        could tell apart. Two slots for one product in one session are
        legitimate -- two cards generated side by side -- so the slot is
        cleared when the product is deleted, not when the page renders.
        """
        self._campaign_with_two_products()
        sessions = webapp.JOBS_DIR / "_sessions"
        sessions.mkdir(parents=True, exist_ok=True)
        for job_id, product in (("j1", "HydroBoost Sports Drink"),
                                ("j2", "FreshGlow Body Wash"),
                                ("j3", "HydroBoost Sports Drink")):
            d = webapp.JOBS_DIR / job_id
            d.mkdir(parents=True, exist_ok=True)
            (d / "form_state.json").write_text(json.dumps({"fields": {
                "product_name": product, "campaign_name": "Winter Glow 2026"}}))
        (sessions / "s.json").write_text(json.dumps({"slots": {"1": "j1", "2": "j2", "4": "j3"}}))

        webapp.app.test_client().post("/delete-campaign", json={
            "product_name": "HydroBoost Sports Drink", "campaign_name": "Winter Glow 2026"})

        slots = json.loads((sessions / "s.json").read_text())["slots"]
        self.assertEqual(slots, {"2": "j2"}, "both HydroBoost slots should be gone, FreshGlow kept")

    def test_the_remembered_form_goes_too_or_the_card_comes_back(self):
        self._campaign_with_two_products()
        webapp._save_preferences({"products": {
            "Winter Glow 2026/HydroBoost Sports Drink": {"product_name": "HydroBoost Sports Drink"},
            "Winter Glow 2026/FreshGlow Body Wash": {"product_name": "FreshGlow Body Wash"},
        }})
        webapp.app.test_client().post("/delete-campaign", json={
            "product_name": "HydroBoost Sports Drink", "campaign_name": "Winter Glow 2026"})
        remembered = webapp._product_memories(webapp._load_preferences())
        self.assertNotIn("Winter Glow 2026/HydroBoost Sports Drink", remembered)
        self.assertIn("Winter Glow 2026/FreshGlow Body Wash", remembered)


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

    def test_the_campaign_is_actually_saved_with_the_batch(self):
        """Every brief field a run posts has to survive into form_state.json.

        campaign_name was in CAMPAIGN_BRIEF_FIELD_NAMES but not in
        EDIT_TEXT_FIELD_NAMES, which is the list that decides what a job
        writes. So a batch named its output files and its template folder
        after the campaign and then forgot which one it was, and every
        reopen had to guess the campaign back from the product name."""
        for name in webapp.CAMPAIGN_BRIEF_FIELD_NAMES:
            self.assertIn(
                name, webapp.EDIT_TEXT_FIELD_NAMES,
                f"{name} is remembered on the form but never saved with the job",
            )

    def test_the_market_breaks_a_tie_between_two_campaigns(self):
        """A product two briefs claim has no answer from its name alone.

        The batch also saved the market it ran for, and only one of the
        two briefs targets it -- so reopening a Mexico batch of HydroBoost
        gets Summer Refresh, not whichever brief file sorted first."""
        both = [
            {"product_name": "HydroBoost Sports Drink", "campaign": "Winter Glow 2026", "market": "France"},
            {"product_name": "HydroBoost Sports Drink", "campaign": "Summer Refresh 2026", "market": "Mexico"},
        ]
        with mock.patch.object(webapp, "_brief_choices", return_value=both):
            webapp._brief_campaign_cache.clear()
            self.assertEqual(
                webapp.campaign_for_product("HydroBoost Sports Drink", "Mexico"),
                "Summer Refresh 2026",
            )
            webapp._brief_campaign_cache.clear()
            self.assertEqual(
                webapp.campaign_for_product("HydroBoost Sports Drink", "France"),
                "Winter Glow 2026",
            )
            # No market, no answer -- better than the wrong one.
            webapp._brief_campaign_cache.clear()
            self.assertEqual(webapp.campaign_for_product("HydroBoost Sports Drink"), "")
        webapp._brief_campaign_cache.clear()

    def test_a_product_two_briefs_claim_is_not_guessed_at(self):
        """HydroBoost Sports Drink is in Winter Glow 2026 AND Summer
        Refresh 2026, and each has its own template folder.

        Filling in whichever brief file sorts first is worse than filling
        nothing: the campaign picks the folder, so a wrong guess edits one
        campaign's templates while the person believes they are editing
        the other's. It is left out of the autofill map and offered as a
        list instead."""
        both = self.choices + [dict(self.choices[0], campaign="Summer Refresh 2026")]
        with mock.patch.object(webapp, "_brief_choices", return_value=both):
            self.assertNotIn("hydroboost sports drink", webapp.brief_campaign_by_product())
            self.assertEqual(
                webapp.brief_campaigns_by_product()["hydroboost sports drink"],
                ["Winter Glow 2026", "Summer Refresh 2026"],
            )
        # And the server-side folder fallback must not guess either.
        webapp._brief_campaign_cache.clear()
        with mock.patch.object(webapp, "_brief_choices", return_value=both):
            self.assertEqual(webapp.campaign_for_product("HydroBoost Sports Drink"), "")
        webapp._brief_campaign_cache.clear()

    def test_a_product_only_one_brief_claims_is_still_filled_in(self):
        with mock.patch.object(webapp, "_brief_choices", return_value=self.choices):
            self.assertEqual(
                webapp.brief_campaign_by_product()["hydroboost sports drink"],
                "Winter Glow 2026",
            )

    def test_the_page_gets_a_product_to_campaign_map(self):
        with mock.patch.object(webapp, "_brief_choices", return_value=self.choices):
            table = webapp.brief_campaign_by_product()
        self.assertEqual(table["hydroboost sports drink"], "Winter Glow 2026")
        self.assertEqual(table["sunburst lemonade"], "Summer Refresh 2026")
        self.assertNotIn("", table, "a brief with no campaign contributes nothing")


if __name__ == "__main__":
    unittest.main()
