"""Choosing a language, choosing who translates it, and what happens when
nobody can.

The translating itself is covered in test_translation_providers.py. This
is the layer above: which provider gets asked, what it falls back to, and
the fact that a run whose copy could not be translated draws English and
says so rather than shipping the wrong language quietly.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.localization as localization
from src.localization import infer_language, localize_message
from src.translation_providers import TranslationError


def _allow_the_network(test):
    """Let this test reach the (mocked) providers.

    localize_message refuses to call out from a test process -- the suite
    drives the real pipeline, and a full run was firing hundreds of
    requests at Google's free endpoint until the day's quota was gone.
    Tests that exercise the call itself have to say so.
    """
    patch = mock.patch.object(localization, "_network_is_off_limits", return_value=False)
    patch.start()
    test.addCleanup(patch.stop)


class _Provider:
    """A stand-in for a translation provider: answers, or raises with a
    reason, and records that it was asked."""

    def __init__(self, name, answer=None, error=None):
        self.name = name
        self.answer = answer
        self.error = error
        self.asked = []

    def translate(self, text, language):
        self.asked.append((text, language))
        if self.error:
            raise self.error
        return self.answer


def _using(*providers):
    return mock.patch.object(localization, "active_providers", return_value=list(providers))


class InferLanguageTest(unittest.TestCase):
    def test_region_table_and_explicit_override(self):
        self.assertEqual(infer_language("France"), "fr")
        self.assertEqual(infer_language(" france "), "fr")
        self.assertEqual(infer_language("Narnia"), "en")
        self.assertEqual(infer_language("France", "es"), "es")


class EnglishAndOfflineTest(unittest.TestCase):
    def setUp(self):
        localization._last_failure = ""
        localization._last_provider = ""
        self.addCleanup(setattr, localization, "_last_failure", "")
        self.addCleanup(setattr, localization, "_last_provider", "")

    def test_english_never_asks_anyone(self):
        provider = _Provider("deepl", "should not happen")
        with _using(provider):
            self.assertEqual(localize_message("Shop now", "en"), ("Shop now", False))
        self.assertEqual(provider.asked, [])

    def test_a_test_process_never_reaches_a_translator(self):
        """Nothing in the suite asserts a translation -- the pipeline
        tests just run copy through the real path. Hundreds of live calls
        per run went out unnoticed until the quota ran out."""
        provider = _Provider("deepl", "should not happen")
        with _using(provider):
            text, ok = localize_message("Shop now", "es")
        self.assertEqual((text, ok), ("Shop now", False))
        self.assertEqual(provider.asked, [])
        self.assertEqual(localization.take_last_failure(), "offline")

    def test_the_packaged_app_is_never_held_back(self):
        with mock.patch.object(sys, "frozen", True, create=True):
            self.assertFalse(localization._network_is_off_limits())

    def test_the_environment_variable_holds_outside_a_test_too(self):
        with mock.patch.dict("sys.modules"), mock.patch.dict("os.environ"):
            sys.modules.pop("unittest", None)
            os.environ.pop("CREATIVE_PIPELINE_OFFLINE", None)
            self.assertFalse(localization._network_is_off_limits())
            os.environ["CREATIVE_PIPELINE_OFFLINE"] = "1"
            self.assertTrue(localization._network_is_off_limits())


class WhoAnswersTest(unittest.TestCase):
    def setUp(self):
        _allow_the_network(self)
        localization._last_failure = ""
        localization._last_provider = ""
        self.addCleanup(setattr, localization, "_last_failure", "")
        self.addCleanup(setattr, localization, "_last_provider", "")

    def test_the_first_provider_that_answers_wins(self):
        first = _Provider("deepl", "Siente lo fresco.")
        second = _Provider("google-free", "should not be needed")
        with _using(first, second):
            self.assertEqual(localize_message("Feel the Fresh.", "es"), ("Siente lo fresco.", True))
        self.assertEqual(second.asked, [], "the fallback was asked despite the first one answering")
        self.assertEqual(localization.take_last_provider(), "deepl")

    def test_a_spent_deepl_quota_falls_through_to_the_free_endpoint(self):
        """The whole reason for keeping two: a key that runs out mid-month
        should cost quality, not the run."""
        first = _Provider("deepl", error=TranslationError("used up", "quota"))
        second = _Provider("google-free", "Siente lo fresco.")
        with _using(first, second):
            text, ok = localize_message("Feel the Fresh.", "es")
        self.assertEqual((text, ok), ("Siente lo fresco.", True))
        self.assertEqual(localization.take_last_provider(), "google-free")
        self.assertEqual(localization.take_last_failure(), "", "it succeeded in the end")

    def test_a_language_deepl_lacks_falls_through_too(self):
        first = _Provider("deepl", error=TranslationError("no Hindi", "unsupported language"))
        second = _Provider("google-free", "अभी खरीदें")
        with _using(first, second):
            _, ok = localize_message("Shop now", "hi")
        self.assertTrue(ok)
        self.assertEqual(localization.take_last_provider(), "google-free")

    def test_when_everyone_fails_the_english_comes_back_with_the_last_reason(self):
        first = _Provider("deepl", error=TranslationError("used up", "quota"))
        second = _Provider("google-free", error=TranslationError("slow down", "rate limit"))
        with _using(first, second):
            text, ok = localize_message("Feel the Fresh.", "es")
        self.assertEqual((text, ok), ("Feel the Fresh.", False))
        self.assertEqual(localization.take_last_failure(), "rate limit")
        self.assertEqual(localization.take_last_provider(), "")

    def test_a_provider_that_raises_something_unexpected_does_not_end_the_run(self):
        """There is another one behind it. A provider's bug should cost
        its turn, not the campaign."""
        first = _Provider("deepl", error=ValueError("something nobody predicted"))
        second = _Provider("google-free", "Siente lo fresco.")
        with _using(first, second), mock.patch.object(sys, "stderr", mock.MagicMock()):
            text, ok = localize_message("Feel the Fresh.", "es")
        self.assertEqual((text, ok), ("Siente lo fresco.", True))


class ReasonIsNotLeftLyingAroundTest(unittest.TestCase):
    def setUp(self):
        _allow_the_network(self)
        localization._last_failure = ""
        localization._last_provider = ""
        self.addCleanup(setattr, localization, "_last_failure", "")
        self.addCleanup(setattr, localization, "_last_provider", "")

    def test_the_reason_is_cleared_as_it_is_read(self):
        """Left behind, a reason from one run gets reported against the
        next -- a worse lie than no reason at all."""
        with _using(_Provider("google-free", error=TranslationError("down", "unavailable"))):
            localize_message("Feel the Fresh.", "es")
        self.assertEqual(localization.take_last_failure(), "unavailable")
        self.assertEqual(localization.take_last_failure(), "")

    def test_a_success_clears_an_earlier_failure(self):
        with _using(_Provider("google-free", error=TranslationError("down", "unavailable"))):
            localize_message("Feel the Fresh.", "es")
        with _using(_Provider("deepl", "Siente lo fresco.")):
            localize_message("Feel the Fresh.", "es")
        self.assertEqual(localization.take_last_failure(), "")


if __name__ == "__main__":
    unittest.main()
