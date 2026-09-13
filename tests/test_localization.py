"""What localize_message asks the translator for, and what it does when
the translator misbehaves. The rest of the suite mocks localize_message
out, so nothing else covers this function itself.
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.localization as localization
from src.localization import infer_language, localize_message


class FakeTranslator:
    """Stands in for deep_translator.GoogleTranslator, recording how it
    was constructed so the source language can be asserted."""

    calls = []

    def __init__(self, source=None, target=None):
        FakeTranslator.calls.append((source, target))
        self.target = target

    def translate(self, text):
        return "[%s] %s" % (self.target, text)


def _with_translator(cls):
    """deep_translator is imported inside the function, so the module has
    to be patched where it is looked up."""
    module = mock.MagicMock()
    module.GoogleTranslator = cls
    return mock.patch.dict(sys.modules, {"deep_translator": module})



def _allow_the_network(test):
    """Let this test reach the (mocked) translator.

    localize_message refuses to call out from a test process -- the
    suite drives the real pipeline, and a full run was firing hundreds
    of requests at Google's free endpoint. The tests that exist to
    exercise the call itself have to say so.
    """
    patch = mock.patch.object(localization, "_network_is_off_limits", return_value=False)
    patch.start()
    test.addCleanup(patch.stop)


class LocalizeMessageTest(unittest.TestCase):
    def setUp(self):
        FakeTranslator.calls = []
        _allow_the_network(self)

    def test_lets_google_detect_the_source_language(self):
        # The point of the fix: text reaching here is not always English --
        # a template exported from a Spanish run carries Spanish -- and
        # declaring it English returns it barely changed.
        with _with_translator(FakeTranslator):
            text, ok = localize_message("BEBIDA REFRESCANTE", "fr")
        self.assertEqual((text, ok), ("[fr] BEBIDA REFRESCANTE", True))
        self.assertEqual(FakeTranslator.calls, [("auto", "fr")])

    def test_english_target_never_calls_the_translator(self):
        with _with_translator(FakeTranslator):
            self.assertEqual(localize_message("Shop now", "en"), ("Shop now", False))
        self.assertEqual(FakeTranslator.calls, [])

    def test_a_failure_falls_back_to_english_and_says_why(self):
        class Broken(FakeTranslator):
            def translate(self, text):
                raise RuntimeError("no network")

        stderr = io.StringIO()
        with _with_translator(Broken), redirect_stderr(stderr):
            text, ok = localize_message("Shop now", "fr")
        self.assertEqual((text, ok), ("Shop now", False))
        self.assertIn("RuntimeError", stderr.getvalue())
        self.assertIn("no network", stderr.getvalue())

    def test_an_empty_answer_is_not_a_translation(self):
        class Empty(FakeTranslator):
            def translate(self, text):
                return ""

        with _with_translator(Empty):
            self.assertEqual(localize_message("Shop now", "fr"), ("Shop now", False))


class InferLanguageTest(unittest.TestCase):
    def test_region_table_and_explicit_override(self):
        self.assertEqual(infer_language("France"), "fr")
        self.assertEqual(infer_language(" france "), "fr")
        self.assertEqual(infer_language("Narnia"), "en")
        self.assertEqual(infer_language("France", "es"), "es")


class ErrorPageIsNotATranslationTest(unittest.TestCase):
    """Google's 500 page comes back as a string, not an exception.

    So a failed call read as a successful translation: three campaigns
    shipped with the headline "ERROR 500 (SERVER ERROR)" and the body
    "That's an error. There was an error. Please try again later. That's
    all we know." painted into the creative by the model -- and the same
    text cached against the source phrase, so every later run reused it
    without calling the translator at all.
    """

    ERROR_PAGE = (
        "Error 500 (Server Error)!!1500.That\u2019s an error.There was an error. "
        "Please try again later.That\u2019s all we know."
    )

    def setUp(self):
        _allow_the_network(self)

    def test_an_error_page_is_refused_and_the_source_comes_back(self):
        from unittest import mock

        import src.localization as localization

        class _Translator:
            def __init__(self, *a, **k): pass
            def translate(self, text): return ErrorPageIsNotATranslationTest.ERROR_PAGE

        with mock.patch.dict("sys.modules", {"deep_translator": mock.MagicMock(GoogleTranslator=_Translator)}):
            text, ok = localization.localize_message("Feel the Fresh.", "es")
        self.assertFalse(ok, "an error page must not report itself as a translation")
        self.assertEqual(text, "Feel the Fresh.", "the source text comes back untouched")

    def test_a_real_translation_is_still_accepted(self):
        from unittest import mock

        import src.localization as localization

        class _Translator:
            def __init__(self, *a, **k): pass
            def translate(self, text): return "Siente lo fresco."

        with mock.patch.dict("sys.modules", {"deep_translator": mock.MagicMock(GoogleTranslator=_Translator)}):
            text, ok = localization.localize_message("Feel the Fresh.", "es")
        self.assertTrue(ok)
        self.assertEqual(text, "Siente lo fresco.")

    def test_a_translation_that_mentions_an_error_is_not_an_error_page(self):
        # "error" is a word. A page is several markers together.
        from src.localization import _looks_like_an_error_page

        self.assertFalse(_looks_like_an_error_page("Une erreur de serveur est survenue"))
        self.assertFalse(_looks_like_an_error_page("BOISSON RAFRA\u00ceCHISSANTE"))
        self.assertTrue(_looks_like_an_error_page(self.ERROR_PAGE))
        self.assertTrue(_looks_like_an_error_page("<!doctype html><title>Error</title>"))


class RateLimitTest(unittest.TestCase):
    """Spacing calls out, and waiting a rate limit out.

    A Spanish run came back entirely in English and the results page
    blamed the connection. The connection was fine: the form's eight
    copy fields went out back to back against an endpoint that allows
    about five calls a second, so the run rate-limited itself and then
    misreported why.
    """

    def setUp(self):
        import src.localization as localization

        self.localization = localization
        localization._last_call_at = 0.0
        localization._last_failure = ""
        _allow_the_network(self)
        self.pauses = []
        patch = mock.patch.object(localization, "_pause", self.pauses.append)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(setattr, localization, "_last_failure", "")

    def _translator(self, translate):
        """A stand-in GoogleTranslator whose translate() is the callable
        given -- raising, or returning a string."""
        class _T:
            def __init__(self, *a, **k):
                pass

            def translate(self, text):
                return translate(text)

        return mock.patch.dict("sys.modules", {"deep_translator": mock.MagicMock(GoogleTranslator=_T)})

    class TooManyRequests(Exception):
        """Named to match deep_translator's, which is how it is
        recognised -- the class itself can't be imported here, since the
        library not being installed is one of the cases to survive."""

    def test_back_to_back_calls_are_spaced_apart(self):
        with self._translator(lambda text: "ok"):
            self.localization.localize_message("one", "es")
            self.localization.localize_message("two", "es")
        self.assertTrue(
            any(pause > 0 for pause in self.pauses),
            "the second call went out with no gap after the first",
        )
        self.assertTrue(
            all(pause <= self.localization._MIN_SECONDS_BETWEEN_CALLS for pause in self.pauses),
            "spacing should never wait longer than the interval itself",
        )

    def test_a_rate_limit_is_waited_out_not_given_up_on(self):
        attempts = []

        def translate(text):
            attempts.append(text)
            if len(attempts) < 3:
                raise self.TooManyRequests("Server Error: You made too many requests")
            return "Siente lo fresco."

        with self._translator(translate), redirect_stderr(io.StringIO()):
            text, ok = self.localization.localize_message("Feel the Fresh.", "es")

        self.assertEqual((text, ok), ("Siente lo fresco.", True))
        self.assertEqual(len(attempts), 3)
        backoffs = [pause for pause in self.pauses if pause >= 1.0]
        self.assertEqual(backoffs, [1.0, 2.0], "the wait should rise between tries")
        self.assertEqual(self.localization.take_last_failure(), "", "it succeeded in the end")

    def test_a_rate_limit_that_never_clears_gives_up_and_says_so(self):
        attempts = []

        def translate(text):
            attempts.append(text)
            raise self.TooManyRequests("Server Error: You made too many requests")

        with self._translator(translate), redirect_stderr(io.StringIO()):
            text, ok = self.localization.localize_message("Feel the Fresh.", "es")

        self.assertEqual((text, ok), ("Feel the Fresh.", False))
        self.assertEqual(len(attempts), 1 + len(self.localization._RATE_LIMIT_BACKOFF))
        self.assertEqual(self.localization.take_last_failure(), "rate limit")

    def test_a_dead_endpoint_is_not_retried_as_though_it_were_busy(self):
        """Backing off is for being told "slow down". Nothing else gets
        the treatment -- a missing library would just wait seven seconds
        to fail the same way."""
        attempts = []

        def translate(text):
            attempts.append(text)
            raise RuntimeError("no network")

        with self._translator(translate), redirect_stderr(io.StringIO()):
            self.localization.localize_message("Feel the Fresh.", "es")

        self.assertEqual(len(attempts), 1)
        self.assertEqual(self.localization.take_last_failure(), "unavailable")

    def test_an_error_page_reports_itself_as_an_error_page(self):
        page = ErrorPageIsNotATranslationTest.ERROR_PAGE
        with self._translator(lambda text: page), redirect_stderr(io.StringIO()):
            self.localization.localize_message("Feel the Fresh.", "es")
        self.assertEqual(self.localization.take_last_failure(), "error page")

    def test_the_reason_is_cleared_as_it_is_read(self):
        """Left lying around, a reason from one run gets reported against
        the next one -- which would be a worse lie than no reason."""
        with self._translator(lambda text: (_ for _ in ()).throw(RuntimeError("x"))), \
                redirect_stderr(io.StringIO()):
            self.localization.localize_message("Feel the Fresh.", "es")
        self.assertEqual(self.localization.take_last_failure(), "unavailable")
        self.assertEqual(self.localization.take_last_failure(), "")

    def test_a_success_clears_an_earlier_failure(self):
        with self._translator(lambda text: (_ for _ in ()).throw(RuntimeError("x"))), \
                redirect_stderr(io.StringIO()):
            self.localization.localize_message("Feel the Fresh.", "es")
        with self._translator(lambda text: "Siente lo fresco."):
            self.localization.localize_message("Feel the Fresh.", "es")
        self.assertEqual(self.localization.take_last_failure(), "")


class NoNetworkFromATestRunTest(unittest.TestCase):
    """The suite must not call the translator.

    Nothing here was ever asserting a translation -- the pipeline tests
    just happen to run copy through the real code path. Hundreds of live
    requests per run went out unnoticed until the day's quota ran out
    and real runs started coming back in English.
    """

    def setUp(self):
        localization._last_failure = ""
        self.addCleanup(setattr, localization, "_last_failure", "")

    def test_a_test_process_never_reaches_the_translator(self):
        called = []

        class _T:
            def __init__(self, *a, **k):
                pass

            def translate(self, text):
                called.append(text)
                return "should never happen"

        with mock.patch.dict("sys.modules", {"deep_translator": mock.MagicMock(GoogleTranslator=_T)}):
            text, ok = localization.localize_message("Feel the Fresh.", "es")

        self.assertEqual((text, ok), ("Feel the Fresh.", False))
        self.assertEqual(called, [], "a test run called the live endpoint")
        self.assertEqual(localization.take_last_failure(), "offline")

    def test_the_packaged_app_is_never_held_back(self):
        """sys.frozen is the packaged build. Whatever it drags in, it
        translates for real -- a shipped app silently refusing to
        translate would be far worse than a chatty test suite."""
        with mock.patch.object(sys, "frozen", True, create=True):
            self.assertFalse(localization._network_is_off_limits())

    def test_the_environment_variable_holds_outside_a_test_too(self):
        """For anything driving the pipeline by hand -- a script, the
        CLI in a loop -- without wanting to spend the quota."""
        with mock.patch.dict("sys.modules"), mock.patch.dict("os.environ"):
            # Out of a test process, so only the variable is left to answer.
            sys.modules.pop("unittest", None)
            os.environ.pop("CREATIVE_PIPELINE_OFFLINE", None)
            self.assertFalse(localization._network_is_off_limits())
            os.environ["CREATIVE_PIPELINE_OFFLINE"] = "1"
            self.assertTrue(localization._network_is_off_limits())


if __name__ == "__main__":
    unittest.main()
