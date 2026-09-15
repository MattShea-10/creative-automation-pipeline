"""The two translation providers.

Localization used to be one function wrapping one free, unauthenticated
endpoint. It failed in the least useful way available: a test run spent
the day's invisible quota, and hours later a Spanish campaign came out in
English blaming the network. These cover the pieces that made that
possible -- an error page arriving as an ordinary string, a rate limit
indistinguishable from an outage, a quota nobody can query.
"""

import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.translation_providers import (
    DeepLProvider,
    GoogleFreeProvider,
    LingvaProvider,
    MyMemoryProvider,
    TranslationError,
    active_providers,
)


class _Response:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body if body is not None else {"translations": [{"text": "Siente lo fresco."}]}

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class DeepLKeyTest(unittest.TestCase):
    def test_a_free_key_goes_to_the_free_host(self):
        """A free key sent to the paid host answers 403, which reads
        exactly like a bad key. The ":fx" suffix says which is which, so
        nobody should have to know this."""
        self.assertIn("api-free.deepl.com", DeepLProvider("0000-1111:fx").endpoint)

    def test_a_paid_key_goes_to_the_paid_host(self):
        endpoint = DeepLProvider("0000-1111").endpoint
        self.assertIn("api.deepl.com", endpoint)
        self.assertNotIn("api-free", endpoint)

    def test_no_key_is_not_configured(self):
        self.assertFalse(DeepLProvider("").is_configured())
        self.assertTrue(DeepLProvider("abc").is_configured())

    def test_translating_without_a_key_says_so_rather_than_calling_out(self):
        with self.assertRaises(TranslationError) as caught:
            DeepLProvider("").translate("Shop now", "es")
        self.assertEqual(caught.exception.reason, "no key")


class DeepLLanguageTest(unittest.TestCase):
    def test_the_app_codes_map_to_deepl_targets(self):
        provider = DeepLProvider("k")
        self.assertEqual(provider.target_for("es"), "ES")
        self.assertEqual(provider.target_for("fr"), "FR")
        self.assertEqual(provider.target_for("pt"), "PT-BR")
        self.assertEqual(provider.target_for("zh-CN"), "ZH-HANS")

    def test_a_language_deepl_lacks_is_a_reason_not_a_crash(self):
        """Hindi is in the app's region table and not in DeepL's. The
        caller falls through to the free endpoint on this reason, which
        is how Hindi keeps working."""
        with self.assertRaises(TranslationError) as caught:
            DeepLProvider("k").target_for("hi")
        self.assertEqual(caught.exception.reason, "unsupported language")


class DeepLResponseTest(unittest.TestCase):
    def setUp(self):
        self.provider = DeepLProvider("key-1234:fx")

    def _translate(self, response):
        with mock.patch("requests.post", return_value=response) as post:
            self.post = post
            return self.provider.translate("Feel the Fresh.", "es")

    def test_a_translation_comes_back(self):
        self.assertEqual(self._translate(_Response()), "Siente lo fresco.")

    def test_the_key_is_sent_as_a_header_not_in_the_url(self):
        self._translate(_Response())
        _, kwargs = self.post.call_args
        self.assertIn("DeepL-Auth-Key", kwargs["headers"]["Authorization"])
        self.assertEqual(kwargs["json"]["target_lang"], "ES")

    def test_a_spent_month_is_quota_not_a_general_failure(self):
        """Worth its own reason: a rate limit is worth waiting out and a
        spent quota is not, and telling someone to wait when the month is
        gone wastes their evening."""
        with self.assertRaises(TranslationError) as caught:
            self._translate(_Response(456))
        self.assertEqual(caught.exception.reason, "quota")

    def test_being_told_to_slow_down_is_a_rate_limit(self):
        with self.assertRaises(TranslationError) as caught:
            self._translate(_Response(429))
        self.assertEqual(caught.exception.reason, "rate limit")

    def test_a_refused_key_says_so(self):
        for status in (401, 403):
            with self.subTest(status=status), self.assertRaises(TranslationError) as caught:
                self._translate(_Response(status))
            self.assertEqual(caught.exception.reason, "no key")

    def test_an_unreadable_reply_is_a_failure_not_a_translation(self):
        with self.assertRaises(TranslationError):
            self._translate(_Response(200, {"unexpected": True}))

    def test_an_empty_translation_is_a_failure(self):
        with self.assertRaises(TranslationError):
            self._translate(_Response(200, {"translations": [{"text": "   "}]}))

    def test_a_connection_error_is_reported_as_unavailable(self):
        with mock.patch("requests.post", side_effect=OSError("no route")):
            with self.assertRaises(TranslationError) as caught:
                self.provider.translate("Feel the Fresh.", "es")
        self.assertEqual(caught.exception.reason, "unavailable")


class DeepLUsageTest(unittest.TestCase):
    def test_usage_reads_the_months_allowance(self):
        """The point of a key: a number you can look at before a demo."""
        body = {"character_count": 1200, "character_limit": 500000}
        with mock.patch("requests.get", return_value=_Response(200, body)) as get:
            self.assertEqual(DeepLProvider("k:fx").usage(), body)
        self.assertIn("/v2/usage", get.call_args[0][0])

    def test_usage_failing_is_not_an_error(self):
        """A key that translates but whose usage call fails should read
        as working, not as broken."""
        with mock.patch("requests.get", side_effect=OSError("down")):
            self.assertIsNone(DeepLProvider("k:fx").usage())

    def test_no_key_means_no_usage_call(self):
        with mock.patch("requests.get") as get:
            self.assertIsNone(DeepLProvider("").usage())
        get.assert_not_called()


def _with_google(translate):
    class _T:
        def __init__(self, source=None, target=None):
            self.target = target

        def translate(self, text):
            return translate(text)

    return mock.patch.dict("sys.modules", {"deep_translator": mock.MagicMock(GoogleTranslator=_T)})


class GoogleFreeTest(unittest.TestCase):
    ERROR_PAGE = (
        "Error 500 (Server Error)!!1500.That’s an error.There was an error. "
        "Please try again later.That’s all we know."
    )

    def setUp(self):
        GoogleFreeProvider._last_call_at = 0.0
        self.pauses = []
        patch = mock.patch("src.translation_providers._pause", self.pauses.append)
        patch.start()
        self.addCleanup(patch.stop)
        self.stderr = mock.patch.object(sys, "stderr", mock.MagicMock())
        self.stderr.start()
        self.addCleanup(self.stderr.stop)

    def test_a_translation_comes_back(self):
        with _with_google(lambda text: "Siente lo fresco."):
            self.assertEqual(GoogleFreeProvider().translate("Feel the Fresh.", "es"), "Siente lo fresco.")

    def test_an_error_page_is_not_a_translation(self):
        """It arrives as an ordinary string, so a failed call read as a
        successful one -- and got cached, so every later run reused it
        without calling out at all."""
        with _with_google(lambda text: self.ERROR_PAGE):
            with self.assertRaises(TranslationError) as caught:
                GoogleFreeProvider().translate("Feel the Fresh.", "es")
        self.assertEqual(caught.exception.reason, "error page")

    def test_a_translation_mentioning_an_error_is_still_a_translation(self):
        self.assertFalse(GoogleFreeProvider.looks_like_an_error_page("Une erreur de serveur est survenue"))
        self.assertTrue(GoogleFreeProvider.looks_like_an_error_page(self.ERROR_PAGE))

    def test_a_rate_limit_is_waited_out(self):
        attempts = []

        class TooManyRequests(Exception):
            pass

        def translate(text):
            attempts.append(text)
            if len(attempts) < 3:
                raise TooManyRequests("Server Error: You made too many requests")
            return "Siente lo fresco."

        with _with_google(translate):
            self.assertEqual(GoogleFreeProvider().translate("Feel the Fresh.", "es"), "Siente lo fresco.")
        self.assertEqual(len(attempts), 3)
        self.assertEqual([p for p in self.pauses if p >= 1.0], [1.0, 2.0])

    def test_a_rate_limit_that_never_clears_gives_up_with_that_reason(self):
        class TooManyRequests(Exception):
            pass

        with _with_google(lambda text: (_ for _ in ()).throw(TooManyRequests("too many requests"))):
            with self.assertRaises(TranslationError) as caught:
                GoogleFreeProvider().translate("Feel the Fresh.", "es")
        self.assertEqual(caught.exception.reason, "rate limit")

    def test_a_dead_endpoint_is_not_retried_as_though_it_were_busy(self):
        attempts = []

        def translate(text):
            attempts.append(text)
            raise RuntimeError("no network")

        with _with_google(translate):
            with self.assertRaises(TranslationError) as caught:
                GoogleFreeProvider().translate("Feel the Fresh.", "es")
        self.assertEqual(len(attempts), 1)
        self.assertEqual(caught.exception.reason, "unavailable")

    def test_calls_are_spaced_apart(self):
        """The form's copy fields go out in a loop against an endpoint
        that counts five a second. The run was rate-limiting itself."""
        with _with_google(lambda text: "ok"):
            provider = GoogleFreeProvider()
            provider.translate("one", "es")
            GoogleFreeProvider._last_call_at = time.monotonic()
            provider.translate("two", "es")
        self.assertTrue(any(0 < pause <= GoogleFreeProvider.min_seconds_between_calls for pause in self.pauses))


class MyMemoryTest(unittest.TestCase):
    """The second keyless provider, and the traps it shares with the first."""

    def setUp(self):
        MyMemoryProvider._last_call_at = 0.0
        patch = mock.patch("src.translation_providers._pause", lambda seconds: None)
        patch.start()
        self.addCleanup(patch.stop)
        self.provider = MyMemoryProvider()

    def _get(self, body, status=200):
        return mock.patch("requests.get", return_value=_Response(status, body))

    def test_a_translation_comes_back(self):
        with self._get({"responseData": {"translatedText": "Siente lo fresco."}}):
            self.assertEqual(self.provider.translate("Feel the Fresh.", "es"), "Siente lo fresco.")

    def test_a_spent_allowance_arrives_as_a_200_and_is_caught(self):
        """Same trap as Google's error page: the refusal is delivered in
        the translation field, so it reads as a successful translation."""
        body = {"responseData": {"translatedText":
                "YOU USED ALL AVAILABLE FREE TRANSLATIONS FOR TODAY. NEXT AVAILABLE IN 12 HOURS"}}
        with self._get(body), self.assertRaises(TranslationError) as caught:
            self.provider.translate("Feel the Fresh.", "es")
        self.assertEqual(caught.exception.reason, "quota")

    def test_a_warning_in_the_translation_field_is_not_a_translation(self):
        body = {"responseData": {"translatedText": "MYMEMORY WARNING: YOU HAVE USED ALL..."}}
        with self._get(body), self.assertRaises(TranslationError):
            self.provider.translate("Feel the Fresh.", "es")

    def test_the_source_returned_unchanged_is_not_a_translation(self):
        """Cached as one, it would stand in front of the real thing on
        every later run -- which is how a French campaign ended up with
        English in its type layers."""
        body = {"responseData": {"translatedText": "Feel the Fresh."}}
        with self._get(body), self.assertRaises(TranslationError):
            self.provider.translate("Feel the Fresh.", "es")

    def test_portuguese_asks_for_the_regional_pairing(self):
        with self._get({"responseData": {"translatedText": "Sinta o frescor."}}) as get:
            self.provider.translate("Feel the Fresh.", "pt")
        self.assertEqual(get.call_args.kwargs["params"]["langpair"], "en|pt-BR")


class LingvaTest(unittest.TestCase):
    """The third keyless provider. Public instances come and go, so it
    must not trust any single volunteer server."""

    def setUp(self):
        LingvaProvider._last_call_at = 0.0
        patch = mock.patch("src.translation_providers._pause", lambda seconds: None)
        patch.start()
        self.addCleanup(patch.stop)
        self.provider = LingvaProvider()

    def test_a_translation_comes_back(self):
        with mock.patch("requests.get", return_value=_Response(200, {"translation": "Siente lo fresco."})):
            self.assertEqual(self.provider.translate("Feel the Fresh.", "es"), "Siente lo fresco.")

    def test_a_dead_instance_is_stepped_over(self):
        answers = [_Response(502, {}), _Response(200, {"translation": "Siente lo fresco."})]
        with mock.patch("requests.get", side_effect=answers):
            self.assertEqual(self.provider.translate("Feel the Fresh.", "es"), "Siente lo fresco.")

    def test_every_instance_failing_raises(self):
        with mock.patch("requests.get", side_effect=OSError("down")):
            with self.assertRaises(TranslationError):
                self.provider.translate("Feel the Fresh.", "es")

    def test_it_tries_more_than_one_host(self):
        self.assertGreater(len(LingvaProvider.HOSTS), 1)


class KeylessChainTest(unittest.TestCase):
    def test_the_feature_works_with_nothing_configured(self):
        """The point of the whole arrangement: choose Spanish, get
        Spanish, configure nothing. One free endpoint with an invisible
        daily quota was never enough to promise that."""
        with mock.patch.dict("os.environ", {"DEEPL_API_KEY": ""}):
            names = [p.name for p in active_providers()]
        self.assertEqual(names, ["google-free", "mymemory", "lingva"])

    def test_a_key_only_adds_a_preference(self):
        with mock.patch.dict("os.environ", {"DEEPL_API_KEY": "key-1234:fx"}):
            names = [p.name for p in active_providers()]
        self.assertEqual(names[0], "deepl")
        self.assertEqual(names[1:], ["google-free", "mymemory", "lingva"])


if __name__ == "__main__":
    unittest.main()
