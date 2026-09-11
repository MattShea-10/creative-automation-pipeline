"""What localize_message asks the translator for, and what it does when
the translator misbehaves. The rest of the suite mocks localize_message
out, so nothing else covers this function itself.
"""

import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


class LocalizeMessageTest(unittest.TestCase):
    def setUp(self):
        FakeTranslator.calls = []

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


if __name__ == "__main__":
    unittest.main()
