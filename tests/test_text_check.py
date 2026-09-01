"""The no-text check on generated backdrops.

Two layers are being tested: find_text() itself (does it see lettering,
and does it stay quiet on a picture without any), and webapp's retry loop
around it (does a dirty result actually get regenerated, and does the
whole thing stay out of the way when OCR isn't installed).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont

import webapp
from src.text_check import TextCheckResult, find_text, ocr_available


def _font(size):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return None


def _image_with_text(text="SUMMER SALE", size=(1200, 600)):
    image = Image.new("RGB", size, (235, 235, 235))
    ImageDraw.Draw(image).text((60, size[1] // 3), text, fill=(10, 10, 10), font=_font(90))
    return image


def _image_without_text(size=(1200, 600)):
    # Not a flat fill -- a flat image is trivially clean and would pass
    # even a broken checker. Gradient plus noise-ish banding is closer to
    # the photographic backdrops this runs on.
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    for y in range(size[1]):
        shade = int(90 + 90 * (y / size[1]))
        draw.line([(0, y), (size[0], y)], fill=(shade // 2, shade, shade // 3))
    for x in range(0, size[0], 37):
        draw.ellipse([x, 100, x + 60, 190], fill=(60, 120, 40))
    return image


@unittest.skipUnless(ocr_available(), "Tesseract isn't installed")
class FindTextTest(unittest.TestCase):
    def test_reads_lettering_baked_into_an_image(self):
        result = find_text(_image_with_text())
        self.assertTrue(result.available)
        self.assertTrue(result.found_text)
        self.assertIn("SUMMER", result.summary())

    def test_stays_quiet_on_an_image_with_no_text(self):
        # The expensive failure mode: a false alarm spends an API call on
        # a needless regeneration and teaches people to ignore warnings.
        result = find_text(_image_without_text())
        self.assertTrue(result.available)
        self.assertFalse(result.found_text, result.summary())

    def test_ignores_lettering_too_small_to_read(self):
        # A few pixels of "text" in a large frame is texture being
        # over-read, not something a viewer would ever see.
        big = Image.new("RGB", (2000, 1200), (235, 235, 235))
        ImageDraw.Draw(big).text((20, 20), "tiny", fill=(0, 0, 0), font=_font(9))
        self.assertFalse(find_text(big).found_text)


class _ScriptedProvider:
    """Returns a canned sequence of images, recording the prompts used."""

    name = "scripted"

    def __init__(self, images):
        self.images = list(images)
        self.prompts = []

    def generate(self, prompt, width=1024, height=1024):
        self.prompts.append(prompt)
        return self.images.pop(0) if self.images else self.images_last

    @property
    def images_last(self):
        return _image_without_text()


@unittest.skipUnless(ocr_available(), "Tesseract isn't installed")
class RetryLoopTest(unittest.TestCase):
    def test_a_clean_first_result_is_not_regenerated(self):
        provider = _ScriptedProvider([_image_without_text()])
        _image, prompt, attempts, result = webapp._generate_text_free(
            provider, "marathon runners", 600, 300
        )
        self.assertEqual(attempts, 1)
        self.assertFalse(result.found_text)
        self.assertEqual(prompt, "marathon runners")
        self.assertEqual(provider.prompts, ["marathon runners"])

    def test_text_in_the_first_result_triggers_a_harder_retry(self):
        provider = _ScriptedProvider([_image_with_text(), _image_without_text()])
        _image, prompt, attempts, result = webapp._generate_text_free(
            provider, "marathon runners", 600, 300
        )
        self.assertEqual(attempts, 2)
        self.assertFalse(result.found_text)
        # The retry escalates rather than re-sending the phrasing that
        # has already demonstrably failed for this prompt.
        self.assertEqual(provider.prompts[0], "marathon runners")
        self.assertIn(webapp.NO_TEXT_ESCALATION, provider.prompts[1])

    def test_it_gives_up_and_reports_rather_than_looping(self):
        # Every attempt dirty. The budget is finite -- each retry is a
        # real API call -- so it stops and hands back what it has along
        # with the finding, for the caller to warn about.
        provider = _ScriptedProvider([_image_with_text() for _ in range(6)])
        _image, _prompt, attempts, result = webapp._generate_text_free(
            provider, "marathon runners", 600, 300
        )
        self.assertEqual(attempts, webapp.AI_TEXT_RETRY_LIMIT + 1)
        self.assertTrue(result.found_text)

    def test_the_offline_placeholder_is_never_checked(self):
        # It draws the prompt across its own gradient by design, so it
        # would fail every time, burn the whole retry budget regenerating
        # something that is text on purpose, and pay for the OCR to learn
        # nothing.
        provider = _ScriptedProvider([_image_with_text(), _image_without_text()])
        provider.name = "mock"
        _image, _prompt, attempts, result = webapp._generate_text_free(
            provider, "marathon runners", 600, 300
        )
        self.assertEqual(attempts, 1)
        self.assertFalse(result.available)


class OcrUnavailableTest(unittest.TestCase):
    def test_a_missing_engine_is_reported_not_treated_as_clean(self):
        # "Couldn't check" and "checked, clean" are different facts and
        # must never collapse into each other -- the caller says so on
        # the results page rather than implying a verification that never
        # happened.
        import src.text_check as text_check

        original = text_check.ocr_available
        text_check.ocr_available = lambda: False
        try:
            result = text_check.find_text(_image_with_text())
        finally:
            text_check.ocr_available = original
        self.assertFalse(result.available)
        self.assertFalse(result.found_text)

    def test_no_retries_are_spent_when_nothing_can_be_checked(self):
        import src.text_check as text_check

        original = text_check.ocr_available
        text_check.ocr_available = lambda: False
        try:
            provider = _ScriptedProvider([_image_with_text(), _image_without_text()])
            _image, _prompt, attempts, result = webapp._generate_text_free(
                provider, "marathon runners", 600, 300
            )
        finally:
            text_check.ocr_available = original
        self.assertEqual(attempts, 1)
        self.assertFalse(result.available)


if __name__ == "__main__":
    unittest.main()
