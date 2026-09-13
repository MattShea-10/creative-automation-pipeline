"""The no-text check on generated backdrops.

Two layers are being tested: find_text() itself (does it see lettering,
and does it stay quiet on a picture without any), and webapp's retry loop
around it (does a dirty result actually get regenerated, and does the
whole thing stay out of the way when OCR isn't installed).
"""

import sys
import contextlib
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

@contextlib.contextmanager
def _no_ocr():
    """Run with nothing able to read text out of a picture.

    find_text() does not consult ocr_available(); it runs the scene-text
    detector and then Tesseract, so patching ocr_available (which these
    tests used to do) left the detector running and the "couldn't check"
    path untested. Both engines have to go.
    """
    import src.text_check as text_check

    detector, tesseract = text_check.ensure_text_detector, text_check.tesseract_available
    text_check.ensure_text_detector = lambda *a, **k: None
    text_check.tesseract_available = lambda: False
    try:
        yield text_check
    finally:
        text_check.ensure_text_detector = detector
        text_check.tesseract_available = tesseract


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
        # over-read, not something a viewer would ever see -- and each
        # false flag costs a paid regeneration.
        #
        # The threshold is a fraction of the frame compared against the
        # detector's BOX, which is not the glyph: this 9px font comes
        # back a 26px box, nearly 3x the letters. MIN_HEIGHT_FRACTION is
        # set so that reading is still below the line.
        big = Image.new("RGB", (2000, 1200), (235, 235, 235))
        ImageDraw.Draw(big).text((20, 20), "tiny", fill=(0, 0, 0), font=_font(9))
        self.assertFalse(find_text(big).found_text)

    def test_a_headline_sized_word_is_still_read(self):
        # The other side of that threshold: raising it must not blind the
        # check to type a viewer plainly sees.
        page = Image.new("RGB", (1200, 600), (235, 235, 235))
        ImageDraw.Draw(page).text((60, 200), "SUMMER SALE", fill=(10, 10, 10), font=_font(90))
        self.assertTrue(find_text(page).found_text)


class _ScriptedProvider:
    """Returns a canned sequence of images, recording what it was sent."""

    name = "scripted"
    supports_negative_prompt = True

    def __init__(self, images):
        self.images = list(images)
        self.prompts = []
        self.negatives = []

    def generate(self, prompt, width=1024, height=1024, negative_prompt=None):
        self.prompts.append(prompt)
        self.negatives.append(negative_prompt)
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
        # The subject reaches the provider untouched. The exclusion goes
        # in the negative field, NOT appended to the prompt -- a
        # diffusion model handles negation there badly, and Ideogram's
        # docs say the positive prompt wins over the negative one, so
        # "no text" in the prompt is worse than useless.
        self.assertEqual(provider.prompts, ["marathon runners"])
        self.assertIn("no text", provider.negatives[0])
        # What's reported back says what was excluded, since it is no
        # longer visible in the prompt itself.
        self.assertIn("marathon runners", prompt)
        self.assertIn("excluded", prompt)

    def test_text_in_the_first_result_triggers_a_harder_retry(self):
        provider = _ScriptedProvider([_image_with_text(), _image_without_text()])
        _image, prompt, attempts, result = webapp._generate_text_free(
            provider, "marathon runners", 600, 300
        )
        self.assertEqual(attempts, 2)
        self.assertFalse(result.found_text)
        # The retry escalates rather than re-sending the phrasing that
        # has already demonstrably failed for this prompt -- on BOTH
        # channels. The negative prompt alone did not take the brand off
        # a volleyball, so NO_TEXT_RETRY_CLAUSE goes into the positive
        # prompt as well from the second attempt on.
        self.assertEqual(provider.prompts[0], "marathon runners")
        self.assertTrue(provider.prompts[1].startswith("marathon runners, "))
        self.assertIn(webapp.NO_TEXT_RETRY_CLAUSE, provider.prompts[1])
        self.assertNotIn(webapp.NO_TEXT_ESCALATION, provider.negatives[0])
        self.assertIn(webapp.NO_TEXT_ESCALATION, provider.negatives[1])

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


@unittest.skipUnless(ocr_available(), "Tesseract isn't installed")
class RemoveTextTest(unittest.TestCase):
    def test_small_lettering_is_painted_out_and_stays_out(self):
        dirty = _image_with_text("SALE", size=(1200, 900))
        found = find_text(dirty)
        self.assertTrue(found.found_text)
        cleaned, removed, reason = webapp.remove_text(dirty, found)
        self.assertIsNone(reason)
        self.assertGreater(removed, 0)
        # Verified, not assumed: inpainting can leave enough of a word
        # behind to still be read, and claiming a clean image that isn't
        # is worse than not trying.
        self.assertFalse(find_text(cleaned).found_text, find_text(cleaned).summary())

    def test_it_refuses_text_too_large_to_reconstruct(self):
        # Painting out means inventing what was behind the words. Across
        # half a frame that produces a smear more distracting than the
        # lettering was, so it declines instead of quietly wrecking the
        # image.
        huge = Image.new("RGB", (900, 300), (230, 230, 230))
        ImageDraw.Draw(huge).text((10, 40), "SALE", fill=(0, 0, 0), font=_font(240))
        found = find_text(huge)
        self.assertTrue(found.found_text)
        cleaned, removed, reason = webapp.remove_text(huge, found)
        self.assertIsNotNone(reason)
        self.assertEqual(removed, 0)
        self.assertIs(cleaned, huge)  # handed back untouched, not smeared

    def test_lettering_too_big_to_paint_over_is_refused_by_remove_text(self):
        # Painting a word out means inventing what was behind it, which
        # only convinces on small isolated lettering. remove_text() still
        # refuses anything larger rather than trade readable text for an
        # obvious smear: a 240px word fills 43% of this frame against a
        # 6% limit.
        from src.text_check import MAX_REMOVABLE_AREA_FRACTION, build_text_mask, masked_area_fraction, remove_text

        huge = Image.new("RGB", (900, 300), (230, 230, 230))
        ImageDraw.Draw(huge).text((10, 40), "SALE", fill=(0, 0, 0), font=_font(240))
        found = find_text(huge)
        fraction = masked_area_fraction(huge, build_text_mask(huge, found))
        self.assertGreater(fraction, MAX_REMOVABLE_AREA_FRACTION)
        cleaned, removed, reason = remove_text(huge, found)
        self.assertEqual(removed, 0)
        self.assertIn("too much to paint out", reason)
        self.assertEqual(list(cleaned.getdata()), list(huge.getdata()))

    def test_lettering_too_big_to_paint_is_cropped_off_rather_than_refused(self):
        # _clean_text_out() does NOT stop at that refusal any more. It
        # calls scrub_text(), which crops to the largest text-free band
        # when painting would replace the picture, then paints with
        # force=True -- "whatever it takes short of a new image". So the
        # caller gets a note saying what was done, not a warning saying
        # nothing could be. A warning now means something readable
        # survived scrubbing, which is a different fact.
        huge = Image.new("RGB", (900, 300), (230, 230, 230))
        ImageDraw.Draw(huge).text((10, 40), "SALE", fill=(0, 0, 0), font=_font(240))
        found = find_text(huge)
        _image, note, warning = webapp._clean_text_out(huge, found, "backdrop", 3)
        self.assertIsNone(warning)
        self.assertIsNotNone(note)
        self.assertIn("3 attempts", note)

    def test_a_success_is_reported_as_a_note_not_a_warning(self):
        dirty = _image_with_text("SALE", size=(1200, 900))
        found = find_text(dirty)
        _image, note, warning = webapp._clean_text_out(dirty, found, "backdrop", 1)
        self.assertIsNone(warning)
        self.assertIn("painted out", note)
        self.assertIn("1 attempt", note)


@unittest.skipUnless(ocr_available(), "Tesseract isn't installed")
class PageSegmentationTest(unittest.TestCase):
    def test_more_than_one_segmentation_mode_is_tried(self):
        # The shipped default was mode 3 alone ("fully automatic page
        # segmentation"), which assumes a scanned document. Benchmarked
        # against real generated backdrops with text painted on, it found
        # nothing at all -- 0 of 8, at every size tried -- so the check
        # was live, passing its own tests, and detecting nothing.
        from src.text_check import PAGE_SEGMENTATION_MODES

        self.assertGreater(len(PAGE_SEGMENTATION_MODES), 1)
        self.assertIn(6, PAGE_SEGMENTATION_MODES)
        self.assertIn(11, PAGE_SEGMENTATION_MODES)

    def test_the_same_word_found_by_two_modes_is_counted_once(self):
        # The modes overlap heavily. Triple-counting a word would inflate
        # the covered area and trip the "too large to paint out" limit on
        # a single short word.
        result = find_text(_image_with_text("SALE", size=(1200, 900)))
        boxes = [f.box for f in result.findings]
        self.assertEqual(len(boxes), len(set(boxes)))


class OcrUnavailableTest(unittest.TestCase):
    def test_a_missing_engine_is_reported_not_treated_as_clean(self):
        # "Couldn't check" and "checked, clean" are different facts and
        # must never collapse into each other -- the caller says so on
        # the results page rather than implying a verification that never
        # happened.
        with _no_ocr() as text_check:
            result = text_check.find_text(_image_with_text())
        self.assertFalse(result.available)
        self.assertFalse(result.found_text)

    def test_no_retries_are_spent_when_nothing_can_be_checked(self):
        with _no_ocr():
            provider = _ScriptedProvider([_image_with_text(), _image_without_text()])
            _image, _prompt, attempts, result = webapp._generate_text_free(
                provider, "marathon runners", 600, 300
            )
        self.assertEqual(attempts, 1)
        self.assertFalse(result.available)


if __name__ == "__main__":
    unittest.main()
