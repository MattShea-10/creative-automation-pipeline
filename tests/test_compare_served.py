"""The release check that compares the .exe's page with the source's.

The old smoke test asked the packaged app one question -- does it answer
HTTP 200 -- and a bundle missing templates/ or static/ answers 200 with a
different, unstyled page. So "does the .exe look like the .dmg" could
only be settled by opening it and looking. This makes it a build step,
which means the normalising has to be right: too little and every release
fails on a timestamp, too much and it stops noticing real damage.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_served import compare, normalise

PAGE = (
    '<html><head><link rel="stylesheet" href="/static/tailwind-index.css"></head>'
    '<body><input name="session_id" value="0123456789abcdef0123456789abcdef">'
    "<div class=\"campaign-card\">HydroBoost Sports Drink</div>"
    "<footer>quick-generate UI · code build: 2026-09-13 23:19:52</footer>"
    "</body></html>"
)


def _compare(a, b):
    return compare(a, b, "exe.html", "source.html")


class NoiseIsIgnoredTest(unittest.TestCase):
    """Differences that say nothing about the build."""

    def test_two_identical_pages_agree(self):
        self.assertEqual(_compare(PAGE, PAGE), "")

    def test_the_build_stamp_may_differ(self):
        """A checkout and a PyInstaller bundle date their files
        differently -- that is not a packaging fault."""
        other = PAGE.replace("2026-09-13 23:19:52", "2026-09-11 11:06:36")
        self.assertEqual(_compare(PAGE, other), "")

    def test_a_fresh_session_id_may_differ(self):
        other = PAGE.replace("0123456789abcdef0123456789abcdef", "ffffffffffffffffffffffffffffffff")
        self.assertEqual(_compare(PAGE, other), "")

    def test_the_port_may_differ(self):
        a = PAGE.replace("</body>", '<a href="http://127.0.0.1:5000/x">x</a></body>')
        b = PAGE.replace("</body>", '<a href="http://127.0.0.1:5051/x">x</a></body>')
        self.assertEqual(_compare(a, b), "")

    def test_line_endings_and_trailing_space_may_differ(self):
        self.assertEqual(_compare(PAGE + "\r\n", PAGE + "   \n"), "")


class RealDamageIsCaughtTest(unittest.TestCase):
    """The failures this exists for -- each one still answers HTTP 200."""

    def test_a_bundle_that_lost_the_stylesheet_is_caught(self):
        stripped = PAGE.replace('<link rel="stylesheet" href="/static/tailwind-index.css">', "")
        difference = _compare(stripped, PAGE)
        self.assertTrue(difference)
        self.assertIn("tailwind-index.css", difference)

    def test_a_page_missing_its_content_is_caught(self):
        stripped = PAGE.replace('<div class="campaign-card">HydroBoost Sports Drink</div>', "")
        self.assertTrue(_compare(stripped, PAGE))

    def test_a_wholly_different_page_is_caught(self):
        self.assertTrue(_compare("<html><body>Internal Server Error</body></html>", PAGE))

    def test_the_report_names_both_sides(self):
        difference = _compare(PAGE.replace("HydroBoost", "Something Else"), PAGE)
        self.assertIn("exe.html", difference)
        self.assertIn("source.html", difference)


class NormaliseTest(unittest.TestCase):
    def test_it_returns_lines_not_one_blob(self):
        """A line list is what makes the failure report readable; a single
        string diffs character by character and tells nobody anything."""
        self.assertIsInstance(normalise("a\nb\n"), list)
        self.assertEqual(normalise("a \nb\n"), ["a", "b", ""])

    def test_a_hex_id_of_the_wrong_length_is_left_alone(self):
        """Only the ids the app actually mints are noise. A short hex
        string in the page could be a colour, and colours matter."""
        self.assertIn("#ffd100", normalise('<span style="color:#ffd100">x</span>')[0])


if __name__ == "__main__":
    unittest.main()
