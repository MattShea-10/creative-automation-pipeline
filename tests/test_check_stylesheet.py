"""The release check that the committed stylesheet is the compiled one.

Both builds serve static/tailwind-*.css exactly as committed, so a stale
copy means the .exe and the .dmg are styled from different bytes -- which
no screenshot and no source diff will show you.

The first version of this check compared bytes with `git status`, and
failed the build on its first run: the Windows runner checks text out as
CRLF, the Tailwind CLI writes LF, so an identical file differed on every
line. These tests exist mostly to keep that fixed.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import check_stylesheet

CSS = b"body{font-family:Inter}\n.wrap{max-width:640px}\n"


def _with_committed(data):
    return mock.patch.object(check_stylesheet, "committed", return_value=data)


class StaleTest(unittest.TestCase):
    def test_an_identical_file_is_not_stale(self):
        with _with_committed(CSS):
            self.assertEqual(check_stylesheet.stale("static/x.css", CSS), "")

    def test_line_endings_alone_are_not_staleness(self):
        """The bug this check shipped with: a CRLF checkout against an LF
        rebuild is the same stylesheet, and calling it stale fails every
        build on Windows forever."""
        with _with_committed(CSS):
            crlf = CSS.replace(b"\n", b"\r\n")
            self.assertEqual(check_stylesheet.stale("static/x.css", crlf), "")

    def test_a_real_change_is_stale(self):
        with _with_committed(CSS):
            changed = CSS.replace(b"640px", b"960px")
            problem = check_stylesheet.stale("static/x.css", changed)
            self.assertTrue(problem)
            self.assertIn("static/x.css", problem)

    def test_the_report_says_where_it_first_differs(self):
        """Minified CSS is one enormous line. A report that prints the
        whole file tells nobody anything."""
        with _with_committed(CSS):
            problem = check_stylesheet.stale("static/x.css", CSS.replace(b"640px", b"960px"))
            self.assertIn("line 2", problem)
            self.assertIn("960px", problem, "the report has to show what actually changed")
            self.assertIn("640px", problem)
            self.assertLess(len(problem), 1200)

    def test_the_report_finds_a_change_buried_deep_in_a_minified_line(self):
        """The real file is one 23,000-character line and the change is
        usually nowhere near the start. Printing the first few hundred
        characters showed two identical excerpts and explained nothing."""
        long_line = b"a" * 20000 + b"max-width:640px" + b"b" * 2000 + b"\n"
        with _with_committed(long_line):
            problem = check_stylesheet.stale("static/x.css", long_line.replace(b"640px", b"960px"))
            self.assertIn("960px", problem)
            self.assertLess(len(problem), 1200, "a window, not the whole line")

    def test_a_file_not_in_head_is_reported_rather_than_passed(self):
        """Silence here would mean a newly added stylesheet is never
        checked at all."""
        with _with_committed(None):
            self.assertIn("not committed", check_stylesheet.stale("static/x.css", CSS))

    def test_a_longer_rebuild_is_stale_even_with_no_differing_line(self):
        with _with_committed(CSS):
            self.assertTrue(check_stylesheet.stale("static/x.css", CSS + b".extra{color:red}\n"))


class LinesTest(unittest.TestCase):
    def test_both_line_endings_land_on_the_same_list(self):
        self.assertEqual(check_stylesheet.lines(b"a\r\nb\n"), check_stylesheet.lines(b"a\nb\n"))

    def test_undecodable_bytes_do_not_raise(self):
        """A half-written file should report a difference, not a crash in
        the check that was meant to catch it."""
        self.assertTrue(check_stylesheet.lines(b"\xff\xfe not utf-8"))


class WhatIsCheckedTest(unittest.TestCase):
    def test_both_stylesheets_and_the_stamp_are_covered(self):
        self.assertEqual(
            set(check_stylesheet.BUILT),
            {"static/tailwind-index.css", "static/tailwind-result.css", "static/styles.sha256"},
        )


if __name__ == "__main__":
    unittest.main()
