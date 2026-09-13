"""The .dmg launcher's "don't install backwards" guard.

The launcher copies the disk image's project over the installed one on
every launch. That is what makes a new dmg update an old install -- and
it is also how an old dmg, opened to check something, silently reverted
a working folder to two-day-old source. The only symptom was the app
behaving as though a run of fixes had been undone, which is a miserable
thing to debug, so the comparison it now makes is worth pinning down.

The guard is shell, so these drive it through bash rather than import
it. The launcher itself can't be run here (it ends in osascript and a
Terminal window), which is why the comparison lives in its own file.
"""

import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "macos" / "Creative Automation Pipeline.app"
GUARD = APP / "Contents" / "Resources" / "install_guard.sh"
LAUNCHER = APP / "Contents" / "MacOS" / "start"
WORKFLOW = REPO / ".github" / "workflows" / "windows-exe.yml"

BASH = shutil.which("bash")


def _write(path: Path, text: str, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (mtime, mtime))


@unittest.skipUnless(BASH and GUARD.is_file(), "needs bash and the launcher guard")
class ImageIsOlderThanInstallTests(unittest.TestCase):
    """image_is_older_than_install SOURCE DEST -- exit 0 means "copying
    this image over that install would go backwards"."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.image = self.tmp / "image"
        self.install = self.tmp / "install"
        self.now = time.time()

    def guard_says_older(self) -> bool:
        result = subprocess.run(
            [BASH, "-c", f'. "$1"; image_is_older_than_install "$2" "$3"',
             "_", str(GUARD), str(self.image), str(self.install)],
            capture_output=True, text=True,
        )
        self.assertIn(result.returncode, (0, 1), result.stderr)
        return result.returncode == 0

    def test_a_newer_image_installs(self):
        _write(self.image / "webapp.py", "new", self.now)
        _write(self.install / "webapp.py", "old", self.now - 86400)
        self.assertFalse(self.guard_says_older())

    def test_an_older_image_is_refused(self):
        _write(self.image / "webapp.py", "old", self.now - 86400)
        _write(self.install / "webapp.py", "new", self.now)
        self.assertTrue(self.guard_says_older())

    def test_the_first_install_is_never_blocked(self):
        """Nothing installed yet: the guard must not stand in the way of
        someone opening the dmg for the first time."""
        _write(self.image / "webapp.py", "new", self.now)
        self.assertFalse(self.guard_says_older())

    def test_an_install_with_no_source_is_not_newer(self):
        """A folder holding only a person's own output can't be dated as
        code, and an undecidable comparison lets the copy through."""
        _write(self.image / "webapp.py", "new", self.now)
        _write(self.install / "outputs" / "web" / "run.log", "x", self.now)
        self.assertFalse(self.guard_says_older())

    def test_the_users_own_files_dont_count_as_newer_code(self):
        """The install is always the side with fresh outputs, a .venv and
        cached images -- if those counted, every legitimate update would
        be refused."""
        _write(self.image / "webapp.py", "new", self.now)
        _write(self.install / "webapp.py", "old", self.now - 86400)
        for user_file in (
            self.install / "outputs" / "web" / "preferences.json",
            self.install / ".venv" / "lib" / "flask.py",
            self.install / "downloads" / "batch.json",
            self.install / "__pycache__" / "webapp.py",
            self.install / "_template_backups" / "keep.json",
        ):
            _write(user_file, "{}", self.now + 600)
        self.assertFalse(self.guard_says_older())

    def test_a_newer_template_counts_even_when_the_python_is_unchanged(self):
        """A release that only touches templates/ or styles/ is still a
        release; dating the tree by webapp.py alone would miss it."""
        _write(self.image / "webapp.py", "same", self.now - 86400)
        _write(self.image / "templates" / "index.html", "new", self.now)
        _write(self.install / "webapp.py", "same", self.now - 86400)
        _write(self.install / "templates" / "index.html", "old", self.now - 172800)
        self.assertFalse(self.guard_says_older())


@unittest.skipUnless(LAUNCHER.is_file(), "needs the launcher")
class LauncherWiringTests(unittest.TestCase):
    """The guard only helps if the launcher actually consults it, and
    the wiring is three lines of shell nobody runs in a test."""

    def setUp(self):
        self.script = LAUNCHER.read_text()

    @unittest.skipUnless(BASH, "needs bash")
    def test_the_launcher_is_valid_shell(self):
        result = subprocess.run([BASH, "-n", str(LAUNCHER)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_launcher_asks_the_guard_before_copying(self):
        self.assertIn("image_is_older_than_install", self.script)
        self.assertLess(
            self.script.index("image_is_older_than_install"),
            self.script.index("rsync -a"),
            "the guard has to run before the copy, not after it",
        )

    def test_going_backwards_needs_an_explicit_answer(self):
        """Every answer but the explicit one -- Cancel, Escape, a dialog
        that failed to appear at all -- leaves the install alone."""
        self.assertIn("COPY_FROM_IMAGE=no", self.script)
        self.assertIn('*) COPY_FROM_IMAGE=no ;;', self.script)

    def test_the_guard_ships_inside_the_app(self):
        """cp -R of the .app is what puts it on the dmg; a guard kept
        anywhere else would simply not be there at launch."""
        self.assertTrue(GUARD.is_file())
        self.assertIn("Contents/Resources/install_guard.sh", self.script)


@unittest.skipUnless(WORKFLOW.is_file(), "needs the release workflow")
class ReleaseWorkflowTests(unittest.TestCase):
    """One disk image per release, built from the tagged commit.

    The dmg used to be built by hand and dragged onto the release page.
    Two images from different days ended up in circulation, and opening
    the older one overwrote a newer install without a word -- twice in
    one evening, costing hours of confused debugging. Building it in the
    same workflow as the Windows zip is what stops there being a second
    image at all.
    """

    def setUp(self):
        self.text = WORKFLOW.read_text()

    def test_the_workflow_builds_the_disk_image(self):
        self.assertIn("macos-latest", self.text)
        self.assertIn("./macos/make_dmg.sh", self.text)

    def test_a_tag_publishes_both_downloads(self):
        self.assertIn("dist/CreativeAutomationPipeline-windows.zip", self.text)
        self.assertIn("files: CreativeAutomationPipeline.dmg", self.text)

    def test_the_image_is_checked_for_the_guard_before_it_is_published(self):
        """A published image without the guard is a downgrade waiting to
        happen, and nobody would notice until it had already fired."""
        self.assertIn("grep -q image_is_older_than_install", self.text)


if __name__ == "__main__":
    unittest.main()
