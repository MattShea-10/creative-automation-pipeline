"""Image transforms: pixel-size cropping and campaign-message overlay."""

from __future__ import annotations

import math
import re
import sys
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import logging

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Video files a hero "image" can also be sourced from -- a single frame is
# extracted (see extract_video_frame()) and used exactly like any other
# hero image from that point on. Requires opencv-python(-headless), listed
# in requirements.txt.
VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")


_log = logging.getLogger(__name__)


def extract_video_frame(path: Union[str, Path], frame_seconds: Optional[float] = None) -> Image.Image:
    """Grab a single frame from a video file and return it as an RGB image.

    Defaults to the middle frame of the video when `frame_seconds` isn't
    given, since the first frame of a lot of real-world video is a black
    frame, a fade-in, or a title card -- the middle is a much safer generic
    default for "a representative frame of this product video." Pass
    `frame_seconds` (a per-product `video_frame_seconds` in the brief) to
    pick a specific timestamp instead, e.g. one you know shows the product
    clearly.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ValueError(
            f"Could not open video asset '{path}': the 'opencv-python-headless' package "
            "isn't installed. Run `pip install -r requirements.txt` (or `pip install "
            "opencv-python-headless` directly) to enable video hero images."
        ) from exc

    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(
            f"Could not open video asset '{path}' -- the file may be corrupted, or use a "
            "container/codec OpenCV can't decode on this machine. Try re-exporting it as "
            "H.264 MP4, which is broadly supported."
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = (frame_count / fps) if fps > 0 else 0

    if frame_seconds is None:
        target_seconds = duration / 2 if duration > 0 else 0
    else:
        target_seconds = max(0.0, min(frame_seconds, duration)) if duration > 0 else max(0.0, frame_seconds)

    frame = None
    if target_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, target_seconds * 1000)
        ok, candidate = cap.read()
        if ok:
            frame = candidate

    if frame is None:
        # Millisecond-based seeking is unreliable on some containers/codecs
        # -- fall back to reading frames sequentially up to the target.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        target_frame_index = int(target_seconds * fps) if fps > 0 else 0
        last_good = None
        for _ in range(max(target_frame_index, 0) + 1):
            ok, candidate = cap.read()
            if not ok:
                break
            last_good = candidate
        frame = last_good

    cap.release()
    if frame is None:
        raise ValueError(f"Could not read any frame from video asset '{path}'.")

    # OpenCV decodes frames as BGR -- flip channel order for PIL/RGB.
    rgb = frame[:, :, ::-1]
    return Image.fromarray(rgb, mode="RGB")


def open_as_rgb(path: Union[str, Path], frame_seconds: Optional[float] = None) -> Image.Image:
    """Open an image OR video file and return it as RGB.

    If `path` is a video file (see VIDEO_EXTENSIONS), a single frame is
    extracted instead -- see extract_video_frame(). `frame_seconds` is
    ignored for non-video files.
    """
    path = Path(path)
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return extract_video_frame(path, frame_seconds=frame_seconds)
    if path.suffix.lower() == ".psd":
        flat = open_psd_flat(path)
        if flat is not None:
            return flat
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not open asset '{path}': {exc}.") from exc


def open_psd_flat(path: Union[str, Path]) -> Optional[Image.Image]:
    """A PSD's merged picture as RGB, read with psd-tools rather than
    Pillow.

    The picture Photoshop stored in the file is what is used when there
    is one: it is Photoshop's own render, with every layer effect (a
    drop shadow, a glow, a stroke) drawn the way the designer saw it.
    psd-tools can redraw the layers itself, but it does not draw
    shadows or glows -- a header with a drop shadow came back flat -- so
    that redraw is the fallback, for a file with no stored preview.

    Why psd-tools and not Pillow for the stored preview: Pillow's PSD
    reader assumes it has at most four channels, and a file saved with
    a fifth (a spot or extra alpha channel) is read with its planes
    misaligned -- bands of shifted colour, one layer's lettering
    showing through another. psd-tools reads the same bytes correctly.

    A preview is only a snapshot, so this app rewrites it whenever it
    edits a file's layers (see psd_export.set_flattened_preview). None
    when psd-tools can't open the file, so the caller can try Pillow.
    """
    try:
        from psd_tools import PSDImage

        psd = PSDImage.open(path)
    except Exception:  # noqa: BLE001
        return None
    try:
        preview = psd.composite()  # the stored picture when the file has one
    except Exception:  # noqa: BLE001
        preview = None
    if preview is not None and getattr(psd, "has_preview", lambda: True)():
        return preview.convert("RGB")
    try:
        drawn = psd.composite(force=True)
    except Exception:  # noqa: BLE001
        drawn = None
    if drawn is not None:
        return drawn.convert("RGB")
    return preview.convert("RGB") if preview is not None else None


# Default render targets, as explicit pixel dimensions. Kept as a list (not
# a dict keyed by ratio name) since sizes are now first-class WxH pixel
# pairs -- a human-readable ratio label (e.g. "1:1") is derived from the
# pixels via ratio_label() below, rather than being the source of truth.
DEFAULT_SIZES: List[Tuple[int, int]] = [
    (1080, 1080),   # 1:1
    (1080, 1920),   # 9:16
    (1920, 1080),   # 16:9
]

# Common web/display ad sizes, in pixels -- as specified by the user
# (desktop placements, one mobile placement, plus commonly-used additional
# sizes). Kept under the "web-top7" preset name for continuity even though
# this list has grown to 9 entries.
WEB_AD_SIZES: List[Tuple[int, int]] = [
    (728, 90),    # Leaderboard
    (300, 250),   # Medium Rectangle
    (336, 280),   # Large Rectangle
    (160, 600),   # Skyscraper
    (320, 50),    # Mobile Leaderboard
    (250, 250),   # Square
    (200, 200),   # Small Square
    (468, 60),    # Banner
    (970, 90),    # Large Leaderboard
]

# If these same creatives were extended to video, the pixel dimensions a
# broadcast/TV delivery would target. Unlike web, broadcast standardizes
# on 16:9 -- there's no vertical or square broadcast format -- so this is
# a short list of resolution tiers rather than a variety of aspect ratios.
BROADCAST_VIDEO_SIZES: List[Tuple[int, int]] = [
    (1920, 1080),  # Full HD (1080i/1080p) -- the primary US broadcast delivery standard
    (1280, 720),   # 720p -- used by some networks (e.g. Fox/Disney-owned) instead of 1080i
    (3840, 2160),  # 4K UHD -- increasingly requested for premium/streaming-adjacent delivery
]

# Friendly names for well-known sizes, used to make filenames/reports more
# readable when a requested size matches a recognized standard. Includes a
# few sizes (300x600 "Half Page Ad", 970x250 "Billboard", 720x480 "Wide
# Rectangle") that aren't in the active WEB_AD_SIZES preset above but are
# still recognized if requested explicitly via --sizes. "Wide Rectangle"
# isn't an official IAB name -- there isn't one for this size -- just a readable
# label for the 720x480 (3:2) delivery size these templates use. The
# 728x480 that used to sit here was a near miss of it, and typing the
# real size then drew a "did you mean 728x480" warning.
SIZE_NAMES = {
    (728, 90): "Leaderboard",
    (300, 250): "Medium Rectangle",
    (336, 280): "Large Rectangle",
    (160, 600): "Skyscraper",
    (320, 50): "Mobile Leaderboard",
    (250, 250): "Square",
    (200, 200): "Small Square",
    (468, 60): "Banner",
    (970, 90): "Large Leaderboard",
    (720, 480): "Wide Rectangle",
    (300, 600): "Half Page Ad",
    (970, 250): "Billboard",
    (1920, 1080): "Full HD / 1080p Broadcast",
    (1280, 720): "720p Broadcast",
    (3840, 2160): "4K UHD Broadcast",
    (1200, 627): "LinkedIn Article",  # link-share / article image, 1.91:1
}

SIZE_PRESETS = {
    "default": DEFAULT_SIZES,
    "web-top7": WEB_AD_SIZES,
    "broadcast": BROADCAST_VIDEO_SIZES,
}

# Device placement for the standard display ad sizes -- confirmed against
# published IAB ad-size guides (see README Sources): only 320x50 is a
# mobile-specific unit; the rest are desktop/web placements. This is only
# meaningful for classic display "ad sizes" -- the social defaults (1:1,
# 9:16, 16:9) and broadcast/video resolutions aren't display ad units, so
# they're intentionally left uncategorized (device_category() -> None).
DEVICE_CATEGORY = {
    (728, 90): "desktop",
    (300, 250): "desktop",
    (336, 280): "desktop",
    (160, 600): "desktop",
    (320, 50): "mobile",
    (250, 250): "desktop",
    (200, 200): "desktop",
    (468, 60): "desktop",
    (970, 90): "desktop",
    (300, 600): "desktop",
    (970, 250): "desktop",
    (720, 480): "desktop",
}


def device_category(width: int, height: int):
    """'mobile' or 'desktop' for a recognized display ad size, else None."""
    return DEVICE_CATEGORY.get((width, height))


_SIZE_RE = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")


def size_name(width: int, height: int) -> str:
    """Friendly name for a known standard size, falling back to its ratio label.

    The three built-in default sizes (1080x1080 / 1080x1920 / 1920x1080)
    are excluded even though 1920x1080 also happens to equal Full HD --
    when used as a plain 16:9 social ratio, "16:9" is the more useful label
    than "Full HD / 1080p Broadcast".
    """
    if (width, height) in DEFAULT_SIZES:
        return ratio_label(width, height)
    return SIZE_NAMES.get((width, height), ratio_label(width, height))


def parse_size(spec: str) -> Tuple[int, int]:
    """Parse a single 'WIDTHxHEIGHT' string, e.g. '1080x1080' -> (1080, 1080)."""
    m = _SIZE_RE.match(spec)
    if not m:
        raise ValueError(f"Invalid size '{spec}'. Expected format WIDTHxHEIGHT, e.g. 1080x1080.")
    w, h = int(m.group(1)), int(m.group(2))
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid size '{spec}': width and height must be positive.")
    return w, h


def parse_sizes(spec: str) -> List[Tuple[int, int]]:
    """Resolve a sizes argument into a deduplicated, order-preserving list of
    (width, height) pixel pairs.

    `spec` is a comma-separated list where each item is *either* a known
    preset name ('default', 'web-top7', 'broadcast') or an explicit
    'WIDTHxHEIGHT' pair -- and the two can be freely mixed, so a single
    campaign can render more than one size family in one run, e.g.:
        "default,web-top7"        -> all 3 social sizes + all 9 web ad sizes
        "web-top7,broadcast"      -> all 9 web ad sizes + all 3 broadcast sizes
        "default,1200x628"        -> the 3 social sizes plus one extra custom size
    Preset names are case-insensitive. Exact pixel-size duplicates across
    presets (e.g. 1920x1080 appearing in both 'default' and 'broadcast')
    are only rendered once.
    """
    parts = [p for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("No sizes provided.")

    resolved: List[Tuple[int, int]] = []
    for part in parts:
        key = part.strip().lower()
        if key in SIZE_PRESETS:
            resolved.extend(SIZE_PRESETS[key])
        else:
            resolved.append(parse_size(part))

    # De-duplicate while preserving first-seen order.
    seen = set()
    deduped = []
    for size in resolved:
        if size not in seen:
            seen.add(size)
            deduped.append(size)
    return deduped


def ratio_label(width: int, height: int) -> str:
    """Derive a human-readable aspect ratio label (e.g. '16:9') from pixel dimensions.

    Sizes that don't reduce to small numbers (1200x627 is 400:209 exactly)
    are labelled the way their platforms quote them -- '1.91:1' -- since
    a reduced fraction nobody recognises says less than the decimal.
    """
    g = math.gcd(width, height) or 1
    w, h = width // g, height // g
    if max(w, h) <= 40 or not width or not height:
        return f"{w}:{h}"
    if width >= height:
        return f"{width / height:.2f}:1"
    return f"1:{height / width:.2f}"


def size_label(width: int, height: int) -> str:
    """Filename/report label for a given render size, e.g. '1080x1080'."""
    return f"{width}x{height}"


# A small, curated set of typeface "families" -- (bold filename, regular
# filename) pairs -- for text that lets the caller pick a font, rather than
# the single hardcoded DejaVu Sans every other piece of text still uses.
# All four ship with the same DejaVu font package already relied on
# elsewhere, so there's no new dependency and no risk of a missing-font
# fallback to the (much uglier, fixed-size) PIL default bitmap font.
_FONT_FAMILIES = {
    "sans": ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"),  # default -- clean, modern sans-serif
    "serif": ("DejaVuSerif-Bold.ttf", "DejaVuSerif.ttf"),  # classic, traditional look
    "mono": ("DejaVuSansMono-Bold.ttf", "DejaVuSansMono.ttf"),  # fixed-width, technical/code look
    "condensed": ("DejaVuSansCondensed-Bold.ttf", "DejaVuSansCondensed.ttf"),  # narrow, fits more text per line
}
VALID_FONT_FAMILIES = tuple(_FONT_FAMILIES.keys())


# Fonts are bundled in fonts/ (next to this project's other top-level
# folders like default_templates/) rather than relied upon from the OS.
# ImageFont.truetype("DejaVuSans.ttf", ...) with a bare filename only
# resolves on systems that happen to have that exact file on a path
# FreeType searches (e.g. Debian/Ubuntu's /usr/share/fonts/truetype/dejavu/)
# -- it silently fails on macOS and Windows, where PIL falls back to
# ImageFont.load_default(), a tiny FIXED-SIZE bitmap font that ignores
# whatever size was requested. That failure mode is *silent*: every call
# still returns a usable font object, so nothing raises or logs -- it just
# renders every piece of overlay text at ~10px forever, on every size, no
# matter what font size was asked for, which is exactly the "text stuck
# tiny" bug this bundling fixes. Bundling these files means font loading
# never depends on what's installed on the host OS at all.
# Inside a packaged build (PyInstaller) the fonts ship in the bundle's
# extraction dir, not next to a source tree.
_FONTS_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)) / "fonts"

_FONT_LOAD_WARNED = False


# The fonts installed on this machine, by PostScript name, so text
# redrawn into a template can use the face the designer set it in
# rather than the nearest bundled DejaVu. Photoshop stores a text
# layer's font as its PostScript name ("MyriadPro-Regular",
# "AvenirNextCondensed-DemiBold"); every font file carries that same
# name in its name table (nameID 6), so matching is exact. Built once
# per process, lazily: reading name tables from a few hundred files
# takes a second or two.
_SYSTEM_FONT_DIRS = [
    Path("/System/Library/Fonts"),
    Path("/System/Library/Fonts/Supplemental"),
    Path("/Library/Fonts"),
    Path.home() / "Library" / "Fonts",
    # Adobe Fonts (Creative Cloud) activates fonts into a hidden cache.
    Path.home() / "Library" / "Application Support" / "Adobe" / "CoreSync" / "plugins" / "livetype" / ".r",
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts",
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
    Path.home() / ".local" / "share" / "fonts",
]
_FONT_FILE_SUFFIXES = (".ttf", ".otf", ".ttc", ".otc")
_font_index_cache: Optional[dict] = None


def _font_index() -> dict:
    """{postscript name (lower): (path, face index)} for every font file
    in the system font folders; empty when fontTools isn't available."""
    global _font_index_cache
    if _font_index_cache is not None:
        return _font_index_cache
    index: dict = {}
    try:
        from fontTools.ttLib import TTFont, TTCollection
    except ImportError:
        TTFont = TTCollection = None  # noqa: N806 -- filename matching only, below
    seen_dirs = set()
    for folder in _SYSTEM_FONT_DIRS:
        try:
            folder = folder.resolve()
        except Exception:  # noqa: BLE001
            continue
        if not folder.is_dir() or folder in seen_dirs:
            continue
        seen_dirs.add(folder)
        for path in folder.rglob("*"):
            if path.suffix.lower() not in _FONT_FILE_SUFFIXES or not path.is_file():
                continue
            # By filename too, as a fallback key: "Apple Symbols.ttf" is
            # AppleSymbols, "MyriadPro-Regular.otf" is MyriadPro-Regular.
            # Without fontTools this is the only key there is; with it,
            # the name table wins for the same key.
            index.setdefault(_font_key(path.stem), (path, 0))
            # Pillow can open each face of a file and report its family
            # and style ("Avenir Next Condensed", "Demi Bold"), which
            # joined is the PostScript name's key -- so a collection
            # like macOS's Avenir Next Condensed.ttc resolves face by
            # face even without fontTools.
            face_no = 0
            while face_no < 64:
                try:
                    probe = ImageFont.truetype(str(path), 10, index=face_no)
                    family, style = probe.getname()
                except Exception:  # noqa: BLE001
                    break
                if family:
                    index.setdefault(_font_key(f"{family}{style or ''}"), (path, face_no))
                    if (style or "").lower() in ("regular", "book", "roman", "plain", ""):
                        index.setdefault(_font_key(family), (path, face_no))
                face_no += 1
                if path.suffix.lower() not in (".ttc", ".otc"):
                    break
            if TTFont is None:
                continue
            try:
                if path.suffix.lower() in (".ttc", ".otc"):
                    faces = list(TTCollection(str(path), lazy=True).fonts)
                else:
                    faces = [TTFont(str(path), lazy=True)]
                for i, face in enumerate(faces):
                    try:
                        ps_name = face["name"].getDebugName(6)
                    except Exception:  # noqa: BLE001
                        ps_name = None
                    if ps_name:
                        index[_font_key(ps_name)] = (path, i)
                    try:
                        face.close()
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001
                continue
    _font_index_cache = index
    return index


def _font_key(name: str) -> str:
    """A PostScript name or filename reduced to letters and digits, so
    "Apple Symbols", "AppleSymbols" and "Apple-Symbols" all meet."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def find_font_file(postscript_name: Optional[str]):
    """(path, face index) of the installed font with this PostScript
    name, or None. Adobe's "AdobeInvisFont" placeholder and empty names
    are never matched."""
    if not postscript_name:
        return None
    key = _font_key(postscript_name)
    if not key or key == "adobeinvisfont":
        return None
    return _font_index().get(key)


def _load_font(
    size: int, bold: bool = True, family: str = "sans", font_name: Optional[str] = None
) -> ImageFont.FreeTypeFont:
    global _FONT_LOAD_WARNED
    # The template's own face first, when it is installed here.
    found = find_font_file(font_name)
    if found is not None:
        path, face_index = found
        try:
            return ImageFont.truetype(str(path), size=size, index=face_index)
        except Exception:  # noqa: BLE001
            pass
    bold_name, regular_name = _FONT_FAMILIES.get(family, _FONT_FAMILIES["sans"])
    name = bold_name if bold else regular_name
    candidates = [_FONTS_DIR / name, Path(name)]
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except Exception:
            continue
    if not _FONT_LOAD_WARNED:
        # This should never actually trigger now that fonts/ is bundled --
        # if it does, every overlay text render silently degrades to a
        # fixed ~10px bitmap font regardless of requested size, which is
        # exactly the bug this loud warning exists to make impossible to
        # miss a second time.
        print(
            f"[image_ops] WARNING: could not load bundled font {name!r} from "
            f"{_FONTS_DIR} -- falling back to PIL's tiny fixed-size default "
            "font. Text overlays will render far too small. Check that the "
            "fonts/ folder shipped with this project and wasn't excluded."
        )
        _FONT_LOAD_WARNED = True
    return ImageFont.load_default()


def center_crop_to_ratio(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Center-crop `image` to the target aspect ratio, then resize to exact pixels.

    This is the classic 'fill' strategy: crop away the longer dimension so
    no letterboxing/bars are introduced, matching how social platforms
    expect creative to fill the full frame.
    """
    target_w, target_h = target_size
    target_ratio = target_w / target_h
    src_w, src_h = image.size
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        # source is wider than target -> crop left/right
        new_w = int(src_h * target_ratio)
        offset = (src_w - new_w) // 2
        box = (offset, 0, offset + new_w, src_h)
    else:
        # source is taller than target -> crop top/bottom
        new_h = int(src_w / target_ratio)
        offset = (src_h - new_h) // 2
        box = (0, offset, src_w, offset + new_h)

    cropped = image.crop(box)
    return cropped.resize(target_size, Image.LANCZOS)


def resize_to_contain(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Scale `image` down/up to fit entirely within `target_size` -- no cropping
    -- and letterbox the remaining space with a softly blurred, filled
    version of the same image rather than plain bars.

    This is the 'fit' strategy, as opposed to center_crop_to_ratio()'s
    'fill' strategy. Use it for a finished, already-composed creative
    (e.g. a flattened Photoshop export) where every pixel -- text, logo,
    a call-to-action -- was placed deliberately and cropping any of it off
    would break the design. A generic product photo usually looks fine
    cropped; a finished ad creative usually does not.
    """
    target_w, target_h = target_size
    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(int(round(src_w * scale)), 1)
    new_h = max(int(round(src_h * scale)), 1)
    fitted = image.resize((new_w, new_h), Image.LANCZOS)

    if new_w == target_w and new_h == target_h:
        return fitted.convert("RGB")

    # Background: the same image, cropped-to-fill the frame (so it covers
    # every pixel) and then heavily blurred, so the letterbox bars read as
    # an intentional soft backdrop rather than dead space or a hard color.
    background = center_crop_to_ratio(image, target_size)
    blur_radius = max(target_w, target_h) // 30
    if blur_radius > 0:
        background = background.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    canvas = background.convert("RGB")
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    canvas.paste(fitted, offset)
    return canvas


def wrap_text_to_width(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int
) -> List[str]:
    """Word-wrap `text` so every line fits within `max_width` at the given font.

    Falls back to a hard character-break if a single word alone is wider
    than max_width (e.g. a very long word at a small canvas size), so a
    line can never silently overflow the frame.
    """
    lines: List[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        current = ""
        for word in words:
            trial = f"{current} {word}".strip()
            if current and draw.textlength(trial, font=font) > max_width:
                # Doesn't fit onto the line so far -- flush it and start a
                # new line with just this word.
                lines.append(current)
                current = word
            else:
                # Either it fits, or `current` was empty and this is the
                # line's first word -- always accepted, since a line has to
                # start with *something*.
                current = trial
            # Whatever `current` holds now -- a fresh line's first word, or
            # an accumulated line the new word still fit onto -- might
            # itself be wider than max_width on its own (a single very long
            # word, e.g. a header with no spaces at a large autofit size).
            # Hard-break it right here rather than only when a *later*
            # word fails to fit onto it -- otherwise a too-wide word that's
            # the very first thing on a line (nothing accumulated yet to
            # trigger the "doesn't fit" branch above) never gets broken at
            # all and silently overflows the frame. This is a no-op
            # whenever `current` already fits.
            while draw.textlength(current, font=font) > max_width and len(current) > 1:
                lo, hi = 1, len(current)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if draw.textlength(current[:mid], font=font) <= max_width:
                        lo = mid
                    else:
                        hi = mid - 1
                lines.append(current[:lo])
                current = current[lo:]
        if current:
            lines.append(current)
    return lines or [""]


def fit_text_block(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_height: int,
    min_font_size: int = 14,
    max_font_size: Optional[int] = None,
    family: str = "sans",
    bold: bool = True,
    leading: Optional[int] = None,
    leading_reference_size: Optional[int] = None,
    font_name: Optional[str] = None,
) -> Tuple[ImageFont.FreeTypeFont, List[str], int]:
    """Find the *largest* font size whose wrapped text fits within (max_width, max_height).

    Returns (font, wrapped_lines, line_height). Binary-searches font sizes
    between `min_font_size` and `max_font_size` (which defaults to a
    generous cap derived from `max_height`), using wrap_text_to_width() at
    each candidate size to see how many lines it wraps to.

    This replaces an earlier shrink-only design that started from a single
    guessed font size and only ever got smaller. That worked fine when the
    guess happened to be reasonable, but it had no way to grow text toward
    the available space -- so a short headline on a spacious, wide frame
    (e.g. a 4:3 creative) stayed small even though the frame had plenty of
    room, and a narrow-but-tall frame (e.g. a 160x600 skyscraper) could get
    stuck at a tiny size because the guess was based on the cramped width,
    ignoring the abundant height. Searching for the largest font that fits
    -- rather than shrinking from a fixed starting point -- grows to fill
    whichever dimension (width or height) is actually generous, and still
    shrinks all the way down to `min_font_size` for long text in a small
    box, so it's a strict improvement in both directions.
    """
    if max_font_size is None:
        max_font_size = max(int(max_height), min_font_size)
    max_font_size = max(max_font_size, min_font_size)

    def layout_for(size: int):
        font = _load_font(size, bold=bold, family=family, font_name=font_name)
        lines = wrap_text_to_width(draw, text, font, max_width)
        if leading and leading_reference_size:
            # Use the PSD's own leading, scaled proportionally to this
            # candidate size, for the *fit check itself* -- not just for
            # the final render. Previously the search validated candidates
            # against a generic ~1.2x-of-font-size guess, then the caller
            # swapped in the PSD-accurate (often larger) leading afterward
            # without re-checking it still fit -- so a "fits" result from
            # the search could still overflow once actually drawn. Feeding
            # the real leading into the search itself makes that
            # impossible: whatever size wins here is guaranteed to still
            # fit once rendered with this same line_height.
            line_height = max(round(leading * (size / leading_reference_size)), 1)
        else:
            line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + int(size * 0.3)
        total_height = line_height * len(lines)
        return font, lines, line_height, total_height

    # The floor is always an acceptable fallback, even if it still overflows
    # max_height (matching the old behavior of never going below min_font_size).
    best = layout_for(min_font_size)

    lo, hi = min_font_size, max_font_size
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = layout_for(mid)
        if candidate[3] <= max_height:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1

    font, lines, line_height, _total_height = best
    return font, lines, line_height


VALID_TEXT_ALIGNMENTS = ("left", "center", "right")


def _add_edge_banner(
    image: Image.Image,
    text: str,
    *,
    edge: str,
    max_height_frac: float,
    align: str = "left",
    reserved_right: int = 0,
    reserved_left: int = 0,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    show_background: bool = True,
    glow: bool = False,
    glow_color: Tuple[int, int, int] = (255, 255, 255),
    font_size: Optional[int] = None,
    _rgba: bool = False,
) -> Image.Image:
    """Shared implementation behind add_message_banner() and add_header_banner().

    Overlays a banner (top or bottom edge) with `text`. The banner grows to
    fit the wrapped text (up to `max_height_frac` of the frame), and --
    unless `font_size` pins an exact size -- the font size is chosen by
    fit_text_block() to be the *largest* size that still fits the available
    width/height, so a short headline on a spacious frame renders big
    enough to use the space, a narrow-but-tall frame (e.g. a skyscraper)
    gets a legible size driven by its abundant height rather than its
    cramped width, and long text still wraps and shrinks down to a sane
    floor instead of overflowing the frame.

    `align` is one of "left", "center", "right" -- which edge of the usable
    text area each wrapped line is anchored to.

    `font_size`, if given, pins the text to that exact pixel size instead of
    autofitting -- the banner grows (up to the full frame height, as a hard
    safety limit) to accommodate whatever that size needs, rather than
    clipping it to the usual `max_height_frac` cap, since a size the caller
    explicitly chose shouldn't be silently overridden.

    `reserved_right`/`reserved_left` keep text out of a strip along the
    right/left edge (in pixels) -- used to keep a headline from running
    underneath the brand logo when it's composited in the top-right or
    top-left corner, respectively.

    `text_color` is the RGB color the text itself is drawn in (default
    white). `show_background` controls whether a semi-transparent black
    plate is drawn behind the text (the original/default look, guaranteeing
    contrast against any hero image) or the text floats directly over the
    image with no plate at all. When there's no plate, a thin outline is
    added around the text instead -- automatically black or white depending
    on how light `text_color` is -- since without a background there's
    otherwise no guarantee the chosen text color will be legible against
    whatever the hero image looks like at that spot.

    `glow`/`glow_color` add a soft colored halo behind the text (a blurred,
    colorized copy of the text mask, composited under the crisp foreground
    text) -- a stylistic alternative to the plain outline. When `glow` is
    on, the automatic contrast outline is skipped so the two effects don't
    visually clash.

    `_rgba` is an internal knob (not exposed by add_header_banner()/
    add_message_banner()'s own signatures, only forwarded from callers that
    know what they're doing -- see build_layered_psd() in src/psd_export.py)
    that returns the RGBA composite as-is instead of flattening it to RGB.
    Since the banner overlay's own content never depends on `image`'s
    pixels (only its size), calling this with a fully transparent `image`
    and `_rgba=True` yields exactly the banner's own pixels, isolated on
    transparency -- usable as its own layer.
    """
    if align not in VALID_TEXT_ALIGNMENTS:
        raise ValueError(f"align must be one of {VALID_TEXT_ALIGNMENTS}, got {align!r}")

    img = image.convert("RGBA")
    w, h = img.size

    padding = max(int(w * 0.05), 4)
    # The usable text area is [left_bound, right_bound) -- reserved_left/
    # reserved_right carve out strips on either edge (e.g. for a top-left
    # or top-right logo), never shrinking the remaining area below 30% of
    # the frame width.
    min_usable_w = int(w * 0.3)
    left_bound = min(reserved_left, max(w - min_usable_w, 0))
    right_bound = max(w - reserved_right, left_bound + min_usable_w)
    usable_w = right_bound - left_bound
    max_text_width = max(usable_w - 2 * padding, 10)
    max_banner_height = max(int(h * max_height_frac), 12)

    draw_probe = ImageDraw.Draw(img)
    explicit_font_size = font_size is not None
    if explicit_font_size:
        # An explicit size -- skip autofit entirely and just wrap at it.
        font = _load_font(font_size)
        lines = wrap_text_to_width(draw_probe, text, font, max_text_width)
        line_height = draw_probe.textbbox((0, 0), "Ag", font=font)[3] + int(font_size * 0.3)
    else:
        # Compact web ad formats (e.g. a 320x50 mobile leaderboard or 728x90
        # leaderboard) are far shorter than the social aspect ratios this
        # template was designed around, so the floor a font is allowed to
        # shrink to also scales down with the available height -- otherwise
        # a 14px floor alone can still be too tall for a 50px-high banner.
        min_font_size = max(min(int(h * 0.18), 14), 7)
        font, lines, line_height = fit_text_block(
            draw_probe,
            text,
            max_text_width,
            max(max_banner_height - int(h * 0.06), 8),
            min_font_size=min_font_size,
        )
    font_size = font.size

    text_block_height = line_height * len(lines)
    banner_height_cap = h if explicit_font_size else max_banner_height
    banner_height = min(max(text_block_height + int(h * 0.06), int(h * 0.14)), banner_height_cap)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if edge == "top":
        band_box = [(0, 0), (w, banner_height)]
        text_top = (banner_height - text_block_height) / 2
    else:
        band_box = [(0, h - banner_height), (w, h)]
        text_top = h - banner_height + (banner_height - text_block_height) / 2
    if show_background:
        draw.rectangle(band_box, fill=(0, 0, 0, 150))

    # Compute each line's draw position once -- reused for both the glow
    # pass (if any) and the final crisp text pass so they line up exactly.
    positions = []
    y = text_top
    for line in lines:
        line_width = draw.textlength(line, font=font)
        if align == "center":
            x = left_bound + max((usable_w - line_width) / 2, padding)
        elif align == "right":
            x = left_bound + max(usable_w - line_width - padding, padding)
        else:
            x = left_bound + padding
        positions.append((x, y, line))
        y += line_height

    stroke_width = 0
    stroke_fill = None
    if not show_background and not glow:
        # No background plate to guarantee contrast -- outline the text
        # instead. Pick a black or white outline based on the chosen text
        # color's perceived brightness, so e.g. dark text over a dark photo
        # still reads, and light text over a light photo still reads too.
        luminance = 0.299 * text_color[0] + 0.587 * text_color[1] + 0.114 * text_color[2]
        stroke_fill = (0, 0, 0, 255) if luminance > 140 else (255, 255, 255, 255)
        stroke_width = max(round(min(w, h) * 0.006), 2)

    if glow:
        # Render the text as a plain white mask on its own transparent
        # layer, blur its alpha channel into a soft halo, boost the
        # blurred alpha back up (blurring dims it substantially), tint it
        # with `glow_color`, and composite that colored halo onto the
        # overlay *before* the crisp foreground text -- so the glow sits
        # behind the sharp letterforms rather than washing over them.
        mask_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_layer)
        for x, y, line in positions:
            mask_draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))

        blur_radius = max(round(font_size * 0.22), 3)
        alpha = mask_layer.split()[3].filter(ImageFilter.GaussianBlur(radius=blur_radius))
        alpha = alpha.point(lambda a: min(255, int(a * 1.8)))
        colored_glow = Image.new("RGBA", img.size, (glow_color[0], glow_color[1], glow_color[2], 0))
        colored_glow.putalpha(alpha)

        overlay = Image.alpha_composite(overlay, colored_glow)
        draw = ImageDraw.Draw(overlay)  # alpha_composite() returns a new Image -- rebind the Draw handle to it

    fill = (text_color[0], text_color[1], text_color[2], 255)
    for x, y, line in positions:
        draw.text((x, y), line, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)

    combined = Image.alpha_composite(img, overlay)
    return combined if _rgba else combined.convert("RGB")


def message_banner_height(
    image_size: Tuple[int, int],
    message: str,
    *,
    font_size: Optional[int] = None,
) -> int:
    """The pixel height add_message_banner() will use to render `message`
    at `image_size`, computed without actually drawing anything.

    Lets a caller reserve/avoid that space -- specifically,
    render_creative()'s `cta_above_message` uses this to sit the CTA
    button just above the message banner's top edge instead of
    overlapping it (the button is always drawn last/on top, so without
    this it would render right on top of the banner's own text).

    Mirrors _add_edge_banner()'s edge="bottom" sizing math (the same
    call add_message_banner() itself makes: max_height_frac=0.4, no
    reserved_left/reserved_right -- the message banner never reserves
    space for a logo, only the header does). Kept as a separate,
    lightweight computation rather than a refactor of _add_edge_banner()
    so this can't change that function's tested behavior; the two are
    cross-checked directly in tests -- keep them in sync if
    _add_edge_banner()'s bottom-edge sizing math ever changes.
    """
    w, h = image_size
    probe_img = Image.new("RGBA", (max(w, 1), max(h, 1)))
    draw_probe = ImageDraw.Draw(probe_img)
    padding = max(int(w * 0.05), 4)
    max_text_width = max(w - 2 * padding, 10)
    max_height_frac = 0.4
    max_banner_height = max(int(h * max_height_frac), 12)
    if font_size is not None:
        font = _load_font(font_size)
        lines = wrap_text_to_width(draw_probe, message, font, max_text_width)
        line_height = draw_probe.textbbox((0, 0), "Ag", font=font)[3] + int(font_size * 0.3)
        banner_height_cap = h
    else:
        min_font_size = max(min(int(h * 0.18), 14), 7)
        font, lines, line_height = fit_text_block(
            draw_probe,
            message,
            max_text_width,
            max(max_banner_height - int(h * 0.06), 8),
            min_font_size=min_font_size,
        )
        banner_height_cap = max_banner_height
    text_block_height = line_height * len(lines)
    return min(max(text_block_height + int(h * 0.06), int(h * 0.14)), banner_height_cap)


def add_message_banner(
    image: Image.Image,
    message: str,
    *,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    show_background: bool = True,
    glow: bool = False,
    glow_color: Tuple[int, int, int] = (255, 255, 255),
    align: str = "left",
    font_size: Optional[int] = None,
    _rgba: bool = False,
) -> Image.Image:
    """Overlay a banner with the (localized) campaign message.

    Left-aligned by default along the bottom edge -- reads like a
    caption/CTA line -- but `align` ("left"/"center"/"right") can override
    that. See _add_edge_banner() for what `text_color`/`show_background`/
    `glow`/`glow_color`/`font_size`/`_rgba` do.
    """
    return _add_edge_banner(
        image,
        message,
        edge="bottom",
        max_height_frac=0.4,
        align=align,
        text_color=text_color,
        show_background=show_background,
        glow=glow,
        glow_color=glow_color,
        font_size=font_size,
        _rgba=_rgba,
    )


def add_header_banner(
    image: Image.Image,
    headline: str,
    reserved_right: int = 0,
    *,
    reserved_left: int = 0,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    show_background: bool = True,
    glow: bool = False,
    glow_color: Tuple[int, int, int] = (255, 255, 255),
    align: str = "center",
    font_size: Optional[int] = None,
    _rgba: bool = False,
) -> Image.Image:
    """Overlay a banner with a title/headline.

    Centered by default along the top edge -- reads like an ad's headline,
    distinct from the bottom message/CTA banner -- but `align`
    ("left"/"center"/"right") can override that. Capped at a smaller
    fraction of the frame than the message banner since headline text (a
    product name or short tagline) is typically much shorter than the full
    campaign message.

    `reserved_right`/`reserved_left`: pixels to keep clear along the right
    or left edge so a long headline doesn't run underneath the brand logo --
    pass logo_render_size()'s reserved width (via `reserved_right` for a
    top-right logo, `reserved_left` for a top-left one) when a logo will
    also be composited on top of this same image. See _add_edge_banner()
    for what `text_color`/`show_background`/`glow`/`glow_color`/`font_size`
    do.
    """
    return _add_edge_banner(
        image,
        headline,
        edge="top",
        max_height_frac=0.22,
        align=align,
        reserved_right=reserved_right,
        reserved_left=reserved_left,
        text_color=text_color,
        show_background=show_background,
        glow=glow,
        glow_color=glow_color,
        font_size=font_size,
        _rgba=_rgba,
    )


def header_banner_height(
    image_size: Tuple[int, int],
    headline: str,
    *,
    font_size: Optional[int] = None,
    reserved_left: int = 0,
    reserved_right: int = 0,
) -> int:
    """The pixel height add_header_banner() will use to render `headline`
    at `image_size`, computed without actually drawing anything.

    Mirrors message_banner_height() but for the top edge (max_height_frac
    matches add_header_banner()'s own 0.22, and reserved_left/
    reserved_right work the same as they do there). Used by
    render_creative()'s "below-header" logo position to know how far down
    the header actually extends, so the logo can sit just under it
    instead of a fixed guess. Keep in sync with _add_edge_banner()'s
    top-edge sizing math if that ever changes -- cross-checked directly
    in tests.
    """
    w, h = image_size
    probe_img = Image.new("RGBA", (max(w, 1), max(h, 1)))
    draw_probe = ImageDraw.Draw(probe_img)
    padding = max(int(w * 0.05), 4)
    min_usable_w = int(w * 0.3)
    left_bound = min(reserved_left, max(w - min_usable_w, 0))
    right_bound = max(w - reserved_right, left_bound + min_usable_w)
    usable_w = right_bound - left_bound
    max_text_width = max(usable_w - 2 * padding, 10)
    max_height_frac = 0.22
    max_banner_height = max(int(h * max_height_frac), 12)
    if font_size is not None:
        font = _load_font(font_size)
        lines = wrap_text_to_width(draw_probe, headline, font, max_text_width)
        line_height = draw_probe.textbbox((0, 0), "Ag", font=font)[3] + int(font_size * 0.3)
        banner_height_cap = h
    else:
        min_font_size = max(min(int(h * 0.18), 14), 7)
        font, lines, line_height = fit_text_block(
            draw_probe,
            headline,
            max_text_width,
            max(max_banner_height - int(h * 0.06), 8),
            min_font_size=min_font_size,
        )
        banner_height_cap = max_banner_height
    text_block_height = line_height * len(lines)
    return min(max(text_block_height + int(h * 0.06), int(h * 0.14)), banner_height_cap)


# A brand logo defaults to a small top-right watermark (and always has,
# for backward compatibility), but can be placed at any corner, dead
# center, or just underneath the header banner instead -- unlike the
# badge image, there's no "full" option, since a logo stretched to fill
# the frame stops reading as a watermark.
VALID_LOGO_POSITIONS = (
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
    "center",
    "below-header-left",
    "below-header-center",
    "below-header-right",
)


def logo_render_size(
    canvas_size: Tuple[int, int],
    logo: Image.Image,
    scale_frac: float = 0.16,
    margin_frac: float = 0.04,
    height_cap_frac: float = 0.6,
) -> Tuple[int, int, int]:
    """Compute the on-canvas (width, height, margin) a logo will render at.

    Shared by add_logo_watermark() (to actually place it) and callers that
    need to know its footprint ahead of time -- e.g. reserving space so a
    header headline doesn't run underneath it.
    """
    w, h = canvas_size
    # Scale relative to the *shorter* side, capped by an absolute fraction
    # of height too, so the logo doesn't dwarf a very short canvas (e.g. a
    # 728x90 leaderboard or 320x50 mobile banner) the way sizing purely off
    # width would.
    scale_frac = max(min(scale_frac, 1.0), 0.01)
    logo_w = max(int(min(w, h * 1.6) * scale_frac), 12)
    ratio = logo_w / logo.width
    logo_h = int(logo.height * ratio)
    if logo_h > h * height_cap_frac:
        ratio = (h * height_cap_frac) / logo.height
        logo_w = int(logo.width * ratio)
        logo_h = int(logo.height * ratio)
    margin = max(int(w * margin_frac), 2)
    return logo_w, logo_h, margin


def add_logo_watermark(
    image: Image.Image,
    logo: Image.Image,
    *,
    position: str = "top-right",
    scale: float = 0.16,
    opacity: float = 1.0,
    margin_frac: float = 0.04,
    x_offset: int = 0,
    y_offset: int = 0,
    _rgba: bool = False,
) -> Image.Image:
    """Composite a brand logo onto `image` (used for the brand-compliance check too).

    `position` is one of VALID_LOGO_POSITIONS, defaulting to "top-right" --
    the original, only-ever-supported placement. `scale` and `opacity` work
    the same way they do for add_badge_image(): `scale` is the logo's size
    as a fraction of the frame (same idea as `logo_render_size()`'s default
    0.16), and `opacity` (0.0-1.0) scales the logo's own alpha before
    compositing, for a subtler watermark look.

    The three "below-header-*" positions start from the same top-row spot
    as "top-left"/"top-right"/"below-header-center" (left-aligned,
    horizontally centered, or right-aligned, respectively, all `margin`
    from the top edge) -- `y_offset` (see below) is what actually pushes
    them down clear of the header banner; with no offset they just sit at
    the top like the corresponding top position.

    `x_offset`/`y_offset` nudge the logo right/down (negative for
    left/up) by that many pixels from its normal computed position, on
    top of whatever `position` already resolves to. Clamped so the logo
    can never be pushed above or left of the frame's own edge (a large
    negative nudge just stops at 0, it doesn't wrap or go off-canvas the
    other way); there's no clamp against the bottom/right edge, so a
    large positive nudge can push the logo partly off-frame -- the same
    trade-off `y_offset` already made for render_creative()'s
    "below-header-*" positions, which is what actually clears the
    rendered header banner height there. Both default to 0 (no shift)
    and are harmless for every position if ever passed.
    """
    if position not in VALID_LOGO_POSITIONS:
        raise ValueError(f"position must be one of {VALID_LOGO_POSITIONS}, got {position!r}")
    opacity = max(0.0, min(opacity, 1.0))

    img = image.convert("RGBA")
    w, h = img.size
    logo_w, logo_h, margin = logo_render_size((w, h), logo, scale_frac=scale, margin_frac=margin_frac)
    # LANCZOS explicitly, like every other resize here. The default
    # filter is softer, and a brand mark scaled down without it is the
    # one place that shows.
    logo_resized = logo.convert("RGBA").resize(
        (max(logo_w, 1), max(logo_h, 1)), Image.LANCZOS
    )

    if position == "top-left":
        x, y = margin, margin
    elif position == "top-right":
        x, y = w - logo_resized.width - margin, margin
    elif position == "bottom-left":
        x, y = margin, h - logo_resized.height - margin
    elif position == "bottom-right":
        x, y = w - logo_resized.width - margin, h - logo_resized.height - margin
    elif position == "below-header-left":
        x, y = margin, margin
    elif position == "below-header-center":
        x, y = (w - logo_resized.width) // 2, margin
    elif position == "below-header-right":
        x, y = w - logo_resized.width - margin, margin
    else:  # center
        x, y = (w - logo_resized.width) // 2, (h - logo_resized.height) // 2
    x += x_offset
    y += y_offset
    x, y = max(int(x), 0), max(int(y), 0)

    if opacity < 1.0:
        alpha = logo_resized.split()[3].point(lambda a: int(a * opacity))
        logo_resized = logo_resized.copy()
        logo_resized.putalpha(alpha)

    img.paste(logo_resized, (x, y), logo_resized)
    return img if _rgba else img.convert("RGB")


# A second, independent image slot -- distinct from the brand logo above.
# The logo is always a small top-right watermark tied into the
# brand-compliance check; this is a general-purpose badge image for anything
# else: a "Sale" sticker or seasonal seal positioned at a corner/center, or
# a full-frame graphic (a tint, gradient, texture, or decorative frame)
# stretched across the whole creative.
VALID_BADGE_POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right", "center", "full")


def badge_render_size(
    canvas_size: Tuple[int, int],
    badge_image: Image.Image,
    scale_frac: float = 0.35,
    margin_frac: float = 0.04,
) -> Tuple[int, int, int]:
    """Compute the on-canvas (width, height, margin) a corner/center badge
    will render at, given as a fraction of the frame -- same idea as
    logo_render_size(), but with a caller-controlled scale instead of a
    fixed 0.16. Not used for the "full" position, which always stretches to
    the exact canvas size regardless of scale.
    """
    w, h = canvas_size
    scale_frac = max(min(scale_frac, 1.0), 0.01)
    target_w = max(int(min(w, h * 1.6) * scale_frac), 8)
    ratio = target_w / badge_image.width
    target_h = int(badge_image.height * ratio)
    if target_h > h * 0.9:
        ratio = (h * 0.9) / badge_image.height
        target_w = int(badge_image.width * ratio)
        target_h = int(badge_image.height * ratio)
    margin = max(int(w * margin_frac), 2)
    return target_w, target_h, margin


def add_badge_image(
    image: Image.Image,
    badge_image: Image.Image,
    *,
    position: str = "top-right",
    scale: float = 0.35,
    opacity: float = 1.0,
    margin_frac: float = 0.04,
    _rgba: bool = False,
) -> Image.Image:
    """Composite a secondary badge image onto `image`.

    `position` is one of VALID_BADGE_POSITIONS:
      - "top-left"/"top-right"/"bottom-left"/"bottom-right"/"center": the
        badge is sized to `scale` (a fraction of the frame, same idea as
        the brand logo) and placed at that corner/center with a margin --
        for a badge, sticker, or seal that should read as sitting on top of
        the creative.
      - "full": the badge is stretched to exactly cover the whole frame,
        ignoring `scale` and `margin_frac` entirely -- for a tint,
        gradient, texture, or decorative frame graphic meant to sit behind
        the header/message text rather than as a standalone badge.

    `opacity` (0.0-1.0) scales the badge's own alpha before compositing,
    so a fully opaque source image can still be blended in subtly (e.g. as
    a soft tint) without needing a pre-baked semi-transparent asset.
    """
    if position not in VALID_BADGE_POSITIONS:
        raise ValueError(f"position must be one of {VALID_BADGE_POSITIONS}, got {position!r}")
    opacity = max(0.0, min(opacity, 1.0))

    img = image.convert("RGBA")
    w, h = img.size
    badge_rgba = badge_image.convert("RGBA")

    if position == "full":
        resized = badge_rgba.resize((max(w, 1), max(h, 1)), Image.LANCZOS)
        x, y = 0, 0
    else:
        target_w, target_h, margin = badge_render_size((w, h), badge_rgba, scale_frac=scale, margin_frac=margin_frac)
        resized = badge_rgba.resize((max(target_w, 1), max(target_h, 1)), Image.LANCZOS)
        if position == "top-left":
            x, y = margin, margin
        elif position == "top-right":
            x, y = w - resized.width - margin, margin
        elif position == "bottom-left":
            x, y = margin, h - resized.height - margin
        elif position == "bottom-right":
            x, y = w - resized.width - margin, h - resized.height - margin
        else:  # center
            x, y = (w - resized.width) // 2, (h - resized.height) // 2
        x, y = max(x, 0), max(y, 0)

    if opacity < 1.0:
        alpha = resized.split()[3].point(lambda a: int(a * opacity))
        resized = resized.copy()
        resized.putalpha(alpha)

    img.paste(resized, (int(x), int(y)), resized)
    return img if _rgba else img.convert("RGB")


# A call-to-action button -- distinct from the header/message banners
# (full-width text bands) and the badge image (an arbitrary picture):
# this is a small, filled, pill-shaped button with its own short label
# (e.g. "Shop Now", "Learn More"), meant to read as clickable/actionable
# rather than as body copy. "bottom-center" is included alongside the
# usual four corners and center since it's the most common real-world CTA
# placement (a "sticky" action bar along the bottom edge).
VALID_CTA_POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right", "center", "bottom-center")


def add_cta_button(
    image: Image.Image,
    text: str,
    *,
    position: str = "bottom-center",
    button_color: Tuple[int, int, int] = (0, 87, 184),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    font_size: Optional[int] = None,
    font_family: str = "sans",
    glow: bool = False,
    glow_color: Tuple[int, int, int] = (255, 255, 255),
    margin_frac: float = 0.05,
    y_offset: int = 0,
    _rgba: bool = False,
) -> Image.Image:
    """Composite a filled, pill-shaped call-to-action button with `text`.

    Always renders on top of every other overlay -- the header/message
    banners, the brand logo, and the badge image -- since a CTA needs to
    stay visible/actionable regardless of what else is on the creative.

    `position` is one of VALID_CTA_POSITIONS. `font_size`, if omitted, is
    chosen automatically relative to the frame (similar to the brand logo);
    the button also auto-shrinks its font if the requested text would
    otherwise make the button wider than the frame allows, since a CTA
    label is meant to stay on one line rather than wrap. `button_color`/
    `text_color` default to a brand-blue-on-white look, independent of any
    other color choice on the creative. `font_family` is one of
    VALID_FONT_FAMILIES ("sans" by default, matching every other piece of
    text on the creative).

    `glow`/`glow_color` add a soft colored halo *around the whole button
    shape* (not just its text) -- a blurred, colorized copy of the pill
    outline, composited underneath the crisp button -- for a "this button
    is lit up" emphasis effect. Off by default.

    `y_offset` shifts the button up by that many pixels from its normal
    computed position (never below 0) -- used by render_creative()'s
    `cta_above_message` to lift a bottom-positioned button clear of the
    message banner instead of overlapping it. 0 (no shift) by default.
    """
    if position not in VALID_CTA_POSITIONS:
        raise ValueError(f"position must be one of {VALID_CTA_POSITIONS}, got {position!r}")
    if font_family not in VALID_FONT_FAMILIES:
        raise ValueError(f"font_family must be one of {VALID_FONT_FAMILIES}, got {font_family!r}")
    if not text:
        return image.convert("RGBA") if _rgba else image.convert("RGB")

    img = image.convert("RGBA")
    w, h = img.size
    draw_probe = ImageDraw.Draw(img)

    if font_size is None:
        font_size = max(min(int(min(w, h) * 0.06), 72), 12)
    margin = max(int(min(w, h) * margin_frac), 6)
    max_button_width = max(w - 2 * margin, 20)

    # A CTA label is meant to stay on one line -- if it would make the
    # button wider than the frame allows, shrink the font to fit rather
    # than wrapping or overflowing. Repeatedly rescale (a single pass can
    # undershoot when the padding itself shrinks along with the font) down
    # to a small absolute legibility floor, so even a long label on a
    # narrow frame (e.g. a 160px-wide skyscraper) ends up fitting rather
    # than spilling past the canvas edge.
    min_font_size = 6
    font = _load_font(font_size, family=font_family)
    for _ in range(6):
        bbox = draw_probe.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        pad_x = max(int(font_size * 0.9), 6)
        if text_w + 2 * pad_x <= max_button_width or font_size <= min_font_size:
            break
        scale = max(max_button_width - 2 * pad_x, 10) / max(text_w, 1)
        new_font_size = max(int(font_size * scale), min_font_size)
        if new_font_size >= font_size:
            font_size = min_font_size
        else:
            font_size = new_font_size
        font = _load_font(font_size, family=font_family)
    bbox = draw_probe.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    pad_x = max(int(font_size * 0.9), 6)

    # Even at the smallest legible size, an unusually long label on a very
    # narrow frame can still be too wide -- truncate with an ellipsis as a
    # last resort so the button never spills past the canvas edge.
    if text_w + 2 * pad_x > max_button_width:
        truncated = text
        while len(truncated) > 1:
            truncated = truncated[:-1]
            candidate = truncated.rstrip() + "…"
            bbox = draw_probe.textbbox((0, 0), candidate, font=font)
            text_w = bbox[2] - bbox[0]
            if text_w + 2 * pad_x <= max_button_width:
                text = candidate
                break
        else:
            text = "…"
            bbox = draw_probe.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]

    text_h = bbox[3] - bbox[1]
    pad_y = max(int(font_size * 0.55), 6)
    button_w = int(text_w) + 2 * pad_x
    button_h = int(text_h) + 2 * pad_y
    radius = button_h // 2  # pill shape

    if position == "top-left":
        bx, by = margin, margin
    elif position == "top-right":
        bx, by = w - button_w - margin, margin
    elif position == "bottom-left":
        bx, by = margin, h - button_h - margin
    elif position == "bottom-right":
        bx, by = w - button_w - margin, h - button_h - margin
    elif position == "center":
        bx, by = (w - button_w) // 2, (h - button_h) // 2
    else:  # bottom-center
        bx, by = (w - button_w) // 2, h - button_h - margin
    by -= y_offset
    bx, by = max(int(bx), 0), max(int(by), 0)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if glow:
        # Same technique as the header/message text glow: render the
        # button's pill shape as a plain white mask on its own layer, blur
        # its alpha into a soft halo, boost the blurred alpha back up
        # (blurring dims it substantially), tint it with `glow_color`, and
        # composite that behind the crisp button -- so the halo reads as
        # light spilling out from around the button rather than washing
        # over it.
        glow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_layer)
        glow_draw.rounded_rectangle(
            [bx, by, bx + button_w, by + button_h],
            radius=radius,
            fill=(255, 255, 255, 255),
        )
        blur_radius = max(round(button_h * 0.25), 4)
        alpha = glow_layer.split()[3].filter(ImageFilter.GaussianBlur(radius=blur_radius))
        alpha = alpha.point(lambda a: min(255, int(a * 1.6)))
        colored_glow = Image.new("RGBA", img.size, (glow_color[0], glow_color[1], glow_color[2], 0))
        colored_glow.putalpha(alpha)

        overlay = Image.alpha_composite(overlay, colored_glow)
        draw = ImageDraw.Draw(overlay)  # alpha_composite() returns a new Image -- rebind the Draw handle to it

    draw.rounded_rectangle(
        [bx, by, bx + button_w, by + button_h],
        radius=radius,
        fill=(button_color[0], button_color[1], button_color[2], 255),
    )
    text_x = bx + pad_x - bbox[0]
    text_y = by + pad_y - bbox[1]
    draw.text((text_x, text_y), text, font=font, fill=(text_color[0], text_color[1], text_color[2], 255))

    combined = Image.alpha_composite(img, overlay)
    return combined if _rgba else combined.convert("RGB")


# ---------------------------------------------------------------------------
# PSD layer-region overrides
#
# Reading is done with psd-tools, not Pillow's own (much simpler) PSD
# parser -- Pillow's reader silently drops any layer with more than 4
# channels, which includes any ordinary layer with a ordinary Photoshop
# layer mask attached (ct_types > 4 in PIL/PsdImagePlugin.py's
# _layerinfo()) -- a completely routine thing for a real, hand-authored
# template to have, and something src/psd_export.py's own downloads can
# trigger too if the parent document isn't RGBA. A masked layer isn't an
# edge case here, it's common enough that treating it as "missing" broke
# real uploads -- see get_psd_layer_background()'s docstring for the same
# limitation, already worked around there the same way.
#
# psd-tools gives reliable layer *names* and *bounding boxes* for every
# top-level layer, but does not reliably decode arbitrary layers' actual
# pixel content (text layers, smart objects, and layers with layer
# effects/blend modes are especially unreliable to extract cleanly -- the
# same limitation that made the earlier layer-role extraction system
# fragile enough to remove entirely). So "updating a layer" here doesn't
# mean editing the PSD itself -- it means compositing new content directly
# onto the already-flattened composite image, positioned at that named
# layer's original bounding box. Good enough for swapping a logo/CTA
# image/product image (paste, alpha-composited, contain-fit, centered) or
# a caption (paint over with a sampled fill, then draw fresh autofit text)
# without touching anything else in the design.
# ---------------------------------------------------------------------------


def get_psd_layer_boxes(psd_path: Union[str, Path]) -> dict:
    """Return {lowercased layer name: clipped (x0, y0, x1, y1)} for every
    top-level layer in the PSD at `psd_path`, clipped to the canvas bounds
    (a layer's raw bbox can extend past the canvas edge, e.g. a background
    layer painted larger than the frame for bleed).

    A layer whose bbox is entirely outside the canvas, or that has no
    pixel content at all (an empty layer that was never painted into or
    resized in Photoshop can end up with a zero-area bbox), is skipped --
    there's no usable region to report for it, and callers should treat a
    name missing from the result the same as it not being in the file at
    all.

    Returns {} if `psd-tools` isn't installed, the file can't be opened,
    or it isn't a layered PSD (e.g. a flat image saved with a .psd
    extension) -- callers should treat that as "no layer overrides
    available for this size" rather than an error, so one template
    missing a layer never blocks the others.
    """
    try:
        from psd_tools import PSDImage
    except ImportError:
        return {}
    try:
        psd = PSDImage.open(psd_path)
    except Exception:
        return {}

    canvas_w, canvas_h = psd.size
    boxes: dict = {}
    for layer in psd:
        name = (layer.name or "").strip()
        if not name:
            continue
        try:
            bbox = layer.bbox
        except Exception:
            continue
        if not bbox:
            continue
        x0, y0, x1, y1 = bbox
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, canvas_w), min(y1, canvas_h)
        if x1 <= x0 or y1 <= y0:
            continue
        boxes[name.lower()] = (x0, y0, x1, y1)
    return boxes


def get_psd_group_text(psd_path: Union[str, Path], group_name: str):
    """The words on the type layer inside the named group, or None.

    get_psd_text_layers() reads top-level layers, so a CTA built as a
    group -- rectangle plus label -- looks textless to it. Restyling the
    button has to redraw the label over the new shape, and this is where
    the words come from when the user changed a colour without retyping
    them.
    """
    try:
        from psd_tools import PSDImage
    except ImportError:
        return None
    try:
        psd = PSDImage.open(psd_path)
    except Exception:
        return None
    wanted = group_name.strip().lower()
    for layer in psd:
        if (layer.name or "").strip().lower() != wanted or not layer.is_group():
            continue
        for child in layer:
            if getattr(child, "kind", None) == "type":
                try:
                    text = (child.text or "").replace("\r", " ").strip()
                except Exception:
                    return None
                return text or None
    return None


def get_psd_group_text_box(psd_path: Union[str, Path], group_name: str):
    """The bounding box of the first type layer INSIDE the named group,
    or None.

    A CTA built as a group -- a rounded rectangle with its label sitting
    on top -- is one layer to everything that reads top-level boxes, so
    replacing the CTA meant painting over the designer's button and
    drawing a generic pill in its place. The label's own box is what
    makes it possible to change the words and keep the button.

    Returns None when psd-tools isn't installed, the file can't be read,
    the group isn't there, it isn't a group, or it holds no type layer --
    every one of which is a caller's cue to fall back to the whole-box
    behaviour.
    """
    try:
        from psd_tools import PSDImage
    except ImportError:
        return None
    try:
        psd = PSDImage.open(psd_path)
    except Exception:
        return None
    wanted = group_name.strip().lower()
    for layer in psd:
        if (layer.name or "").strip().lower() != wanted:
            continue
        if not layer.is_group():
            return None
        for child in layer:
            if getattr(child, "kind", None) == "type":
                box = child.bbox
                if box and box[2] > box[0] and box[3] > box[1]:
                    return tuple(box)
        return None
    return None


def get_psd_layer_names(psd_path: Union[str, Path]) -> set:
    """Lowercased names of every top-level layer in a PSD, drawn or not.

    get_psd_visible_layers() answers "what does this file draw"; this
    answers "what does it contain". The difference between the two is
    the set of layers a designer has switched off, which is worth naming
    to whoever is wondering where their product shot went.

    A switched-off GROUP is why this exists rather than reading the keys
    of get_psd_layer_boxes(): a hidden group reports an empty bounding
    box, so it drops out of the box map entirely and a CTA that had been
    turned off looked, from there, like a template that never had one.

    Empty set if psd-tools isn't installed or the file can't be read.
    """
    try:
        from psd_tools import PSDImage
    except ImportError:
        return set()
    try:
        psd = PSDImage.open(psd_path)
    except Exception:  # noqa: BLE001
        return set()
    return {
        (layer.name or "").strip().lower()
        for layer in psd
        if (layer.name or "").strip()
    }


def layers_under_background(psd) -> set:
    """Lowercased names of the top-level layers that sit BELOW the
    `background` layer in an open psd-tools document.

    Photoshop draws the stack bottom to top, so anything under an
    opaque, canvas-filling background is covered and never shows.
    Treating those layers as visible -- because their eye icon is on
    -- was drawing them: a description dragged under the background
    in the Layers panel came back on top of the render, and its text
    override was applied to a layer the designer had buried. They are
    hidden layers in every sense that matters here.
    """
    names = [(layer.name or "").strip().lower() for layer in psd]
    if "background" not in names:
        return set()
    index = names.index("background")
    background = list(psd)[index]
    if not background.visible:
        return set()
    return {n for n in names[:index] if n}


def get_psd_buried_layers(psd_path: Union[str, Path]) -> set:
    """layers_under_background() for a file on disk; empty when it can't
    be read."""
    try:
        from psd_tools import PSDImage

        return layers_under_background(PSDImage.open(psd_path))
    except Exception:  # noqa: BLE001
        return set()


def get_psd_visible_layers(psd_path: Union[str, Path]) -> set:
    """Lowercased names of the top-level layers a PSD actually draws --
    the ones switched ON in Photoshop.

    Used to tell "hide this layer" from "this layer is already hidden".
    They are not the same instruction: hiding wipes the layer's whole box
    back to the backdrop, and a box routinely overlaps its neighbours (a
    header banner drawn across the logo, a legal line running under the
    CTA). Wiping the box of a layer that was never drawn removes the
    neighbours and nothing else.

    Returns an empty set if psd-tools isn't installed or the file can't
    be read -- callers should treat that as "can't tell" and fall back to
    their previous behaviour.
    """
    try:
        from psd_tools import PSDImage
    except ImportError:
        return set()
    try:
        psd = PSDImage.open(psd_path)
    except Exception:
        return set()
    buried = layers_under_background(psd)
    return {
        (layer.name or "").strip().lower()
        for layer in psd
        if (layer.name or "").strip() and layer.visible
        and (layer.name or "").strip().lower() not in buried
    }


def get_psd_layer_stack(psd_path: Union[str, Path]) -> Optional[List[Tuple[str, Image.Image]]]:
    """Return every top-level layer in `psd_path` as its own canvas-sized
    RGBA image (transparent everywhere except where that one layer
    itself draws, its own real alpha preserved), in the same back-to-
    front order psd-tools stores them in -- ready to feed straight into
    save_layered_psd() to rebuild an equivalent layered PSD.

    Unlike get_psd_layer_boxes() (a bounding box per layer) or
    get_psd_layer_background()/get_psd_layer_foreground() (one flattened
    composite with a layer hidden), this keeps every layer as its own
    separate image -- so a caller can swap out one or two entries (e.g.
    with a freshly-overridden layer's own isolated RGBA patch -- see
    apply_layer_image_override()'s `keep_alpha`) and rebuild a real,
    still-editable PSD where every OTHER layer is untouched, instead of
    collapsing everything down to one flattened layer.

    Each layer's own `layer.composite()` isn't used here -- confirmed
    directly against this project's own templates, psd-tools renders a
    text (type) layer composited on its own as fully blank (transparent),
    even though that exact same layer renders correctly as *part of* a
    whole-document composite (see get_psd_layer_background()'s "hide one
    layer" trick, which already depends on that working). So each layer
    is isolated the same way: every OTHER layer's visibility is toggled
    off, one at a time, and psd.composite() is called for the whole
    document -- already canvas-sized and correctly positioned, text
    included, with everything else transparent.

    Returns None if `psd-tools` isn't installed, the file can't be
    opened, or it isn't a layered PSD -- callers should treat that as
    "no layer stack available" and fall back to their own handling (e.g.
    a single flattened layer).
    """
    try:
        from psd_tools import PSDImage
    except ImportError:
        return None
    try:
        psd = PSDImage.open(psd_path)
    except Exception:
        return None

    layers = [layer for layer in psd if (layer.name or "").strip()]
    original_visibility = [(layer, layer.visible) for layer in layers]
    stack: List[Tuple[str, Image.Image]] = []
    try:
        for target in layers:
            for layer in layers:
                layer.visible = layer is target
            try:
                composite = psd.composite()
            except ImportError as exc:
                # Same aggdraw dependency as
                # _psd_composite_with_layers_hidden() above. Skipped
                # silently, the layer simply vanishes from the stack --
                # and from the rebuilt PSD download built out of it.
                _log.error(
                    "Cannot isolate layer %r in %s -- %s. Install aggdraw "
                    "(pip install aggdraw) or this layer is dropped from the "
                    "exported PSD.", target.name, psd_path, exc,
                )
                continue
            except Exception:
                continue
            if composite is None:
                continue
            stack.append((target.name.strip(), composite.convert("RGBA")))
    finally:
        for layer, visible in original_visibility:
            layer.visible = visible
    return stack


def get_psd_text_layers(psd_path: Union[str, Path], visible_only: bool = False) -> dict:
    """Return {lowercased layer name: text content} for every top-level
    text (type) layer in the PSD at `psd_path` -- the actual words someone
    typed into that layer in Photoshop (e.g. a "description" layer baked
    straight into an uploaded template), not styling.

    Used by the web app's profanity check so flagged language embedded
    directly in an uploaded PSD's text layer gets caught the same as
    flagged language typed into one of the web form's own fields --
    otherwise that check would be trivially bypassable by putting the
    text in the PSD instead of the form.

    `visible_only=True` additionally skips layers switched off in
    Photoshop. The profanity check wants them -- text hidden in a file is
    still text someone shipped, and skipping it would make the check
    trivially bypassable. Anything that *renders* a layer's words wants
    the opposite: a layer the designer turned off must stay off, not get
    resurrected onto the creative.

    A layer with no text (an empty type layer) is skipped. Returns {} if
    `psd-tools` isn't installed, the file can't be opened, or it isn't a
    layered PSD -- same "nothing usable here" convention as
    get_psd_layer_boxes() above, so callers can treat a missing/unreadable
    file the same as one with no text layers at all.
    """
    try:
        from psd_tools import PSDImage
    except ImportError:
        return {}
    try:
        psd = PSDImage.open(psd_path)
    except Exception:
        return {}

    texts: dict = {}
    buried = layers_under_background(psd) if visible_only else set()
    for layer in psd:
        name = (layer.name or "").strip()
        if not name or layer.kind != "type":
            continue
        if visible_only and (not layer.visible or name.lower() in buried):
            continue
        try:
            text = layer.text
        except Exception:
            continue
        if text:
            texts[name.lower()] = text
    return texts


def find_missing_brand_colors(
    image: Image.Image,
    colors: List[Tuple[int, int, int]],
    tolerance: int = 30,
) -> List[Tuple[int, int, int]]:
    """Return the subset of `colors` that don't appear ANYWHERE in
    `image`, as a simple brand-compliance sanity check ("is every brand
    color actually present in this creative?").

    A pixel counts as a match within a Euclidean RGB distance of
    `tolerance` rather than requiring an exact hit -- a brand color
    placed as a flat swatch still gets softened a little by resizing,
    JPEG-ish recompression, or sitting under a semi-transparent overlay,
    so an exact-match check would false-flag colors that are clearly
    "there" to a human looking at the image.

    Downsamples large images before comparing purely for speed --
    presence of a close-enough color somewhere in the image doesn't
    depend on exact resolution, and this keeps the distinct-color count
    (and therefore the comparison work) bounded regardless of how large
    the creative is.
    """
    if not colors:
        return []

    rgb = image.convert("RGB")
    max_dim = 300
    w, h = rgb.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        rgb = rgb.resize(
            (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)),
            Image.BILINEAR,
        )

    color_counts = rgb.getcolors(maxcolors=rgb.size[0] * rgb.size[1])
    if color_counts is None:
        # Shouldn't happen at this resolution (maxcolors is set to the
        # exact pixel count, so it can never be exceeded) -- fail open
        # rather than false-flagging every color as missing if it ever did.
        return []
    present_colors = [c for _count, c in color_counts]

    tolerance_sq = tolerance * tolerance
    missing = []
    for target in colors:
        tr, tg, tb = target
        found = any(
            (tr - r) ** 2 + (tg - g) ** 2 + (tb - b) ** 2 <= tolerance_sq
            for r, g, b in present_colors
        )
        if not found:
            missing.append(target)
    return missing


def get_psd_canvas_size(psd_path: Union[str, Path]) -> Optional[Tuple[int, int]]:
    """Return the PSD file's own saved (width, height) in pixels, straight
    from Image.open() -- the same call open_as_rgb() and
    get_psd_layer_boxes() both use, so this always matches the canvas
    every layer box in get_psd_layer_boxes() is measured against.

    Returns None if the file can't be opened. Used to detect when a PSD's
    actual saved dimensions don't exactly match the nominal size bucket
    it's being used for (e.g. a 728x480 upload used for the "720x480"
    slot) -- see map_box_through_fit().
    """
    try:
        with Image.open(psd_path) as img:
            return img.size
    except Exception:
        return None


def map_box_through_fit(
    box: Tuple[int, int, int, int],
    src_size: Tuple[int, int],
    target_size: Tuple[int, int],
    fit_mode: str,
) -> Tuple[int, int, int, int]:
    """Map an (x0, y0, x1, y1) box measured in a PSD's own native pixel
    space (`src_size`, from get_psd_canvas_size()) into the coordinate
    space `final_image` actually ends up in after being fit to
    `target_size` via resize_to_contain()/center_crop_to_ratio().

    A template PSD's saved canvas size is *usually* identical to its
    nominal size bucket (a "tester-1080x1080.psd" that's actually
    1080x1080px), in which case this is a no-op and every layer box lines
    up perfectly with the rendered creative. But nothing enforces that --
    a user-uploaded PSD assigned to a size slot by its filename (see
    webapp.py's size-matching) can have a canvas that's off by a handful
    of pixels (e.g. 728x480 used for the "720x480" slot). Without this
    mapping, layer boxes read in the PSD's own pixel space silently drift
    from where that content actually lands in the resized/cropped final
    image -- small enough to be easy to miss in review, but big enough
    that a background patch can miss a layer's true edge by a few pixels,
    leaving a sliver of the PSD's original content (e.g. placeholder
    text) visible right at a box's edge.

    Mirrors resize_to_contain()'s and center_crop_to_ratio()'s own
    arithmetic exactly (same scale/offset formulas) so a mapped box lines
    up with what those functions actually produce, not just an
    approximation of it.
    """
    src_w, src_h = src_size
    target_w, target_h = target_size
    if (src_w, src_h) == (target_w, target_h) or src_w <= 0 or src_h <= 0:
        return box
    x0, y0, x1, y1 = box

    if fit_mode == "contain":
        # Mirrors resize_to_contain(): uniform scale to fit entirely
        # within target, centered (letterboxed) on whichever axis has
        # slack.
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(int(round(src_w * scale)), 1)
        new_h = max(int(round(src_h * scale)), 1)
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        mapped = (
            offset_x + x0 * scale,
            offset_y + y0 * scale,
            offset_x + x1 * scale,
            offset_y + y1 * scale,
        )
    else:
        # Mirrors center_crop_to_ratio(): crop to the target aspect ratio
        # from the center, then resize that crop to exactly target_size.
        target_ratio = target_w / target_h
        src_ratio = src_w / src_h
        if src_ratio > target_ratio:
            crop_w = int(src_h * target_ratio)
            crop_h = src_h
            crop_x0 = (src_w - crop_w) // 2
            crop_y0 = 0
        else:
            crop_h = int(src_w / target_ratio)
            crop_w = src_w
            crop_x0 = 0
            crop_y0 = (src_h - crop_h) // 2
        scale_x = target_w / crop_w if crop_w else 1.0
        scale_y = target_h / crop_h if crop_h else 1.0
        mapped = (
            (x0 - crop_x0) * scale_x,
            (y0 - crop_y0) * scale_y,
            (x1 - crop_x0) * scale_x,
            (y1 - crop_y0) * scale_y,
        )

    mx0, my0, mx1, my1 = mapped
    mx0, my0 = max(int(round(mx0)), 0), max(int(round(my0)), 0)
    mx1, my1 = min(int(round(mx1)), target_w), min(int(round(my1)), target_h)
    if mx1 <= mx0:
        mx1 = min(mx0 + 1, target_w)
    if my1 <= my0:
        my1 = min(my0 + 1, target_h)
    return (mx0, my0, mx1, my1)


def _psd_composite_with_layers_hidden(
    psd_path: Union[str, Path], layer_names: Union[str, Iterable[str]], effects: bool = True
) -> Optional[Image.Image]:
    """Shared by get_psd_layer_background() and get_psd_layer_foreground()
    just below: open `psd_path` with psd-tools, hide every top-level
    layer whose name (case-insensitive) is in `layer_names` -- a single
    string or any iterable of strings -- and hand back psd-tools' own
    recomposite in whatever mode it returns (RGB or RGBA -- alpha is left
    intact when present, since one caller wants it and the other
    doesn't).

    Pillow's own PSD reader can't isolate individual layers (see
    `get_psd_layer_boxes()`'s docstring) -- every `img.seek(i)` on a
    PsdImageFile returns the identical fully-flattened composite
    regardless of `i`, confirmed directly against this project's own
    template files. The third-party `psd-tools` library reads PSDs more
    thoroughly and can toggle a layer's visibility and recomposite the
    document with it hidden, which is exactly "what was behind/around
    this layer" -- e.g. hiding a template's "logo" layer reveals whatever
    the ad's actual background (a gradient, a photo, a solid color) looks
    like at that spot, with no approximation involved. Hiding more than
    one layer at once is for a box that's being "re-cleaned" after the
    background itself was already swapped elsewhere in the same request
    (see get_psd_layer_foreground()'s docstring) -- hiding that layer
    *and* "background" together avoids reintroducing the old background's
    pixels into the patch.

    Returns None (rather than raising) if `psd-tools` isn't installed,
    the file can't be opened, or none of the names match any layer --
    callers should treat that as "no data available" and fall back to
    their own handling.
    """
    try:
        from psd_tools import PSDImage
    except ImportError:
        return None
    try:
        psd = PSDImage.open(psd_path)
    except Exception:
        return None

    if isinstance(layer_names, str):
        wanted = {layer_names.strip().lower()}
    else:
        wanted = {name.strip().lower() for name in layer_names}

    targets = [layer for layer in psd if layer.name.strip().lower() in wanted]
    if not targets:
        return None
    if "background" in wanted:
        # Hiding the background must not surface what it was covering:
        # a layer under it in the stack was never part of the picture.
        buried = layers_under_background(psd)
        targets += [
            layer for layer in psd
            if layer.name.strip().lower() in buried and layer not in targets
        ]

    original_visibility = [(layer, layer.visible) for layer in targets]
    try:
        for layer in targets:
            layer.visible = False
        composite = psd.composite()
    except ImportError as exc:
        # psd-tools can only draw a VECTOR SHAPE layer with aggdraw
        # installed, and it raises the moment a composite has to be
        # re-rendered rather than read from the file's cached preview --
        # which is every call here, since isolating a layer means
        # toggling visibility. A template whose CTA is a group holding a
        # rectangle has such a layer. Swallowed silently this returns
        # None, the caller reads that as "nothing to restore", and the
        # render quietly drops every layer this was meant to bring back.
        _log.error(
            "Cannot recomposite %s with layers hidden -- %s. Install aggdraw "
            "(pip install aggdraw) or the layers around a replaced background "
            "will be lost.", psd_path, exc,
        )
        return None
    except Exception:
        return None
    finally:
        for layer, visible in original_visibility:
            layer.visible = visible

    # psd-tools draws the layers but not their effects, so a design
    # recomposited around a new backdrop lost every drop shadow, glow
    # and stroke -- "the layer styles aren't working". Drawn here from
    # each visible layer's own alpha, under that layer.
    if effects:
        try:
            composite = _add_layer_effects(psd, composite, {layer for layer, _ in original_visibility})
        except Exception:  # noqa: BLE001
            pass
    return composite


def _layer_effect_images(layer_rgba: Image.Image, effects: dict) -> list:
    """RGBA images, canvas-sized like `layer_rgba`, for the layer's drop
    shadow, outer glow and stroke, in the order they go under it."""
    out = []
    alpha = layer_rgba.split()[3]
    shadow = effects.get("shadow")
    if shadow and (shadow.get("distance") or shadow.get("size")):
        angle = math.radians(float(shadow.get("angle", 120) or 0))
        distance = float(shadow.get("distance", 0) or 0)
        dx, dy = -distance * math.cos(angle), distance * math.sin(angle)
        moved = Image.new("L", alpha.size, 0)
        moved.paste(alpha, (int(round(dx)), int(round(dy))))
        size = float(shadow.get("size", 0) or 0)
        if size > 0:
            moved = moved.filter(ImageFilter.GaussianBlur(radius=max(size / 2.0, 0.5)))
        opacity = max(0.0, min(100.0, float(shadow.get("opacity", 75) or 0))) / 100.0
        moved = moved.point(lambda a: int(a * opacity))
        img = Image.new("RGBA", alpha.size, tuple(shadow.get("color") or (0, 0, 0)) + (0,))
        img.putalpha(moved)
        out.append(img)
    glow = effects.get("glow")
    if glow and glow.get("size"):
        size = float(glow["size"])
        grown = alpha.filter(ImageFilter.MaxFilter(max(3, int(size) // 2 * 2 + 1)))
        grown = grown.filter(ImageFilter.GaussianBlur(radius=max(size / 2.0, 0.5)))
        opacity = max(0.0, min(100.0, float(glow.get("opacity", 75) or 0))) / 100.0
        grown = grown.point(lambda a: int(min(255, a * 1.4) * opacity))
        img = Image.new("RGBA", alpha.size, tuple(glow.get("color") or (255, 255, 255)) + (0,))
        img.putalpha(grown)
        out.append(img)
    stroke = effects.get("stroke")
    if stroke and stroke.get("size"):
        size = max(1, int(round(float(stroke["size"]))))
        grown = alpha.filter(ImageFilter.MaxFilter(size * 2 + 1))
        img = Image.new("RGBA", alpha.size, tuple(stroke.get("color") or (0, 0, 0)) + (0,))
        img.putalpha(grown)
        out.append(img)
    return out


def _add_layer_effects(psd, composite: Image.Image, hidden_layers) -> Image.Image:
    """`composite` with every visible top-level layer's effects drawn
    under that layer. Layers above it end up under its shadow too --
    the small price of not re-doing psd-tools' whole composite."""
    hidden = set(hidden_layers)
    result = composite.convert("RGBA")
    changed = False
    for layer in psd:
        if layer in hidden or not layer.visible:
            continue
        try:
            effects = _psd_layer_effects(layer)
        except Exception:  # noqa: BLE001
            continue
        if not effects:
            continue
        try:
            pixels = layer.composite()
        except Exception:  # noqa: BLE001
            continue
        if pixels is None:
            continue
        # Work on a window around the layer, not the whole canvas: a 4K
        # document's full-frame blurs per layer cost seconds.
        pixels = pixels.convert("RGBA")
        x0, y0 = layer.bbox[0], layer.bbox[1]
        reach = 0.0
        for effect in effects.values():
            reach = max(reach, float(effect.get("distance", 0) or 0) + 2 * float(effect.get("size", 0) or 0))
        pad = int(math.ceil(reach)) + 2
        wx0, wy0 = max(0, x0 - pad), max(0, y0 - pad)
        wx1, wy1 = min(result.width, x0 + pixels.width + pad), min(result.height, y0 + pixels.height + pad)
        if wx1 <= wx0 or wy1 <= wy0:
            continue
        window = Image.new("RGBA", (wx1 - wx0, wy1 - wy0), (0, 0, 0, 0))
        window.paste(pixels, (x0 - wx0, y0 - wy0), pixels)
        region = result.crop((wx0, wy0, wx1, wy1))
        touched = False
        for effect_img in _layer_effect_images(window, effects):
            region = Image.alpha_composite(region, effect_img)
            touched = True
        if touched:
            region = Image.alpha_composite(region, window)
            result.paste(region, (wx0, wy0))
            changed = True
    return result if changed else composite


def get_psd_backdrop(
    psd_path: Union[str, Path], keep_layer_names: Iterable[str] = ("background",)
) -> Optional[Image.Image]:
    """Return the PSD composited down to just its backdrop -- every
    top-level layer hidden EXCEPT the ones named in `keep_layer_names`
    (case-insensitive, "background" by default) -- flattened to RGB.

    This is the "clear the whole box" counterpart to
    get_psd_layer_background() just below. That one hides a single named
    layer, which is exactly right when the layer being replaced is the
    only thing occupying its box. It's not enough when a text layer's box
    overlaps other artwork -- a header box sitting across the logo, say,
    which is common in a real template where the designer parked the
    headline over the brand mark. Hiding only "header" there leaves the
    logo in place and the new text lands on top of it, reading as text
    added to the design rather than text replacing it.

    Compositing everything away except the background gives the box's
    true backdrop -- the ad's own gradient or photo -- so a text override
    can wipe its whole box back to that and genuinely own the space.

    Returns None under the same conditions as
    _psd_composite_with_layers_hidden(), plus when the file has no layer
    left to keep (nothing named in `keep_layer_names` exists), since a
    composite of nothing isn't a usable backdrop.
    """
    try:
        from psd_tools import PSDImage
    except ImportError:
        return None
    try:
        psd = PSDImage.open(psd_path)
    except Exception:
        return None

    keep = {name.strip().lower() for name in keep_layer_names}
    names = [layer.name.strip().lower() for layer in psd]
    if not any(name in keep for name in names):
        return None
    hide = [name for name in names if name not in keep]
    if not hide:
        # Nothing to hide -- the document is already just its backdrop.
        try:
            return psd.composite().convert("RGB")
        except Exception:
            return None

    composite = _psd_composite_with_layers_hidden(psd_path, hide)
    if composite is None:
        return None
    return composite.convert("RGB")


def carry_flattened_effects(
    preview: Image.Image,
    old_backdrop: Image.Image,
    new_backdrop: Image.Image,
    foreground_alpha: Image.Image,
) -> Image.Image:
    """Photoshop's own rendering of the layers -- effects and all --
    moved onto a new backdrop.

    `preview` is the file's flattened picture as Photoshop saved it (the
    layers over the old backdrop, with every layer style rendered by
    Photoshop itself); `old_backdrop` is the backdrop alone; `new_backdrop`
    is the picture the layers now go over; `foreground_alpha` is where the
    layers' own pixels are (an "L" mask).

    A layer style is not in any layer's pixels -- Photoshop draws a drop
    shadow or glow at display time from the style's numbers -- so a
    composite made outside Photoshop had to approximate it, and a
    header's shadow came out soft and faint where Photoshop's was hard
    and black ("the layer style is not the same"). The preview has the
    real thing. Where the layers sit, the preview's pixels go over the
    new backdrop as they are. Everywhere else the preview differs from
    the old backdrop only by the styles: a darkening (a shadow) is the
    ratio preview/old, which for a black shadow is exactly its own
    coverage whatever the backdrop was, so the new backdrop takes the
    same ratio; a brightening (a glow) is the difference, added on.
    """
    try:
        import numpy as np
    except ImportError:
        out = new_backdrop.convert("RGB").copy()
        out.paste(preview.convert("RGB"), mask=foreground_alpha)
        return out
    size = new_backdrop.size
    P = np.asarray(preview.convert("RGB").resize(size), dtype=np.float32)
    old_rgba = old_backdrop.convert("RGBA").resize(size)
    B = np.asarray(old_rgba.convert("RGB"), dtype=np.float32)
    N = np.asarray(new_backdrop.convert("RGB"), dtype=np.float32)
    ratio = np.clip(P / np.maximum(B, 1.0), 0.0, 1.0)
    bright = np.clip(P - B, 0.0, 255.0)
    # Where the old backdrop had no pixels (a letterboxed fit, a backdrop
    # smaller than the canvas) there is nothing to compare against: the
    # new backdrop stands as it is there.
    known = (np.asarray(old_rgba.getchannel("A"), dtype=np.float32) > 0)[..., None]
    ratio = np.where(known, ratio, 1.0)
    bright = np.where(known, bright, 0.0)
    out = np.clip(N * ratio + bright, 0.0, 255.0)
    a = np.asarray(foreground_alpha.convert("L").resize(size), dtype=np.float32)[..., None] / 255.0
    out = out * (1.0 - a) + P * a
    return Image.fromarray(out.astype(np.uint8), "RGB")


def get_psd_layer_background(psd_path: Union[str, Path], layer_name: str) -> Optional[Image.Image]:
    """Return the PSD's own full composite with the named layer hidden --
    the *true* original pixels behind that layer, straight from the file,
    not a guess or a reconstruction, flattened to RGB.

    See _psd_composite_with_layers_hidden() above for how this is built.
    Flattening to RGB here is deliberate: this is used to patch a small
    "clean box" behind ONE foreground layer (a logo, a CTA, a product
    cutout) before drawing a new override into it, where the box is
    fully repainted anyway and no transparency needs to survive.
    get_psd_layer_foreground() just below is the RGBA counterpart, for
    the opposite case.
    """
    composite = _psd_composite_with_layers_hidden(psd_path, layer_name)
    if composite is None:
        return None
    return composite.convert("RGB")


def get_psd_layer_effect_reach(psd_path: Union[str, Path], layer_name: str) -> int:
    """How far, in the PSD's own pixels, the named layer's enabled
    effects reach beyond its pixels: a drop shadow's distance plus its
    size, an outer glow's or stroke's size. Photoshop draws these into
    the stored preview outside the layer's box, so a wipe of the box
    alone leaves a halo of the old words -- the frame and the dark
    band under a redrawn header."""
    try:
        from psd_tools import PSDImage

        psd = PSDImage.open(psd_path)
    except Exception:  # noqa: BLE001
        return 0
    reach = 0.0
    for layer in psd:
        if layer.name.strip().lower() != layer_name.strip().lower():
            continue
        try:
            effects = list(layer.effects or [])
        except Exception:  # noqa: BLE001
            effects = []
        for effect in effects:
            try:
                if getattr(effect, "enabled", True) is False:
                    continue
                kind = type(effect).__name__.lower()
                size = float(getattr(effect, "size", 0) or 0)
                distance = float(getattr(effect, "distance", 0) or 0)
                if "shadow" in kind:
                    reach = max(reach, distance + size)
                elif "glow" in kind or "stroke" in kind:
                    reach = max(reach, size)
            except Exception:  # noqa: BLE001
                continue
        break
    return int(math.ceil(reach))


_glyph_coverage_cache: dict = {}


def font_covers_text(font_name: Optional[str], text: str) -> bool:
    """Whether the installed font with this PostScript name has a glyph
    for every letter in `text`. Apple Symbols, say, has no accented
    Latin letters, so Spanish set in it came out as boxes; the caller
    falls back to a bundled face that does have them."""
    found = find_font_file(font_name)
    if found is None or not text:
        return True
    path, index = found
    letters = {ch for ch in text if not ch.isspace()}
    try:
        try:
            from fontTools.ttLib import TTFont

            key = (str(path), index)
            cmap = _glyph_coverage_cache.get(key)
            if cmap is None:
                font = TTFont(str(path), fontNumber=index, lazy=True)
                cmap = set(font.getBestCmap().keys())
                font.close()
                _glyph_coverage_cache[key] = cmap
            if not all(ord(ch) in cmap for ch in letters):
                return False
            # In the cmap is not the same as drawable: a font can map
            # a letter to an empty box of its own. The mask check
            # below settles it either way.
        except ImportError:
            pass
        # Without fontTools: a missing glyph draws as the font's
        # .notdef box, so a letter whose mask equals a private-use
        # character's mask (never in any real font) is missing.
        # U+0378 is unassigned in Unicode, so no font has a glyph for
        # it: its mask is what a missing letter looks like in this font.
        # (A private-use code point was the probe before, and Apple
        # Symbols has real glyphs there -- so nothing ever counted as
        # missing and the boxes stayed.)
        probe = ImageFont.truetype(str(path), 40, index=index)
        missing = probe.getmask("\u0378").tobytes()
        return all(probe.getmask(ch).tobytes() != missing for ch in letters)
    except Exception:  # noqa: BLE001
        return True


# The type layers this app ever retypes -- and so the only ones whose
# cached picture can disagree with their words.
APP_RETYPED_TEXT_LAYERS = ("header", "description", "legal", "cta")


def psd_saved_by_photoshop(psd) -> bool:
    """Whether Photoshop wrote this document last. Photoshop stores the
    document-level text engine block (Txt2) on every save of a file
    with type in it; this app removes that block whenever it retypes a
    layer (Photoshop would otherwise revert the words to it on click),
    so a file without it was last written here."""
    try:
        from psd_tools.constants import Tag

        blocks = psd.tagged_blocks
        return blocks is not None and Tag.TEXT_ENGINE_DATA in blocks
    except Exception:  # noqa: BLE001
        return True


def get_psd_untrusted_type_layers(psd_path: Union[str, Path]) -> set:
    """Lowercased names of the visible type layers whose cached picture
    can't be taken as showing their words.

    A type layer carries its words and a picture of how they last
    looked, and a composite made outside Photoshop shows the picture.
    Photoshop redraws the picture from the words on every save, so in
    a file it saved the two agree. In a file this app saved last they
    need not: the app retypes a layer's words and replaces its picture
    with its own drawing of them -- but a saved template gets its
    words from one run's form and its picture from what that run drew,
    which in French is a different thing, and an export that was never
    opened in Photoshop keeps whatever the app last wrote. A template
    in that state showed the old words no matter what language was
    chosen ("the 720x480 is not updating language"): the words said
    French, the picture still said English, and "already in French,
    leave it as saved" left the picture.

    So: a file Photoshop saved is trusted entirely; in one this app
    saved, every layer it ever retypes is drawn from its words afresh.
    Returns {} when the file can't be read."""
    try:
        from psd_tools import PSDImage
    except ImportError:
        return set()
    try:
        psd = PSDImage.open(psd_path)
    except Exception:  # noqa: BLE001
        return set()
    if psd_saved_by_photoshop(psd):
        return set()
    untrusted = set()
    for layer in psd:
        name = (layer.name or "").strip().lower()
        if not layer.visible or name not in APP_RETYPED_TEXT_LAYERS:
            continue
        if layer.kind == "type" or (layer.is_group() and any(
            child.kind == "type" for child in layer.descendants()
        )):
            untrusted.add(name)
    return untrusted


def get_psd_layer_foreground(
    psd_path: Union[str, Path], layer_names: Union[str, Iterable[str]], effects: bool = True
) -> Optional[Image.Image]:
    """Return the PSD's own full composite with `layer_names` hidden
    (a single layer name, or several at once), same construction as
    get_psd_layer_background() just above -- but kept as RGBA with real
    alpha, instead of flattened to RGB.

    Wherever a hidden layer used to draw, this composite is transparent
    (alpha 0); wherever any OTHER layer draws (a logo, a CTA button, a
    product cutout, body/description/header text, etc.) it stays fully
    opaque, exactly as Photoshop rendered it. Two use cases:

    - Hide just "background": fill the new background image in first,
      then alpha-composite this "everything but the background" layer
      back on top, restoring every other layer's real, original pixels
      pixel-for-pixel -- instead of a plain opaque paste wiping out
      everything else already sitting in that same box (a background
      layer's own box is typically the whole canvas, since it's the
      bottommost, full-frame layer).
    - Hide a specific layer (say "logo") *and* "background" together,
      when the background was already swapped earlier in this same
      request: alpha-compositing this on top of the current canvas
      re-cleans that one layer's old pixels away without reintroducing
      the old (now-replaced) background underneath them.

    Returns None under the same conditions as get_psd_layer_background().
    """
    composite = _psd_composite_with_layers_hidden(psd_path, layer_names, effects=effects)
    if composite is None:
        return None
    return composite.convert("RGBA")


def get_psd_layer_text_style(psd_path: Union[str, Path], layer_name: str) -> Optional[dict]:
    """Read the font family/size/color/weight a PSD's own text layer was
    set to in Photoshop, so a text-layer override (see
    apply_layer_text_override()) can match the template designer's
    original styling by default instead of always falling back to this
    tool's generic sans/near-black/autofit look.

    Returns a dict {"family": one of VALID_FONT_FAMILIES, "font_size": int,
    "color": (r, g, b) or None, "bold": bool}, or None if `psd-tools`
    isn't installed, the file can't be opened, no layer with that name
    exists, it isn't a text layer, or its styling data can't be parsed --
    callers should treat None as "no PSD styling available" and fall back
    to their own defaults, same as get_psd_layer_background().

    Only the *first* style run is read -- i.e. this assumes the layer's
    text is styled uniformly (one font/size/color throughout), which
    matches how these ad templates' description layers are actually set
    up. A layer with genuinely mixed inline styling just yields whatever
    its first run says.
    """
    try:
        from psd_tools import PSDImage
    except ImportError:
        return None
    try:
        psd = PSDImage.open(psd_path)
    except Exception:
        return None

    target = None
    for layer in psd:
        if layer.name.strip().lower() == layer_name.strip().lower():
            target = layer
            break
    if target is None or target.kind != "type":
        return None

    try:
        style_sheet = target.engine_dict["StyleRun"]["RunArray"][0]["StyleSheet"]["StyleSheetData"]
    except (KeyError, IndexError, TypeError):
        return None

    result: dict = {}

    # psd-tools represents these as its own Float/Integer/List/Dict
    # wrapper types (psd_tools.psd.engine_data), not plain Python
    # int/float/list/dict -- they support the usual operations (float(),
    # len(), indexing, .get()) but fail isinstance() checks against the
    # builtins, so everything below converts explicitly instead of
    # isinstance-checking, catching whatever conversion errors that turns
    # up as "this field wasn't usable" rather than letting them propagate.
    # FontSize is the size the type was SET at; if the layer was then
    # scaled with Free Transform (the norm in these templates -- a
    # description set at 66 and dragged up to 1.9x reads as 126 on the
    # canvas), the scale lives in the layer's transform matrix, not in
    # FontSize. Read without it, the redrawn text came out at the set
    # size: about half the height of the design in the worst cases.
    # Leading is stored in the same unscaled units, so it scales too.
    transform_scale = 1.0
    try:
        xx, _xy, _yx, yy, _tx, _ty = target.transform
        scale = float(yy) if yy else float(xx)
        if scale and scale > 0:
            transform_scale = scale
    except Exception:  # noqa: BLE001
        pass
    result["transform_scale"] = transform_scale
    try:
        font_size = float(style_sheet.get("FontSize"))
    except (TypeError, ValueError):
        font_size = None
    if font_size and font_size > 0:
        result["font_size"] = max(int(round(font_size * transform_scale)), 1)

    result["bold"] = bool(style_sheet.get("FauxBold"))

    # Paragraph alignment, as set in Photoshop: 0 left, 1 right, 2 centre;
    # the justified variants (3-6) fall back to left.
    try:
        justification = int(
            target.engine_dict["ParagraphRun"]["RunArray"][0]["ParagraphSheet"]["Properties"]["Justification"]
        )
        result["align"] = {0: "left", 1: "right", 2: "center"}.get(justification, "left")
    except Exception:  # noqa: BLE001
        pass

    if "font_size" in result:
        try:
            explicit_leading = float(style_sheet.get("Leading"))
        except (TypeError, ValueError):
            explicit_leading = None
        auto_leading = bool(style_sheet.get("AutoLeading"))
        if not auto_leading and explicit_leading and explicit_leading > 0:
            # An explicit leading value (line-to-line distance) the
            # template was actually set to in Photoshop, through the
            # same transform as the size.
            # Never tighter than 90% of the size: a leading left over
            # from an earlier, smaller setting would stack the lines on
            # top of each other if the words wrap here.
            result["line_height"] = max(int(round(explicit_leading * transform_scale)), int(result["font_size"] * 0.9), 1)
        else:
            # "Auto" leading (Photoshop's default, and what most text
            # layers actually use) has no single stored value to read --
            # Photoshop computes it from the font's own internal metrics,
            # which isn't available here since this tool substitutes a
            # bundled font for whatever the PSD actually used. 120% of
            # the font size is Photoshop's own standard auto-leading
            # ratio and the best available approximation.
            result["line_height"] = max(int(round(result["font_size"] * 1.2)), 1)

    color = None
    fill_color = style_sheet.get("FillColor")
    if fill_color is not None:
        try:
            color_type = int(fill_color.get("Type"))
            values = list(fill_color.get("Values"))
        except (TypeError, ValueError, AttributeError):
            color_type, values = None, None
        # Values is [alpha, r, g, b], each a 0..1 fraction -- this is the
        # RGB color-space case (Type 1), which is what every real-world
        # template so far has used; other color spaces (grayscale, CMYK,
        # Lab) are left unhandled rather than guessed at.
        if color_type == 1 and values is not None and len(values) == 4:
            try:
                color = tuple(max(0, min(255, round(float(v) * 255))) for v in values[1:4])
            except (TypeError, ValueError):
                color = None
    if color is not None:
        result["color"] = color

    family = "sans"
    try:
        font_names = target.font_names or []
    except Exception:
        font_names = []
    # The run's own font, by PostScript name: the StyleSheet's "Font" is
    # an index into the layer's FontSet, which lists every face the
    # layer uses (plus Adobe's invisible placeholder), not necessarily
    # in run order -- so font_names[0] is not "the font".
    font_name = None
    try:
        font_set = target.resource_dict.get("FontSet") or []
        raw = font_set[int(style_sheet.get("Font", 0))].get("Name")
        # psd-tools hands the name back as its own String type, whose
        # str() keeps the file's quotes ("'MyriadPro-Regular'").
        font_name = str(raw or "").strip().strip("'\"").strip() or None
    except Exception:  # noqa: BLE001
        font_name = font_names[0] if font_names else None
    if font_name:
        result["font_name"] = font_name
    name = (font_name or (font_names[0] if font_names else "")).lower()
    if name:
        if "mono" in name:
            family = "mono"
        elif "cond" in name:
            family = "condensed"
        elif "serif" in name and "sans" not in name:
            family = "serif"
    result["family"] = family

    # The layer's own effects (drop shadow, outer glow, stroke), so a
    # redraw carries them: without them a translated header lost the
    # shadow the design had, until the file was opened in Photoshop.
    try:
        result["effects"] = _psd_layer_effects(target)
    except Exception:  # noqa: BLE001
        pass

    # Per-line styling. A header set as "REHYDRATE / with a new summer /
    # REFRESHING / DRINK" has a size for each line; drawn at the first
    # run's size alone the layout was lost. Each paragraph gets the
    # style of the run its first real character sits in.
    try:
        result["lines"] = _psd_type_layer_lines(target, transform_scale, result)
    except Exception:  # noqa: BLE001
        pass

    return result if "font_size" in result else None


def _effect_color(effect):
    try:
        color = effect.color
        return (
            max(0, min(255, round(float(color[b"Rd  "])))),
            max(0, min(255, round(float(color[b"Grn "])))),
            max(0, min(255, round(float(color[b"Bl  "])))),
        )
    except Exception:  # noqa: BLE001
        return None


def _psd_layer_effects(layer) -> dict:
    """{"shadow": {...}, "glow": {...}, "stroke": {...}} for the layer's
    enabled effects, sizes in the PSD's own pixels."""
    out: dict = {}
    for effect in list(layer.effects or []):
        try:
            if getattr(effect, "enabled", True) is False:
                continue
            kind = type(effect).__name__.lower()
            if kind == "dropshadow" and "shadow" not in out:
                out["shadow"] = {
                    "color": _effect_color(effect) or (0, 0, 0),
                    "opacity": float(getattr(effect, "opacity", 75) or 0),
                    "angle": float(getattr(effect, "angle", 120) or 0),
                    "distance": float(getattr(effect, "distance", 0) or 0),
                    "size": float(getattr(effect, "size", 0) or 0),
                }
            elif kind == "outerglow" and "glow" not in out:
                out["glow"] = {
                    "color": _effect_color(effect) or (255, 255, 255),
                    "opacity": float(getattr(effect, "opacity", 75) or 0),
                    "size": float(getattr(effect, "size", 0) or 0),
                }
            elif kind == "stroke" and "stroke" not in out:
                out["stroke"] = {
                    "color": _effect_color(effect) or (0, 0, 0),
                    "size": float(getattr(effect, "size", 0) or 0),
                }
        except Exception:  # noqa: BLE001
            continue
    return out


def _draw_text_shadow(canvas, lines_xy, font, shadow, keep_alpha):
    """Photoshop-style drop shadow under text: the glyphs in the shadow
    colour, offset along the light angle, blurred by the size. `lines_xy`
    is [(x, y, text, font)] so styled lines can pass their own fonts."""
    if not shadow or not lines_xy:
        return canvas
    distance = float(shadow.get("distance", 0) or 0)
    size = float(shadow.get("size", 0) or 0)
    opacity = max(0.0, min(100.0, float(shadow.get("opacity", 75) or 0)))
    if opacity <= 0 or (distance <= 0 and size <= 0):
        return canvas
    angle = math.radians(float(shadow.get("angle", 120) or 0))
    dx = -distance * math.cos(angle)
    dy = distance * math.sin(angle)
    color = tuple(shadow.get("color") or (0, 0, 0))
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for x, y, text, line_font in lines_xy:
        draw.text((x + dx, y + dy), text, font=line_font or font, fill=(255, 255, 255, 255))
    alpha = layer.split()[3]
    if size > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=max(size / 2.0, 0.5)))
    alpha = alpha.point(lambda a: int(a * opacity / 100.0))
    shadow_layer = Image.new("RGBA", canvas.size, color + (0,))
    shadow_layer.putalpha(alpha)
    if keep_alpha:
        return Image.alpha_composite(canvas.convert("RGBA"), shadow_layer)
    return Image.alpha_composite(canvas.convert("RGBA"), shadow_layer).convert("RGB")


def _style_sheet_color(style_sheet):
    fill_color = style_sheet.get("FillColor")
    if fill_color is None:
        return None
    try:
        color_type = int(fill_color.get("Type"))
        values = list(fill_color.get("Values"))
        if color_type == 1 and len(values) == 4:
            return tuple(max(0, min(255, round(float(v) * 255))) for v in values[1:4])
    except (TypeError, ValueError, AttributeError):
        return None
    return None


def _psd_type_layer_lines(target, transform_scale: float, base: dict) -> list:
    """[{text, font_size, font_name, family, color, bold}] for each
    paragraph of a type layer, in canvas pixels. Empty when the layer
    has one paragraph (nothing per-line to keep) or can't be read."""
    engine = target.engine_dict
    text = str(engine["Editor"]["Text"].value)
    paragraphs = text.split("\r")
    if paragraphs and paragraphs[-1] == "":
        paragraphs = paragraphs[:-1]
    if len(paragraphs) < 2:
        return []
    run_lengths = [int(n) for n in engine["StyleRun"]["RunLengthArray"]]
    runs = list(engine["StyleRun"]["RunArray"])
    font_set = target.resource_dict.get("FontSet") or []

    def style_at(index):
        pos = 0
        for length, run in zip(run_lengths, runs):
            if index < pos + length:
                return run["StyleSheet"]["StyleSheetData"]
            pos += length
        return runs[-1]["StyleSheet"]["StyleSheetData"] if runs else None

    lines = []
    cursor = 0
    for paragraph in paragraphs:
        first = cursor
        for offset, ch in enumerate(paragraph):
            if not ch.isspace():
                first = cursor + offset
                break
        sheet = style_at(first)
        cursor += len(paragraph) + 1
        line = {"text": paragraph.strip()}
        try:
            size = float(sheet.get("FontSize")) * transform_scale
            line["font_size"] = max(int(round(size)), 1)
        except (TypeError, ValueError, AttributeError):
            line["font_size"] = base.get("font_size")
        try:
            raw = font_set[int(sheet.get("Font", 0))].get("Name")
            line["font_name"] = str(raw or "").strip().strip("'\"").strip() or base.get("font_name")
        except Exception:  # noqa: BLE001
            line["font_name"] = base.get("font_name")
        line["color"] = _style_sheet_color(sheet) if sheet is not None else None
        if line["color"] is None:
            line["color"] = base.get("color")
        line["bold"] = bool(sheet.get("FauxBold")) if sheet is not None else base.get("bold", False)
        line["family"] = base.get("family", "sans")
        lines.append(line)
    return lines


def apply_layer_styled_lines(
    base_image: Image.Image,
    bbox: Tuple[int, int, int, int],
    lines: list,
    *,
    align: str = "left",
    scale: float = 1.0,
    keep_alpha: bool = False,
    debug: Optional[dict] = None,
    shadow: Optional[dict] = None,
) -> Image.Image:
    """Paint `lines` -- [{text, font_size, font_name, family, bold,
    color}] -- into `bbox`, each line at its own size, stacked from the
    top of the box, the whole block shrunk uniformly only if it would
    not fit. This is how a translated header keeps the layout it had in
    English: the big word stays big, the small line stays small.
    `scale` is the template-to-output factor already applied to the box.
    """
    x0, y0, x1, y1 = bbox
    box_w, box_h = x1 - x0, y1 - y0
    lines = [dict(line) for line in lines if line.get("text")]
    if box_w <= 0 or box_h <= 0 or not lines:
        return base_image if keep_alpha else base_image.convert("RGB")

    def layout(factor):
        laid = []
        total = 0
        widest = 0
        for line in lines:
            size = max(int(round((line.get("font_size") or 20) * scale * factor)), 6)
            font = _load_font(size, bold=bool(line.get("bold")), family=line.get("family") or "sans", font_name=line.get("font_name"))
            left, top, right, bottom = font.getbbox(line["text"])
            width = right - left
            height = int(round(size * 1.2))
            laid.append((line, font, width, height, top))
            total += height
            widest = max(widest, width)
        return laid, total, widest

    laid, total, widest = layout(1.0)
    factor = min(1.0, box_w / widest if widest else 1.0, box_h / total if total else 1.0)
    if factor < 1.0:
        laid, total, widest = layout(factor)

    canvas = base_image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    y = y0
    used = []
    placed = []
    for line, font, width, height, top in laid:
        if align == "center":
            x = x0 + (box_w - width) // 2
        elif align == "right":
            x = x1 - width
        else:
            x = x0
        placed.append((x, y - top + (height - int(font.size * 1.2)) // 2, line, font))
        used.append(font.size)
        y += height
    if shadow:
        canvas = _draw_text_shadow(canvas, [(x, y_, line["text"], font) for x, y_, line, font in placed], None, shadow, True)
    for x, y_, line, font in placed:
        color = tuple(line.get("color") or (26, 26, 26))
        draw.text((x, y_), line["text"], font=font, fill=color + (255,))
    if debug is not None:
        faces = []
        for _x, _y, _line, font in placed:
            try:
                faces.append(" ".join(part for part in font.getname() if part))
            except Exception:  # noqa: BLE001
                faces.append("?")
        debug.update({
            "font_size": used[0] if used else None, "line_sizes": used, "lines": len(used),
            "family": "template", "bold": None, "line_height": None,
            "font_used": ", ".join(dict.fromkeys(faces)),
        })
    out = Image.alpha_composite(canvas, overlay)
    return out if keep_alpha else out.convert("RGB")


def apply_layer_image_override(
    base_image: Image.Image,
    bbox: Tuple[int, int, int, int],
    replacement: Image.Image,
    *,
    keep_alpha: bool = False,
) -> Image.Image:
    """Return a copy of `base_image` with `replacement` scaled to fit
    `bbox`'s height and centered within it, preserving its own aspect
    ratio.

    `replacement` is first trimmed to its own real content (its bounding
    box of non-transparent pixels), dropping any blank margin the source
    file happened to have around it. It's then scaled so its height
    exactly matches the box's height ("fit vertical") -- unless that
    would make it wider than the box, in which case it's scaled to fit
    the width instead so it never overflows the box's edges -- and
    centered both horizontally and vertically within the box, then
    alpha-composited on top of the original pixels there.

    Because this preserves aspect ratio, the box usually isn't covered
    edge-to-edge (e.g. a squarish logo centered in a wide box leaves
    margin on either side) -- callers should paint in the real
    background there first (see `get_psd_layer_background()`) so that
    margin shows the ad's actual background rather than whatever the
    old layer's content was.

    `keep_alpha=True` skips the final flatten-to-RGB and returns RGBA
    instead, with `base_image`'s own alpha preserved everywhere except
    where `replacement` was just composited in. Passing a fully
    transparent `base_image` this way isolates exactly what this call
    drew -- nothing else -- as its own standalone layer, for a real
    layered PSD export (see webapp.py's render loop) rather than the
    flattened preview this function normally produces.
    """
    x0, y0, x1, y1 = bbox
    box_w, box_h = x1 - x0, y1 - y0
    if box_w <= 0 or box_h <= 0:
        return base_image

    canvas = base_image.convert("RGBA").copy()
    rgba = replacement.convert("RGBA")
    content_bbox = rgba.split()[3].getbbox()
    if content_bbox:
        rgba = rgba.crop(content_bbox)
    content_w, content_h = rgba.size
    if content_w <= 0 or content_h <= 0:
        return canvas if keep_alpha else canvas.convert("RGB")

    scale = box_h / content_h
    if content_w * scale > box_w:
        scale = box_w / content_w
    new_w = max(round(content_w * scale), 1)
    new_h = max(round(content_h * scale), 1)
    resized = rgba.resize((new_w, new_h), Image.LANCZOS)

    paste_x = x0 + (box_w - new_w) // 2
    paste_y = y0 + (box_h - new_h) // 2
    canvas.alpha_composite(resized, (paste_x, paste_y))
    return canvas if keep_alpha else canvas.convert("RGB")


def apply_layer_background_override(
    base_image: Image.Image,
    bbox: Tuple[int, int, int, int],
    replacement: Image.Image,
    *,
    keep_alpha: bool = False,
    fit: str = "crop",
) -> Image.Image:
    """Return a copy of `base_image` with `replacement` filling `bbox`
    completely -- center-cropped to the box's exact aspect ratio (see
    center_crop_to_ratio()), then pasted in edge-to-edge.

    Deliberately different from apply_layer_image_override() above: that
    one is for a foreground object (a logo, a CTA button graphic, a
    product cutout) that should sit centered within its box at its own
    aspect ratio, with margin around it showing the real background
    through -- it trims transparent edges and never covers the box
    edge-to-edge unless the aspect ratios happen to match exactly. A
    background image is the opposite: the whole file is wanted, full
    frame, with no gaps -- so this crops-to-fill instead of fitting-with-
    letterboxing, and doesn't trim/expect any transparency (a background
    upload is typically a flat photo, not a cutout).

    `keep_alpha=True` skips the final flatten-to-RGB and returns RGBA
    instead -- see apply_layer_image_override()'s own `keep_alpha` for
    why (a real layered PSD export, not the flattened preview). The fill
    itself is still fully opaque either way; only `base_image`'s own
    alpha *outside* `bbox` is preserved when this is set.
    """
    x0, y0, x1, y1 = bbox
    box_w, box_h = x1 - x0, y1 - y0
    if box_w <= 0 or box_h <= 0:
        return base_image

    canvas = base_image.convert("RGBA").copy()
    source = replacement.convert("RGB")
    if fit == "contain":
        # For artwork the model laid out -- a headline, a logo lockup --
        # cropping to fill is what takes the right-hand third off the
        # words. Fit the whole thing inside the box instead and pad the
        # remainder with the image's own edge colour, so the margin reads
        # as part of the design rather than as black bars.
        fitted = resize_to_contain(source, (box_w, box_h))
        plate = Image.new("RGB", (box_w, box_h), _edge_colour(source))
        plate.paste(
            fitted,
            ((box_w - fitted.width) // 2, (box_h - fitted.height) // 2),
        )
        filled = plate.convert("RGBA")
    else:
        filled = center_crop_to_ratio(source, (box_w, box_h)).convert("RGBA")
    canvas.paste(filled, (x0, y0))
    return canvas if keep_alpha else canvas.convert("RGB")


def _edge_colour(image: Image.Image) -> Tuple[int, int, int]:
    """The average colour of a one-pixel frame around `image` -- what to
    pad with when fitting it into a box that isn't its shape. Sampling
    the border rather than the whole image keeps a dark vignette dark and
    a pale studio shot pale, instead of averaging a busy picture into
    mud."""
    small = image.convert("RGB").resize((32, 32))
    pixels = list(small.getdata())
    frame = [
        pixels[y * 32 + x]
        for y in range(32)
        for x in range(32)
        if x in (0, 31) or y in (0, 31)
    ]
    count = len(frame) or 1
    return tuple(sum(channel) // count for channel in zip(*frame))


def _reconstruct_box_background(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """Build an RGB patch, sized to `bbox`, that approximates "what was
    behind this box" as a smooth bilinear gradient between the colors
    found just outside its four corners.

    Each corner sample averages a small patch (clamped to the image
    bounds -- a box flush against an edge, like a header banner starting
    at y=0, just samples along that edge instead of reaching past it)
    just outside that corner of `bbox`. A 2x2 image built from the four
    corner colors, resized up to the box's full size with bilinear
    interpolation, reconstructs a smooth two-directional gradient -- a
    good approximation for the gradient/soft-color ad backgrounds this
    pipeline's templates typically use, since it blends the *actual*
    surrounding colors rather than a single flat average or an
    unrelated guess.
    """
    x0, y0, x1, y1 = bbox
    box_w, box_h = x1 - x0, y1 - y0
    w, h = image.size
    patch = 8

    def sample(cx: int, cy: int, dx: int, dy: int) -> Tuple[int, int, int]:
        # Average a small patch just outside the box at this corner, in
        # the (dx, dy) direction away from the box -- clamped so a box
        # flush against the canvas edge still gets a valid sample.
        sx0 = max(min(cx + (dx * 1), cx + dx * patch), 0) if dx >= 0 else max(cx + dx * patch, 0)
        sx0, sx1 = sorted((cx, cx + dx * patch))
        sy0, sy1 = sorted((cy, cy + dy * patch))
        sx0, sy0 = max(sx0, 0), max(sy0, 0)
        sx1, sy1 = min(sx1, w), min(sy1, h)
        if sx1 <= sx0 or sy1 <= sy0:
            # Degenerate (box touches this edge exactly) -- fall back to
            # a 1px-wide/tall strip right at the box's own edge.
            sx0, sy0 = max(min(cx, w - 1), 0), max(min(cy, h - 1), 0)
            sx1, sy1 = sx0 + 1, sy0 + 1
        region = image.crop((sx0, sy0, sx1, sy1)).convert("RGB")
        pixels = list(region.getdata())
        if not pixels:
            return (240, 240, 240)
        r = sum(p[0] for p in pixels) // len(pixels)
        g = sum(p[1] for p in pixels) // len(pixels)
        b = sum(p[2] for p in pixels) // len(pixels)
        return (r, g, b)

    top_left = sample(x0, y0, -1, -1)
    top_right = sample(x1, y0, 1, -1)
    bottom_left = sample(x0, y1, -1, 1)
    bottom_right = sample(x1, y1, 1, 1)

    corners = Image.new("RGB", (2, 2))
    corners.putpixel((0, 0), top_left)
    corners.putpixel((1, 0), top_right)
    corners.putpixel((0, 1), bottom_left)
    corners.putpixel((1, 1), bottom_right)
    return corners.resize((max(box_w, 1), max(box_h, 1)), Image.BILINEAR)


def _sample_edge_color(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    """Average the pixels in a thin ring just outside `bbox` (clamped to the
    image) as a plausible fill color for painting over that box -- a rough
    approximation of "what's behind this layer", good enough to blend a
    text patch into a mostly-uniform or gently-gradiented background."""
    x0, y0, x1, y1 = bbox
    w, h = image.size
    margin = 6
    ring_box = (
        max(x0 - margin, 0),
        max(y0 - margin, 0),
        min(x1 + margin, w),
        min(y1 + margin, h),
    )
    ring = image.crop(ring_box)
    mask = Image.new("L", ring.size, 255)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle(
        [
            x0 - ring_box[0],
            y0 - ring_box[1],
            x1 - ring_box[0],
            y1 - ring_box[1],
        ],
        fill=0,
    )
    pixels = [p for p, m in zip(ring.convert("RGB").getdata(), mask.getdata()) if m]
    if not pixels:
        return (240, 240, 240)
    r = sum(p[0] for p in pixels) // len(pixels)
    g = sum(p[1] for p in pixels) // len(pixels)
    b = sum(p[2] for p in pixels) // len(pixels)
    return (r, g, b)


def upscale_to_cover(
    image: Image.Image, target: Tuple[int, int], sharpen: bool = True
) -> Image.Image:
    """Enlarge `image` until it covers `target` on both axes, preserving
    its aspect ratio, and restore some of the bite the enlargement costs.

    For when a provider hands back less than was asked for -- Pollinations
    returns 768x768 however large a size is requested -- and the shortfall
    would otherwise be made up by each output size upscaling on its own,
    from the same too-small source, with no sharpening at all.

    Enlarging is interpolation: it invents no detail and softens every
    edge it touches. An unsharp mask can't invent detail either, but it
    restores local contrast at edges, which is most of what reads as
    sharpness. Radius scales with the enlargement -- a 2.5x blow-up
    smears over more pixels than a 1.2x one, so a fixed radius would
    under-correct the first and halo the second.

    Returns the image untouched when it already covers the target, so
    this costs nothing on a provider that honours the request.
    """
    width, height = image.size
    if width <= 0 or height <= 0:
        return image
    scale = max(target[0] / width, target[1] / height)
    if scale <= 1.0:
        return image

    enlarged = image.resize(
        (max(round(width * scale), 1), max(round(height * scale), 1)), Image.LANCZOS
    )
    if not sharpen:
        return enlarged
    # Tuned to stay clear of visible haloing: percent rises with the
    # scale factor but is capped, and the threshold leaves flat areas
    # (sky, gradients, bokeh) alone so noise isn't amplified.
    radius = min(0.6 + (scale - 1.0) * 0.7, 2.4)
    percent = int(min(60 + (scale - 1.0) * 45, 140))
    return enlarged.filter(
        ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=3)
    )


def apply_layer_cta_override(
    base_image: Image.Image,
    bbox: Tuple[int, int, int, int],
    text: str,
    *,
    button_color: Tuple[int, int, int] = (0, 87, 184),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    font_size: Optional[int] = None,
    font_family: str = "sans",
    glow: bool = False,
    glow_color: Tuple[int, int, int] = (255, 255, 255),
    glow_size: int = 12,
    glow_opacity: int = 100,
    keep_alpha: bool = False,
    stroke_size: int = 0,
    stroke_color: Tuple[int, int, int] = (0, 0, 0),
    border_size: int = 0,
    border_color: Tuple[int, int, int] = (0, 0, 0),
    corner_radius: Optional[int] = None,
) -> Image.Image:
    """Draw a pill-shaped CTA button filling `bbox`, with `text` centred.

    The layer-box counterpart to add_cta_button(). That one decides where
    a button goes from a `position` and sizes it to its own label; this
    one is handed the footprint the template's designer already chose --
    the "cta" layer's box -- and fills it. So there is no position
    argument here, and none in the form: a template's CTA sits where the
    template puts it.

    `font_size` is a ceiling, not a demand -- the label is shrunk until it
    fits the button's width, and truncated with an ellipsis if it still
    won't, since a CTA label is meant to stay on one line. Omitted, it
    starts from the button's own height, which is what makes the default
    look right at every output size without being told.

    `glow` matches the text override's: the pill is rendered as a mask,
    blurred, boosted and tinted underneath the crisp button, so the halo
    reads as light spilling out from behind it.
    """
    x0, y0, x1, y1 = bbox
    box_w, box_h = x1 - x0, y1 - y0
    if box_w <= 0 or box_h <= 0:
        return base_image
    # An empty label is a legitimate request: restyling the button of a
    # CTA group, where the words are set separately from the group's own
    # text layer. Only the text pass is skipped.
    draw_label = bool(text)

    canvas = base_image.convert("RGBA").copy() if keep_alpha else base_image.convert("RGB").copy()
    probe = ImageDraw.Draw(canvas)

    pad_x = max(int(box_w * 0.08), 6)
    max_text_w = max(box_w - 2 * pad_x, 8)
    size = font_size or max(int(box_h * 0.42), 8)
    font = _load_font(size, family=font_family)
    while size > 7:
        width = probe.textbbox((0, 0), text, font=font)[2]
        if width <= max_text_w:
            break
        size -= 1
        font = _load_font(size, family=font_family)
    # Still too wide at the floor -- trim rather than let it spill.
    if probe.textbbox((0, 0), text, font=font)[2] > max_text_w:
        trimmed = text
        while len(trimmed) > 1:
            trimmed = trimmed[:-1]
            candidate = trimmed.rstrip() + "…"
            if probe.textbbox((0, 0), candidate, font=font)[2] <= max_text_w:
                text = candidate
                break
        else:
            text = "…"

    # A pill by default -- half the height rounds the ends off
    # completely. `corner_radius` is a percentage of that, so 0 is a
    # square-cornered rectangle, 100 the pill, and anything between the
    # softened rectangle most templates actually use.
    radius = box_h // 2
    if corner_radius is not None:
        radius = round((box_h // 2) * (max(0, min(100, corner_radius)) / 100.0))

    if glow and glow_size > 0 and glow_opacity > 0:
        blur_radius = max(round(box_h * (glow_size / 100.0)), 1)
        glow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        ImageDraw.Draw(glow_layer).rounded_rectangle(
            [x0, y0, x1, y1], radius=radius, fill=(255, 255, 255, 255)
        )
        alpha = glow_layer.split()[3].filter(ImageFilter.GaussianBlur(radius=blur_radius))
        boost = 1.6 * (max(0, min(100, glow_opacity)) / 100.0)
        alpha = alpha.point(lambda a: min(255, int(a * boost)))
        colored = Image.new("RGBA", canvas.size, (glow_color[0], glow_color[1], glow_color[2], 0))
        colored.putalpha(alpha)
        if keep_alpha:
            canvas = Image.alpha_composite(canvas, colored)
        else:
            canvas = Image.alpha_composite(canvas.convert("RGBA"), colored).convert("RGB")

    draw = ImageDraw.Draw(canvas)
    # The button's own outline, as a percentage of its height for the
    # same reason every other stroke here is relative: one setting has to
    # hold across a 160x600 and a 1920x1080.
    border_px = 0
    if border_size:
        border_px = max(1, round((y1 - y0) * (max(0, min(100, border_size)) / 100.0)))
    draw.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=radius,
        fill=(button_color[0], button_color[1], button_color[2], 255)
        if keep_alpha
        else button_color,
        outline=(
            ((border_color[0], border_color[1], border_color[2], 255) if keep_alpha else border_color)
            if border_px
            else None
        ),
        width=border_px,
    )
    if draw_label:
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_x = x0 + (box_w - (text_bbox[2] - text_bbox[0])) / 2 - text_bbox[0]
        text_y = y0 + (box_h - (text_bbox[3] - text_bbox[1])) / 2 - text_bbox[1]
        # Same percentage-of-type-size stroke as the text layers -- see
        # apply_layer_text_override() for why it isn't taken in pixels.
        stroke_px = 0
        if stroke_size:
            stroke_px = max(1, round(font.size * (max(0, min(100, stroke_size)) / 100.0)))
        draw.text(
            (text_x, text_y),
            text,
            font=font,
            fill=(text_color[0], text_color[1], text_color[2], 255) if keep_alpha else text_color,
            stroke_width=stroke_px,
            stroke_fill=(
                ((stroke_color[0], stroke_color[1], stroke_color[2], 255) if keep_alpha else stroke_color)
                if stroke_px
                else None
            ),
        )
    return canvas


def apply_layer_text_override(
    base_image: Image.Image,
    bbox: Tuple[int, int, int, int],
    text: str,
    *,
    text_color: Tuple[int, int, int] = (26, 26, 26),
    align: str = "left",
    font_size: Optional[int] = None,
    exact_font_size: Optional[int] = None,
    font_family: str = "sans",
    font_name: Optional[str] = None,
    bold: bool = True,
    leading: Optional[int] = None,
    leading_reference_size: Optional[int] = None,
    debug: Optional[dict] = None,
    keep_alpha: bool = False,
    glow: bool = False,
    glow_color: Tuple[int, int, int] = (255, 255, 255),
    glow_size: int = 12,
    glow_opacity: int = 100,
    show_background: bool = False,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    background_opacity: int = 60,
    background_blur: int = 0,
    stroke_size: int = 0,
    stroke_color: Tuple[int, int, int] = (0, 0, 0),
    keep_size: bool = False,
    shadow: Optional[dict] = None,
) -> Image.Image:
    """Return a copy of `base_image` with `text` painted directly into
    `bbox` -- same idea as apply_layer_image_override(): whatever's
    already there (ideally the PSD's own true background for this box,
    patched in by the caller via get_psd_layer_background() -- see the
    _clean_layer_box() closure in webapp.py's render loop) is left alone
    and shows through around and behind the letters, instead of first
    being painted over with a guessed flat/gradient plate.

    The text is ALWAYS fit to `bbox` -- it never overflows the box,
    whether the size in play came from the PSD's own text layer
    (`font_size`) or from a user-typed override (`exact_font_size`).
    Both are just different sources for the same *ceiling*: text grows up
    to that size for short replacement text and shrinks below it for
    longer text, via the same largest-that-fits search render_creative()'s
    header/message banners use. `exact_font_size` wins when both are
    given (it's what the user explicitly asked for), but it's still a
    ceiling, not a demand -- a size the box's text genuinely can't fit at
    still gets shrunk, same as the PSD's own size would. `debug["clamped"]`
    is set when the rendered size ends up below whatever was requested,
    so a caller can tell the user their requested size didn't fit rather
    than leaving them to wonder why nothing visibly changed.

    `leading` is the PSD's own line-to-line distance in px, measured at
    `leading_reference_size` (the PSD's own font size). It's fed into the
    fit search itself (see fit_text_block()), not just applied
    afterward -- so whatever size wins the search is guaranteed to still
    fit once actually rendered with that line spacing. Without a
    reference pair (neither value given, or the PSD's own text style
    couldn't be read), this falls back to a generic ~1.2x-of-font-size
    approximation.

    Text is drawn flat, in `text_color`, matching a real PSD text layer,
    which is just a flat fill. Legibility normally comes from painting on
    the PSD's own true background for this box (see _clean_layer_box() in
    webapp.py) with the PSD's own text color, the same pairing the
    original template already used successfully.

    `glow=True` adds a soft halo behind the letterforms, in `glow_color`,
    for the case that pairing can't handle: text over a busy photo, where
    any flat colour loses somewhere in the frame. `glow_size` is a
    percentage of the font size (so a halo scales with the type rather
    than being a fixed pixel radius that looks heavy at 160x600 and
    invisible at 1920x1080). It uses the same technique as the CTA
    button's glow: render the glyphs as a white mask, blur its alpha,
    boost it back up (blurring dims it a lot), tint it, and composite it
    behind the crisp text. `glow_opacity` (0-100) scales that halo's
    strength, for when a full-strength one is heavier than the design
    wants -- 0 leaves the text as if no glow had been asked for.

    `show_background` draws a flat band behind the text block instead --
    the same idea as the header/message banners render_creative() draws,
    and the blunter answer to the same problem a glow solves. It spans
    the layer box's width and only the lines' own height, so it reads as
    a banner rather than filling the whole layer. `background_color`,
    `background_opacity` (0-100) and `background_blur` style it; a band is
    drawn under the glow, so the two can be combined.

    `background_blur` (0-100, as a percentage of the band's height) softens
    the band's edges into a gradient instead of a hard rectangle -- the
    difference between a label bar and a wash the text sits in. It's
    applied to the band's alpha before compositing, so the band fades out
    at its edges rather than the artwork behind it being blurred. 0, the
    default, keeps the hard edge.

    `keep_alpha=True` returns RGBA instead of flattening to RGB -- see
    apply_layer_image_override()'s own `keep_alpha` for why (a real
    layered PSD export). Drawing onto a fully transparent `base_image`
    this way isolates just the rendered glyphs, anti-aliased edges and
    all, as their own standalone layer.
    """
    x0, y0, x1, y1 = bbox
    box_w, box_h = x1 - x0, y1 - y0
    if box_w <= 0 or box_h <= 0:
        return base_image

    canvas = base_image.convert("RGBA").copy() if keep_alpha else base_image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    padding = max(int(min(box_w, box_h) * 0.06), 3)
    max_text_width = max(box_w - 2 * padding, 10)
    max_text_height = max(box_h - 2 * padding, 10)
    resolved_family = font_family if font_family in VALID_FONT_FAMILIES else "sans"
    requested_size = exact_font_size or font_size
    clipped_lines = 0
    if keep_size and requested_size:
        # Photoshop's paragraph box: the type keeps its size, wraps at
        # the box's width from the top, and whatever doesn't fit the
        # box's height is simply not shown. Shrinking to fit instead
        # made a translated description come out smaller than the
        # design, every time.
        padding = 0
        max_text_width, max_text_height = box_w, box_h
        font = _load_font(int(requested_size), bold=bold, family=resolved_family, font_name=font_name)
        lines = wrap_text_to_width(draw, text, font, max_text_width)
        if leading and leading_reference_size:
            line_height = max(int(round(leading * (font.size / float(leading_reference_size)))), 1)
        else:
            line_height = max(int(round(font.size * 1.2)), 1)
        fits = max(int(max_text_height // line_height), 1)
        if len(lines) > fits:
            clipped_lines = len(lines) - fits
            lines = lines[:fits]
    else:
        min_font_size = max(min(int(box_h * 0.12), 14), 7)
        max_font_size = requested_size if requested_size else None
        if max_font_size is not None:
            min_font_size = min(min_font_size, max_font_size)
        font, lines, line_height = fit_text_block(
            draw,
            text,
            max_text_width,
            max_text_height,
            min_font_size=min_font_size,
            max_font_size=max_font_size,
            family=resolved_family,
            bold=bold,
            leading=leading,
            leading_reference_size=leading_reference_size,
            font_name=font_name,
        )

    if debug is not None:
        # Populated so a caller (see webapp.py's render loop) can surface
        # exactly what was actually used -- not just what was *asked*
        # for -- on the results page, since "the font settings aren't
        # matching the PSD" can mean either "the PSD's values weren't
        # read" or "they were read but something downstream overrode
        # them," and those need different fixes.
        debug["font_size"] = font.size
        debug["line_height"] = line_height
        debug["family"] = resolved_family
        try:
            debug["font_used"] = " ".join(part for part in font.getname() if part)
        except Exception:  # noqa: BLE001
            debug["font_used"] = "?"
        debug["bold"] = bold
        debug["lines"] = len(lines)
        debug["requested_font_size"] = requested_size
        debug["clamped"] = bool(requested_size and font.size < requested_size)
        debug["clipped_lines"] = clipped_lines
    total_h = line_height * len(lines)
    # Top of the box when keeping the design's size (a paragraph box
    # starts at its top), centred otherwise.
    first_text_y = y0 if keep_size and requested_size else y0 + padding + (max_text_height - total_h) // 2

    def _line_x(line):
        line_w = draw.textlength(line, font=font)
        if align == "center":
            return x0 + padding + (max_text_width - line_w) / 2
        if align == "right":
            return x0 + padding + (max_text_width - line_w)
        return x0 + padding

    if show_background and background_opacity > 0:
        # Sized to the text block, not the layer box: a band that filled
        # the whole box would swamp a template whose text sits in a tall
        # box with room around it.
        band_pad = max(int(padding * 0.5), 2)
        band_top = max(first_text_y - band_pad, y0)
        band_bottom = min(first_text_y + total_h + band_pad, y1)
        band = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        ImageDraw.Draw(band).rectangle(
            [x0, band_top, x1, band_bottom],
            fill=(
                background_color[0],
                background_color[1],
                background_color[2],
                round(255 * max(0, min(100, background_opacity)) / 100.0),
            ),
        )
        if background_blur > 0:
            # Blurring the band's own alpha, not the artwork behind it --
            # the band fades out at its edges while whatever it sits on
            # stays sharp. Sized off the band's height so one setting
            # reads the same at every output size, like the glow's.
            band_h = max(band_bottom - band_top, 1)
            blur_px = max(round(band_h * (max(0, min(100, background_blur)) / 100.0)), 1)
            band.putalpha(band.split()[3].filter(ImageFilter.GaussianBlur(radius=blur_px)))
        if keep_alpha:
            canvas = Image.alpha_composite(canvas, band)
        else:
            canvas = Image.alpha_composite(canvas.convert("RGBA"), band).convert("RGB")
        draw = ImageDraw.Draw(canvas)

    if glow and glow_size > 0 and glow_opacity > 0:
        # Sized off the font rather than the box: a halo that scales with
        # the type reads the same at every output size, where a fixed
        # pixel radius would look heavy on a 160x600 and vanish on a
        # 1920x1080.
        blur_radius = max(round(font.size * (glow_size / 100.0)), 1)
        glow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_layer)
        # The glyphs are thickened before blurring. Blurring spreads a
        # fixed amount of alpha over a bigger area, so without this a
        # larger radius produced a FAINTER halo -- turning the size up to
        # make a glow more visible made it disappear instead. Growing the
        # source with the radius keeps the halo's strength roughly
        # constant and lets size mean spread, which is what it says.
        stroke = max(1, round(blur_radius * 0.8))
        glow_y = first_text_y
        for line in lines:
            glow_draw.text(
                (_line_x(line), glow_y),
                line,
                font=font,
                fill=(255, 255, 255, 255),
                stroke_width=stroke,
                stroke_fill=(255, 255, 255, 255),
            )
            glow_y += line_height
        alpha = glow_layer.split()[3].filter(ImageFilter.GaussianBlur(radius=blur_radius))
        # 1.6 recovers the strength a Gaussian blur costs; the opacity
        # scale is applied in the same pass so a dialled-back glow thins
        # out evenly instead of being clipped.
        boost = 1.6 * (max(0, min(100, glow_opacity)) / 100.0)
        alpha = alpha.point(lambda a: min(255, int(a * boost)))
        colored_glow = Image.new("RGBA", canvas.size, (glow_color[0], glow_color[1], glow_color[2], 0))
        colored_glow.putalpha(alpha)
        if keep_alpha:
            canvas = Image.alpha_composite(canvas, colored_glow)
        else:
            canvas = Image.alpha_composite(canvas.convert("RGBA"), colored_glow).convert("RGB")
        # alpha_composite() returns a new image -- rebind the Draw handle.
        draw = ImageDraw.Draw(canvas)

    # An outline round the glyphs. Scaled off the font size rather than
    # taken as pixels: the same override runs at every output size, and a
    # 3px stroke that frames 176px type at 1920x1080 swallows the 30px it
    # becomes at 160x600. The control is a percentage of the type size,
    # so it stays proportionate wherever it lands.
    stroke_px = 0
    if stroke_size:
        # font.size, not font_size: the latter is the requested ceiling and
        # is None whenever the size came from the PSD. This is the size the
        # fit search actually settled on, which is what the stroke has to
        # stay proportionate to.
        stroke_px = max(1, round(font.size * (max(0, min(100, stroke_size)) / 100.0)))
    stroke_rgb = (
        stroke_color if not keep_alpha else (stroke_color[0], stroke_color[1], stroke_color[2], 255)
    )
    if shadow:
        placed = []
        shadow_y = first_text_y
        for line in lines:
            placed.append((_line_x(line), shadow_y, line, font))
            shadow_y += line_height
        canvas = _draw_text_shadow(canvas, placed, font, shadow, keep_alpha)
        draw = ImageDraw.Draw(canvas)
    text_y = first_text_y
    for line in lines:
        text_x = _line_x(line)
        draw.text(
            (text_x, text_y),
            line,
            font=font,
            fill=text_color if not keep_alpha else (text_color[0], text_color[1], text_color[2], 255),
            stroke_width=stroke_px,
            stroke_fill=stroke_rgb if stroke_px else None,
        )
        text_y += line_height

    return canvas



def auto_transparent_background(image: Image.Image, tolerance: int = 30) -> Image.Image:
    """Best-effort background removal for a layer-override image (e.g. a
    logo) that was exported flat (a solid or near-solid color behind the
    mark) instead of as a proper cutout.

    Whether `image` is "already a cutout" is decided from its *border*
    pixels specifically (checking the whole image's alpha minimum is too
    easily fooled by a handful of anti-aliased edge pixels around opaque
    text/shapes elsewhere in the image, which would wrongly look like
    "already has transparency" and skip removal entirely even though the
    background itself is fully opaque). If those border pixels are
    themselves meaningfully transparent, this is left untouched. Otherwise
    it flood-fills inward from several border points -- corners and edge
    midpoints, since the background might be split into disconnected
    regions relative to any single seed -- treating pixels within
    `tolerance` of each seed's color as background and making them
    transparent. If that ends up erasing almost the whole image (a sign
    the background wasn't actually uniform -- a busy photo, say), it's
    treated as a bad guess and the original opaque image is returned
    instead of risking a mostly-blank logo.
    """
    rgba = image.convert("RGBA")
    w, h = rgba.size
    if w < 2 or h < 2:
        return rgba

    seeds = [
        (0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
        (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2),
    ]
    border_alphas = [rgba.getpixel(seed)[3] for seed in seeds]
    if min(border_alphas) < 200:
        # The border itself already has real transparency -- treat this
        # as an existing cutout rather than a flat export.
        return rgba

    working = rgba.copy()
    for seed in seeds:
        try:
            ImageDraw.floodfill(working, seed, (0, 0, 0, 0), thresh=tolerance)
        except Exception:
            continue

    new_alpha = working.split()[3]
    transparent_fraction = new_alpha.histogram()[0] / float(w * h)
    if transparent_fraction > 0.92:
        # Almost everything got erased -- the "uniform background" guess
        # was likely wrong; better to keep the original opaque image than
        # hand back a nearly-blank one.
        return rgba
    return working
