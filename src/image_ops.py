"""Image transforms: pixel-size cropping and campaign-message overlay."""

from __future__ import annotations

import math
import re
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
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not open asset '{path}': {exc}.") from exc


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
# few sizes (300x600 "Half Page Ad", 970x250 "Billboard", 728x480 "Wide
# Rectangle") that aren't in the active WEB_AD_SIZES preset above but are
# still recognized if requested explicitly via --sizes. "Wide Rectangle"
# isn't an official IAB name -- there isn't one for this size -- just a readable
# label instead of falling back to an ugly reduced-fraction ratio (91:60).
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
    (728, 480): "Wide Rectangle",
    (300, 600): "Half Page Ad",
    (970, 250): "Billboard",
    (1920, 1080): "Full HD / 1080p Broadcast",
    (1280, 720): "720p Broadcast",
    (3840, 2160): "4K UHD Broadcast",
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
    (728, 480): "desktop",
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
    """Derive a human-readable aspect ratio label (e.g. '16:9') from pixel dimensions."""
    g = math.gcd(width, height) or 1
    return f"{width // g}:{height // g}"


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
_FONTS_DIR = Path(__file__).resolve().parent.parent / "fonts"

_FONT_LOAD_WARNED = False


def _load_font(size: int, bold: bool = True, family: str = "sans") -> ImageFont.FreeTypeFont:
    global _FONT_LOAD_WARNED
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
        font = _load_font(size, bold=bold, family=family)
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
    return {
        (layer.name or "").strip().lower()
        for layer in psd
        if (layer.name or "").strip() and layer.visible
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
    for layer in psd:
        name = (layer.name or "").strip()
        if not name or layer.kind != "type":
            continue
        if visible_only and not layer.visible:
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
    psd_path: Union[str, Path], layer_names: Union[str, Iterable[str]]
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

    return composite


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


def get_psd_layer_foreground(
    psd_path: Union[str, Path], layer_names: Union[str, Iterable[str]]
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
    composite = _psd_composite_with_layers_hidden(psd_path, layer_names)
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
    try:
        font_size = float(style_sheet.get("FontSize"))
    except (TypeError, ValueError):
        font_size = None
    if font_size and font_size > 0:
        result["font_size"] = max(int(round(font_size)), 1)

    result["bold"] = bool(style_sheet.get("FauxBold"))

    if "font_size" in result:
        try:
            explicit_leading = float(style_sheet.get("Leading"))
        except (TypeError, ValueError):
            explicit_leading = None
        auto_leading = bool(style_sheet.get("AutoLeading"))
        if not auto_leading and explicit_leading and explicit_leading > 0:
            # An explicit leading value (line-to-line distance) the
            # template was actually set to in Photoshop -- use it as-is.
            result["line_height"] = max(int(round(explicit_leading)), 1)
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
    if font_names:
        name = font_names[0].lower()
        if "mono" in name:
            family = "mono"
        elif "cond" in name:
            family = "condensed"
        elif "serif" in name and "sans" not in name:
            family = "serif"
    result["family"] = family

    return result if "font_size" in result else None


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
        debug["bold"] = bold
        debug["lines"] = len(lines)
        debug["requested_font_size"] = requested_size
        debug["clamped"] = bool(requested_size and font.size < requested_size)

    total_h = line_height * len(lines)
    first_text_y = y0 + padding + (max_text_height - total_h) // 2

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
