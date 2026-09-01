"""A minimal local web UI on top of the same rendering engine the CLI uses.

The CLI (`python -m src.main`) is built around a full campaign brief: a
region, an audience, 2+ products, GenAI generation, localization,
compliance checks. That's the right tool for a real campaign, but it's
overkill when someone just wants to try "this one hero image, this
headline, this description" and see a set of sized creatives immediately.

This app is that quick path: upload a hero image (or a short product
video -- a frame gets extracted, same as the CLI), type a header and a
description, hit generate, and download every size as a zip. It calls the
exact same `render_creative()` function the CLI's pipeline uses (see
src/creative_render.py), so a creative made here looks identical to one
made by the full pipeline at the same size/fit-mode/header/logo/message.

Run with:
    pip install -r requirements.txt
    python webapp.py
then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

import io
import json
import os
import re
import secrets
import shutil
import uuid
import zipfile
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request, send_file, url_for
from PIL import Image
from werkzeug.utils import secure_filename

from src.creative_render import render_creative, render_creative_layers
from src.psd_export import save_layered_psd, set_type_layer_colors
from src.compliance import check_profanity, check_trademark_text
from src.image_ops import (
    get_psd_text_layers,
    DEFAULT_SIZES,
    VALID_BADGE_POSITIONS,
    VALID_CTA_POSITIONS,
    VALID_FONT_FAMILIES,
    VALID_LOGO_POSITIONS,
    VALID_TEXT_ALIGNMENTS,
    VIDEO_EXTENSIONS,
    apply_layer_background_override,
    apply_layer_image_override,
    apply_layer_cta_override,
    apply_layer_text_override,
    auto_transparent_background,
    center_crop_to_ratio,
    find_missing_brand_colors,
    get_psd_canvas_size,
    get_psd_backdrop,
    get_psd_layer_background,
    get_psd_layer_boxes,
    get_psd_layer_foreground,
    get_psd_layer_stack,
    get_psd_layer_text_style,
    map_box_through_fit,
    open_as_rgb,
    parse_size,
    parse_sizes,
    resize_to_contain,
    ratio_label,
    size_label,
    size_name,
)
from src.providers import PROVIDER_NAMES, ImageProviderError, MockImageProvider, get_provider
from src.storage import SUPPORTED_EXTENSIONS

# Sane bounds for a user-supplied font size, in pixels -- just a safety
# valve against nonsense input (0, negative, absurdly huge); the autofit
# path (font size left blank) isn't bound by this at all.
MIN_CUSTOM_FONT_SIZE = 4
MAX_CUSTOM_FONT_SIZE = 2000

DEFAULT_LOGO_SCALE_PERCENT = 16
DEFAULT_LOGO_OPACITY_PERCENT = 100

DEFAULT_BADGE_SCALE_PERCENT = 35
DEFAULT_BADGE_OPACITY_PERCENT = 100

BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "outputs" / "web"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Templates saved here are applied automatically to their matching output
# size on every future /generate request -- no re-upload needed. See
# _default_size_templates() below and default_templates/README.txt.
# Every run's zip is copied here as well as kept in its job folder. A
# job folder is named after a random id and lives under outputs/web/,
# which is fine for serving a page but no good for finding last
# Tuesday's campaign -- this is the browsable copy, named for the
# product and campaign it belongs to.
DOWNLOADS_DIR = BASE_DIR / "downloads"

DEFAULT_TEMPLATES_DIR = BASE_DIR / "default_templates"
DEFAULT_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Logos need alpha transparency to composite cleanly -- keep that upload
# restricted to formats that actually carry it, unlike the hero image
# (which accepts video too, per SUPPORTED_EXTENSIONS).
ALLOWED_LOGO_EXTENSIONS = (".png", ".webp")

# The badge image is more general-purpose than the logo -- it can be a
# full-frame tint/texture as easily as a badge -- so plain photos (JPG) are
# allowed too, not just transparency-capable formats.
ALLOWED_BADGE_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg")

# For the PSD layer-override fields (logo/CTA/product image) -- same
# tolerance as the badge image, since these can be flat JPGs too.
ALLOWED_LAYER_IMAGE_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg")

# Size-specific PSD templates: each one becomes the background for exactly
# the output size it's paired with (a Pillow-rendered flattened preview of
# the PSD, nothing more -- no layer extraction or role recognition). Kept
# separate from SUPPORTED_EXTENSIONS so the general hero image field stays
# plain-image/video only; PSD only enters through this dedicated section.
ALLOWED_PSD_TEMPLATE_EXTENSIONS = (".psd",)
MAX_PSD_TEMPLATES = 4

# The "quick campaign" single-input mode: upload just this one flagship
# size and every other exported size comes from default_templates/.
CONTENT_PSD_SIZE = (728, 480)

# Every PSD template used this request -- a per-request template row or a
# saved default -- must have layers with these exact (case-insensitive)
# names. See get_psd_layer_boxes() for how layer names are read.
REQUIRED_PSD_LAYERS = ("logo", "description", "product")

# Which of a quick-campaign content PSD's own layers get pushed out to
# every OTHER size in the batch. That upload is the flagship design for
# the campaign, so its artwork is meant to restyle the whole set, not
# just fill its own slot -- one upload, a re-skinned campaign. Text isn't
# in this list: the header/description fields already apply across every
# template size on their own.
PROPAGATED_CONTENT_PSD_LAYERS = ("background", "product", "logo", "cta")


def _content_psd_layer_images(psd_path) -> dict:
    """Pull a content PSD's named layers out as standalone RGBA images,
    keyed by lowercased layer name, ready to be fed straight into the
    same layer-override machinery a manual per-layer upload uses.

    Each layer arrives from get_psd_layer_stack() as a canvas-sized image
    that's transparent everywhere except where that layer draws. A
    foreground layer is cropped to its own drawn content first -- that's
    what a user uploading a logo or product cutout by hand would provide,
    and it's what lets apply_layer_image_override() fit it into a
    different size's box instead of scaling a mostly-empty canvas.
    "background" is left full-canvas, since its override fills the box
    edge to edge rather than being fitted inside it.

    Returns {} when the layer stack can't be read at all -- the caller
    treats that as "nothing to propagate" and every other size just
    renders from its saved template unchanged, exactly as before.
    """
    stack = get_psd_layer_stack(psd_path)
    if not stack:
        return {}
    images = {}
    for name, layer_image in stack:
        key = name.strip().lower()
        if key not in PROPAGATED_CONTENT_PSD_LAYERS:
            continue
        if key == "background":
            images[key] = layer_image
            continue
        content_box = layer_image.getbbox()
        if content_box is None:
            # An empty layer -- propagating it would blank that box on
            # every other size rather than restyle it.
            continue
        images[key] = layer_image.crop(content_box)
    return images

# All the plain (non-file) fields captured into form_state.json for the
# Edit button (see /edit/<job_id>) -- everything the form can prefill
# except the multi-value "sizes" checkboxes (handled separately, since
# request.form.getlist() is needed) and the checkbox fields below (stored
# as booleans instead of raw strings).
EDIT_TEXT_FIELD_NAMES = (
    "product_name", "market", "audience", "campaign_message",
    "brand_color_1", "brand_color_2", "brand_color_3",
    "ai_hero_prompt", "ai_hero_provider",
    "upload_ai_prompt", "upload_ai_provider",
    "layer_header_glow_color", "layer_header_glow_size", "layer_header_glow_opacity",
    "layer_header_align", "layer_header_background_color", "layer_header_background_opacity",
    "layer_description_glow_color", "layer_description_glow_size", "layer_description_glow_opacity",
    "layer_description_align", "layer_description_background_color", "layer_description_background_opacity",
    "layer_cta_text", "layer_cta_font_family", "layer_cta_font_size",
    "layer_cta_button_color", "layer_cta_text_color",
    "layer_cta_glow_color", "layer_cta_glow_size", "layer_cta_glow_opacity",
    "header", "description", "custom_sizes",
    "fit_mode",
    "header_text_color", "header_align", "header_font_size",
    "message_text_color", "message_align", "message_font_size",
    "cta_text", "cta_position", "cta_button_color", "cta_text_color",
    "cta_font_size", "cta_font_family",
    "logo_position", "logo_scale", "logo_opacity", "logo_offset_x", "logo_offset_y",
    "badge_position", "badge_scale", "badge_opacity",
    "video_frame_seconds",
    "layer_header_text",
    "layer_header_font_family", "layer_header_font_size", "layer_header_text_color",
    "layer_description_text",
    "layer_description_font_family", "layer_description_font_size", "layer_description_text_color",
    "psd_size_1", "psd_size_2", "psd_size_3", "psd_size_4",
)
EDIT_CHECKBOX_FIELD_NAMES = (
    "header_no_background", "header_glow",
    "message_no_background", "message_glow",
    "cta_glow",
    "cta_above_message",
    "layer_header_use_custom_color",
    "layer_description_use_custom_color",
    "brand_color_1_enabled", "brand_color_2_enabled", "brand_color_3_enabled",
    "ai_hero_enabled",
    "upload_ai_enabled",
    "layer_header_glow",
    "layer_description_glow",
    "layer_header_background",
    "layer_description_background",
    "layer_cta_glow",
)

# Euclidean RGB distance under which a pixel counts as "matching" a brand
# color for find_missing_brand_colors() -- see that function's docstring
# for why an exact-match check would be too strict.
BRAND_COLOR_MATCH_TOLERANCE = 30

# Fixed filename the AI-hero-fallback always saves under (see the
# AI-generated-hero block in generate()). Used both to write it and, on
# the Edit page, to recognize a carried-forward hero image as one we
# generated ourselves rather than something the user uploaded -- see the
# hero_fresh/hero_path block above.
AI_GENERATED_HERO_FILENAME = "ai_generated_hero.png"
# The Upload Creative generator's output -- the campaign artwork used
# when no content PSD was designed. Named apart from the hero image so
# the two never overwrite each other in a job that used both.
AI_GENERATED_CAMPAIGN_FILENAME = "ai_generated_campaign.png"

# Matches a WxH size anywhere in a filename (not just at the start --
# real-world default-template files look like "tester-728x480.psd" or
# "hero_970x90_v2.psd", size embedded mid-name after a prefix).
_SIZE_IN_FILENAME_RE = re.compile(r"(\d+)\s*[xX]\s*(\d+)")


def _default_size_templates() -> tuple:
    """Scan DEFAULT_TEMPLATES_DIR (non-recursive) for .psd files whose
    filename encodes a WxH size, and return (templates, template_paths) --
    both {(width, height): ...}, images and source paths respectively (the
    path is needed later to look up that size's named layer boxes for the
    layer-override feature). Runs on every /generate POST, so a single
    bad/corrupt file in the folder must not take down the whole request --
    it's skipped instead. If two files match the same size, the last one
    found wins.
    """
    templates: dict = {}
    template_paths: dict = {}
    if not DEFAULT_TEMPLATES_DIR.is_dir():
        return templates, template_paths
    for path in sorted(DEFAULT_TEMPLATES_DIR.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_PSD_TEMPLATE_EXTENSIONS:
            continue
        match = _SIZE_IN_FILENAME_RE.search(path.name)
        if not match:
            continue
        width, height = int(match.group(1)), int(match.group(2))
        try:
            templates[(width, height)] = open_as_rgb(path)
        except Exception:
            continue
        template_paths[(width, height)] = path
    return templates, template_paths


# How far a quick-campaign content PSD may sit from a saved template's
# size and still be treated as *that* size rather than a size of its own.
# The motivating case is a hand-built 728x480 delivery file dropped into a
# campaign whose saved template for that slot is 720x480: 8px wider, same
# height, visually the same creative. Exporting both is a near-duplicate
# nobody asked for, so the upload updates the existing slot instead. The
# ratio check is what keeps this honest -- it's the difference between "a
# slightly-off version of this creative" and "a different creative".
CONTENT_PSD_SNAP_RATIO_TOLERANCE = 0.05   # aspect ratio within 5%
CONTENT_PSD_SNAP_SIZE_TOLERANCE = 0.10    # each dimension within 10%


def _snap_to_template_size(size, template_sizes) -> tuple:
    """Map a content PSD's own pixel size onto a near-identical saved
    template size, so an uploaded 728x480 updates the existing 720x480
    creative instead of exporting alongside it.

    Returns the matching size from `template_sizes`, or `size` unchanged
    if nothing is close enough (a genuinely new size still exports as
    itself). Ties break on the smallest combined pixel difference.
    """
    width, height = size
    if not width or not height or size in template_sizes:
        return size
    ratio = width / height
    best = None
    for candidate in template_sizes:
        candidate_width, candidate_height = candidate
        if not candidate_width or not candidate_height:
            continue
        if abs(candidate_width / candidate_height - ratio) / ratio > CONTENT_PSD_SNAP_RATIO_TOLERANCE:
            continue
        if abs(candidate_width - width) / width > CONTENT_PSD_SNAP_SIZE_TOLERANCE:
            continue
        if abs(candidate_height - height) / height > CONTENT_PSD_SNAP_SIZE_TOLERANCE:
            continue
        distance = abs(candidate_width - width) + abs(candidate_height - height)
        if best is None or distance < best[0]:
            best = (distance, candidate)
    return best[1] if best else size


import datetime as _datetime

_WATCHED_SOURCE_FILES = [
    Path(__file__),
    BASE_DIR / "src" / "image_ops.py",
    BASE_DIR / "src" / "creative_render.py",
    BASE_DIR / "templates" / "index.html",
    BASE_DIR / "templates" / "result.html",
]
_newest_mtime = max((p.stat().st_mtime for p in _WATCHED_SOURCE_FILES if p.is_file()), default=None)
BUILD_STAMP = (
    _datetime.datetime.fromtimestamp(_newest_mtime).strftime("%Y-%m-%d %H:%M:%S")
    if _newest_mtime
    else "unknown"
)
# Printed on startup and shown in the page footer -- a running server
# doesn't reload code on a file change, so if this doesn't match the time
# of your latest edit/save, the process needs restarting.
print(f"[webapp] code build stamp: {BUILD_STAMP} (restart after any edit to pick up changes)")

SIZE_PRESET_CHOICES = [
    ("default", "Social defaults -- 1080x1080, 1080x1920, 1920x1080"),
    ("web-top7", "Web ad sizes (9) -- Leaderboard, Medium Rectangle, Skyscraper, etc."),
    ("broadcast", "Broadcast/video frame sizes (3) -- 1080p, 720p, 4K UHD"),
]

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.secret_key = os.environ.get("WEBAPP_SECRET_KEY", secrets.token_hex(16))
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB -- generous enough for a short product video


def _allowed(filename: str, extensions) -> bool:
    return Path(filename).suffix.lower() in extensions


def _parse_hex_color(value: str, default=(255, 255, 255)):
    """Parse a '#rrggbb' string (as sent by <input type="color">) into an
    (r, g, b) tuple, falling back to `default` for anything malformed."""
    value = (value or "").strip().lstrip("#")
    if len(value) != 6:
        return default
    try:
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return default


def _parse_optional_font_size(value: str):
    """Parse an optional font-size field into a clamped int, or None to fall
    back to automatic sizing when it's blank or not a valid number."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        size = int(float(value))
    except ValueError:
        return None
    return max(MIN_CUSTOM_FONT_SIZE, min(size, MAX_CUSTOM_FONT_SIZE))


def _parse_align(value: str, default: str) -> str:
    return value if value in VALID_TEXT_ALIGNMENTS else default


def _parse_font_family(value: str, default: str = "sans") -> str:
    return value if value in VALID_FONT_FAMILIES else default


def _parse_percent(value: str, default: int, min_value: int = 0, max_value: int = 100) -> int:
    """Parse a 0-100 percent field into a clamped int, falling back to
    `default` when it's blank or not a valid number."""
    value = (value or "").strip()
    if not value:
        return default
    try:
        percent = int(float(value))
    except ValueError:
        return default
    return max(min_value, min(percent, max_value))


def _parse_signed_int(value: str, default: int = 0, min_value: int = -2000, max_value: int = 2000) -> int:
    """Parse a possibly-negative pixel-offset field into a clamped int,
    falling back to `default` when it's blank or not a valid number. Used
    for the logo's manual nudge offsets, where negative means left/up."""
    value = (value or "").strip()
    if not value:
        return default
    try:
        parsed = int(float(value))
    except ValueError:
        return default
    return max(min_value, min(parsed, max_value))


def _slugify_for_filename(text: str, *, max_length: int = 40) -> str:
    """Turn arbitrary user text (e.g. a product name) into a short,
    filesystem-safe token usable in a downloaded filename -- runs of
    anything that isn't a letter, digit, dash, or underscore collapse to
    a single underscore, and the result is capped to `max_length` chars
    (leaving room for a "_WIDTHxHEIGHT.ext" suffix alongside it) so an
    unusually long product name can't produce an unwieldy filename.

    Returns "" (never raises) when there's nothing safe left to keep --
    e.g. a product name that's entirely emoji/punctuation -- so callers
    can fall back to a generic default instead of a filename made of
    nothing but underscores.
    """
    collapsed = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_")
    return collapsed[:max_length]


GLOW_SIZE_MIN, GLOW_SIZE_MAX, GLOW_SIZE_DEFAULT = 1, 100, 12


def _parse_glow_size(raw) -> int:
    """A text glow's radius as a percentage of the font size.

    Relative rather than absolute so one setting reads the same across
    every output size -- a fixed pixel radius looks heavy on a 160x600 and
    disappears on a 1920x1080. Anything unparseable or out of range falls
    back to the default rather than erroring: a bad glow size shouldn't
    fail a render.
    """
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return GLOW_SIZE_DEFAULT
    return max(GLOW_SIZE_MIN, min(GLOW_SIZE_MAX, value))


def _parse_glow_opacity(raw) -> int:
    """How strong a text glow is, 0-100. Same forgiving parse as the size
    above: unusable input falls back to full strength rather than failing
    a render over a decoration.
    """
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return 100
    return max(0, min(100, value))


def _save_upload(file_storage, dest_dir: Path) -> Path:
    """Save an uploaded file, preserving its extension (open_as_rgb() and
    render_creative() both branch on it -- e.g. to detect a video)."""
    safe_name = secure_filename(file_storage.filename) or "upload"
    dest = dest_dir / safe_name
    file_storage.save(dest)
    return dest


def _carry_forward_upload(field_name, uploads_dir: Path, prior_job_dir, prior_form_state: dict):
    """When editing a prior job (see /edit/<job_id>) and no new file was
    chosen for `field_name` this time, reuse the file uploaded for it last
    time -- copied into *this* job's uploads_dir so this job's directory
    stays self-contained (safe to delete the prior job later without
    breaking this one). Returns None when there's nothing to carry
    forward: not editing, that field was never set, or the prior file has
    since gone missing from disk (e.g. its job was cleaned up)."""
    if prior_job_dir is None:
        return None
    prior_rel = (prior_form_state.get("files") or {}).get(field_name)
    if not prior_rel:
        return None
    prior_path = prior_job_dir / "uploads" / prior_rel
    if not prior_path.is_file():
        return None
    dest_path = uploads_dir / prior_rel
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prior_path, dest_path)
    return dest_path


def _session_index_path(session_id: str) -> Path:
    """Where a campaign session's {slot -> job_id} index lives -- see
    _load_session_campaigns() and the session_id/campaign_slot handling in
    generate(). Deliberately a sibling of JOBS_DIR's job folders (not
    inside any one job's own folder), since one session covers several
    jobs, not just one."""
    return JOBS_DIR / "_sessions" / f"{secure_filename(session_id)}.json"


def _load_session_campaigns(session_id, fallback_job_id):
    """Build the list of {"prefill", "prefill_files", "edit_job_id"} dicts
    for every campaign card that belongs to `session_id` -- i.e. every
    "Create Campaign" card that was actually generated together on one
    page load (see the hidden session_id/campaign_slot fields each
    campaign card's <form> carries, and how generate() records them into
    the session index). A campaign card that was on the page but never
    itself submitted has nothing saved for it and can't be recovered --
    only campaigns that were actually generated come back.

    Falls back to a single-campaign list built from `fallback_job_id`
    alone (today's pre-session-tracking behavior) whenever there's no
    session_id, or its index can't be read, or it ends up empty -- so a
    job from before this feature existed (or any other edge case) still
    opens to *something* editable rather than an empty page.
    """
    fallback = [{
        "prefill": {},
        "prefill_files": {},
        "edit_job_id": fallback_job_id,
    }]
    if not session_id:
        return fallback
    index_path = _session_index_path(session_id)
    if not index_path.is_file():
        return fallback
    try:
        slots = json.loads(index_path.read_text()).get("slots") or {}
    except (OSError, ValueError):
        return fallback
    campaigns = []
    for slot_key in sorted(slots, key=lambda k: (len(k), k)):
        slot_job_id = slots[slot_key]
        state_path = JOBS_DIR / slot_job_id / "form_state.json"
        if not state_path.is_file():
            continue
        try:
            slot_state = json.loads(state_path.read_text())
        except (OSError, ValueError):
            continue
        campaigns.append({
            "prefill": slot_state.get("fields") or {},
            "prefill_files": slot_state.get("files") or {},
            "edit_job_id": slot_job_id,
        })
    return campaigns or fallback


def _session_campaign_jobs(session_id):
    """[(slot, job_id), ...] for one session's campaigns, in slot order.

    The session index is the only thing that knows a set of jobs belong
    together -- each campaign card submits its own form and becomes its
    own job, so without it a multi-campaign page is just unrelated jobs.
    Slots are sorted the same way _load_session_campaigns() sorts them
    (shorter key first, so 2 comes before 10, not after).
    """
    if not session_id:
        return []
    index_path = _session_index_path(session_id)
    if not index_path.is_file():
        return []
    try:
        slots = json.loads(index_path.read_text()).get("slots") or {}
    except (OSError, ValueError):
        return []
    return [(slot, slots[slot]) for slot in sorted(slots, key=lambda k: (len(k), k))]


def _editable_text_layers() -> set:
    """Which of the named text layers are actually editable right now --
    i.e. present AND switched on in at least one saved template.

    A layer switched off in Photoshop can't be restyled: there are no
    visible words to recolour or resize, and the renderer skips it. The
    form greys its fields out rather than accepting settings that would
    quietly do nothing. Judged across all saved templates together, since
    one enabled somewhere is enough for the field to be worth offering.
    """
    editable = set()
    _templates, template_paths = _default_size_templates()
    for path in template_paths.values():
        for name in get_psd_text_layers(path, visible_only=True):
            editable.add(name)
    return editable


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        size_presets=SIZE_PRESET_CHOICES,
        video_extensions=VIDEO_EXTENSIONS,
        build_stamp=BUILD_STAMP,
        campaigns=[{"prefill": {}, "prefill_files": {}, "edit_job_id": None}],
        session_id=uuid.uuid4().hex,
        editable_text_layers=_editable_text_layers(),
    )


@app.route("/edit/<job_id>", methods=["GET"])
def edit(job_id):
    """Reload the form pre-filled with a prior job's submission -- the
    "Edit" button on the results page. File inputs can't be pre-populated
    by browsers for security reasons, so those instead show a "currently:
    <filename>" hint (see prefill_files) and, on re-submit, the original
    file is carried forward automatically unless the user picks a new
    one -- see _carry_forward_upload() and the edit_job_id handling in
    generate().

    When this job was generated as part of a multi-campaign page (see
    _load_session_campaigns()), every other campaign generated alongside
    it on that same page comes back too, each in its own editable card --
    not just this one job in isolation."""
    state_path = JOBS_DIR / job_id / "form_state.json"
    if not state_path.is_file():
        flash(
            "Can't edit that batch -- its submitted details aren't available "
            "anymore. Starting a fresh form instead."
        )
        return redirect(url_for("index"))
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        flash(
            "Can't edit that batch -- its saved details couldn't be read. "
            "Starting a fresh form instead."
        )
        return redirect(url_for("index"))
    session_id = state.get("session_id") or uuid.uuid4().hex
    campaigns = _load_session_campaigns(state.get("session_id"), job_id)
    return render_template(
        "index.html",
        size_presets=SIZE_PRESET_CHOICES,
        video_extensions=VIDEO_EXTENSIONS,
        build_stamp=BUILD_STAMP,
        campaigns=campaigns,
        session_id=session_id,
        editable_text_layers=_editable_text_layers(),
    )


@app.route("/generate", methods=["POST"])
def generate():
    # Editing a prior job (see /edit/<job_id>) carries a hidden
    # edit_job_id field -- load that job's saved form_state.json so file
    # fields the user didn't re-upload this time can be carried forward
    # (see _carry_forward_upload()) instead of forcing a re-upload.
    edit_job_id = (request.form.get("edit_job_id") or "").strip() or None
    prior_job_dir = None
    prior_form_state: dict = {}
    if edit_job_id and re.fullmatch(r"[0-9a-f]{32}", edit_job_id):
        candidate_dir = JOBS_DIR / edit_job_id
        state_path = candidate_dir / "form_state.json"
        if state_path.is_file():
            try:
                prior_form_state = json.loads(state_path.read_text())
            except (OSError, ValueError):
                prior_form_state = {}
            else:
                prior_job_dir = candidate_dir

    hero_file = request.files.get("hero_image")
    hero_fresh = hero_file is not None and bool(hero_file.filename)
    if hero_fresh and not _allowed(hero_file.filename, SUPPORTED_EXTENSIONS):
        flash(
            f"'{hero_file.filename}' isn't a supported file type. Accepted: "
            + ", ".join(SUPPORTED_EXTENSIONS)
        )
        return redirect(url_for("index"))

    # AI-generated hero image -- an explicitly opted-in fallback for
    # whatever size(s) end up with no uploaded hero image and no matching
    # PSD template (see the generation call further down, once `sizes`
    # and `size_templates` are both final). Reuses the same GenAI
    # provider abstraction (src/providers/) the CLI pipeline
    # (src/pipeline.py) already calls -- this is the first thing in the
    # web app that actually invokes it.
    # The Upload Creative panel's own generator: stands in for a content
    # PSD the user hasn't designed yet. Same providers, same prompt
    # handling as the Manual Creative one below, but a different job --
    # this one supplies the campaign's artwork to the saved templates
    # rather than a hero image for a plain render.
    upload_ai_enabled = bool(request.form.get("upload_ai_enabled"))
    upload_ai_prompt = (request.form.get("upload_ai_prompt") or "").strip() or None
    upload_ai_provider = request.form.get("upload_ai_provider", "pollinations")
    if upload_ai_provider not in PROVIDER_NAMES:
        upload_ai_provider = "pollinations"

    ai_hero_enabled = bool(request.form.get("ai_hero_enabled"))
    ai_hero_prompt = (request.form.get("ai_hero_prompt") or "").strip() or None
    ai_hero_provider = request.form.get("ai_hero_provider", "pollinations")
    if ai_hero_provider not in PROVIDER_NAMES:
        ai_hero_provider = "pollinations"

    # Campaign brief -- product name / market / audience / campaign
    # message. Purely informational context about *this* batch: it isn't
    # composited into the creatives (there's no template placeholder for
    # it), just carried through to the results page and saved/carried
    # forward like every other field, so a batch's intent stays attached
    # to it when reviewing or editing later.
    product_name = (request.form.get("product_name") or "").strip() or None
    # Used to name downloaded files (PNG/PSD per size, and the zip) after
    # the product this batch is for, alongside each creative's own size --
    # see the `filename = f"{file_name_prefix}_{label}.png"` etc. below.
    # Falls back to the original generic "creative" prefix whenever no
    # product name was given (or it didn't sanitize down to anything
    # usable), so a batch with no product name is unaffected.
    product_name_slug = _slugify_for_filename(product_name) if product_name else ""
    # Which campaign card on the page this submission came from. Parsed
    # up here rather than down with the rest of the session bookkeeping
    # because it names files: two campaign cards in one session are two
    # separate jobs producing the same sizes, so without it their
    # downloads are same-named files that overwrite each other in
    # whatever folder they're unzipped into.
    try:
        campaign_slot = int((request.form.get("campaign_slot") or "1").strip())
    except ValueError:
        campaign_slot = 1
    campaign_label = f"campaign{campaign_slot}"
    file_name_prefix = f"{product_name_slug}_{campaign_label}" if product_name_slug else f"creative_{campaign_label}"
    market = (request.form.get("market") or "").strip() or None
    audience = (request.form.get("audience") or "").strip() or None
    campaign_message = (request.form.get("campaign_message") or "").strip() or None

    # Campaign brief is required -- every one of its four fields, not
    # composited into the creatives, this is now attached context that
    # has to travel with every batch (product/downloaded-file naming,
    # results page, editing later) rather than something that might or
    # might not be there.
    missing_brief_fields = [
        field_label
        for value, field_label in (
            (product_name, "Product name"),
            (market, "Market"),
            (audience, "Audience"),
            (campaign_message, "Campaign message"),
        )
        if not value
    ]
    if missing_brief_fields:
        flash(
            "Campaign brief is required -- please fill in: "
            + ", ".join(missing_brief_fields)
            + "."
        )
        return redirect(url_for("index"))

    # Profanity check -- blocks generation outright, same as the campaign
    # brief being incomplete, rather than just a warning on the results
    # page. Covers every free-text field that ends up visible on a
    # creative or the results page, whether or not it's already been
    # parsed into a local variable above; the ones parsed later (header/
    # description/CTA/layer description) are read fresh from the raw form
    # here since this check runs before they're otherwise needed.
    profanity_fields = [
        ("Product name", product_name),
        ("Market", market),
        ("Audience", audience),
        ("Campaign message", campaign_message),
        ("AI hero image prompt", ai_hero_prompt),
        ("Header/title", (request.form.get("header") or "").strip()),
        ("Description/message", (request.form.get("description") or "").strip()),
        ("Call-to-action text", (request.form.get("cta_text") or "").strip()),
        ("Header/title (update)", (request.form.get("layer_header_text") or "").strip()),
        ("Description/message (update)", (request.form.get("layer_description_text") or "").strip()),
    ]
    flagged_fields = [label for label, value in profanity_fields if value and check_profanity(value)]
    if flagged_fields:
        flash(
            "That contains language we can't allow through -- please edit: "
            + ", ".join(flagged_fields)
            + "."
        )
        return redirect(url_for("index"))

    # Brand colors -- up to three, each independently opt-in (a swatch
    # with nothing checked contributes nothing; there's no meaningful
    # "blank" for an <input type="color">, which always carries a value).
    # Checked against every rendered creative below (see the brand-color
    # check in the main render loop) and any that don't show up anywhere
    # in a given size's output get a warning on the results page.
    brand_colors = []
    for i in (1, 2, 3):
        if request.form.get(f"brand_color_{i}_enabled"):
            brand_colors.append(_parse_hex_color(request.form.get(f"brand_color_{i}"), default=(0, 0, 0)))

    headline = (request.form.get("header") or "").strip() or None
    message = (request.form.get("description") or "").strip() or None
    fit_mode = request.form.get("fit_mode", "crop")
    if fit_mode not in ("crop", "contain"):
        fit_mode = "crop"

    header_text_color = _parse_hex_color(request.form.get("header_text_color"), default=(255, 255, 255))
    header_show_background = not request.form.get("header_no_background")
    header_glow = bool(request.form.get("header_glow"))
    header_glow_color = _parse_hex_color(request.form.get("header_glow_color"), default=(255, 255, 255))
    header_align = _parse_align(request.form.get("header_align"), default="center")
    header_font_size = _parse_optional_font_size(request.form.get("header_font_size"))
    message_text_color = _parse_hex_color(request.form.get("message_text_color"), default=(255, 255, 255))
    message_show_background = not request.form.get("message_no_background")
    message_glow = bool(request.form.get("message_glow"))
    message_glow_color = _parse_hex_color(request.form.get("message_glow_color"), default=(255, 255, 255))
    message_align = _parse_align(request.form.get("message_align"), default="left")
    message_font_size = _parse_optional_font_size(request.form.get("message_font_size"))

    cta_text = (request.form.get("cta_text") or "").strip() or None
    cta_position = request.form.get("cta_position", "bottom-center")
    if cta_position not in VALID_CTA_POSITIONS:
        cta_position = "bottom-center"
    cta_button_color = _parse_hex_color(request.form.get("cta_button_color"), default=(0, 87, 184))
    cta_text_color = _parse_hex_color(request.form.get("cta_text_color"), default=(255, 255, 255))
    cta_font_size = _parse_optional_font_size(request.form.get("cta_font_size"))
    cta_font_family = _parse_font_family(request.form.get("cta_font_family"))
    cta_glow = bool(request.form.get("cta_glow"))
    cta_glow_color = _parse_hex_color(request.form.get("cta_glow_color"), default=(255, 255, 255))
    cta_above_message = bool(request.form.get("cta_above_message"))

    selected_presets = request.form.getlist("sizes")
    custom_sizes_raw = (request.form.get("custom_sizes") or "").strip()
    spec_parts = list(selected_presets)
    if custom_sizes_raw:
        spec_parts.append(custom_sizes_raw)
    try:
        sizes = parse_sizes(",".join(spec_parts)) if spec_parts else list(DEFAULT_SIZES)
    except ValueError as exc:
        flash(f"Couldn't parse the sizes you entered: {exc}")
        return redirect(url_for("index"))

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir = job_dir / "uploads"
    uploads_dir.mkdir(exist_ok=True)

    if hero_fresh:
        hero_path = _save_upload(hero_file, uploads_dir)
    else:
        prior_hero_rel = (prior_form_state.get("files") or {}).get("hero_image")
        if ai_hero_enabled and prior_hero_rel == AI_GENERATED_HERO_FILENAME:
            # The hero image being carried forward is itself the AI-generated
            # placeholder from a previous submission, and the AI checkbox is
            # still checked on this edit. Don't carry it forward as-is --
            # that would permanently lock in the first generated image and
            # make the prompt/provider fields silently do nothing on every
            # future edit. Leave hero_path unset so the AI block below
            # regenerates from the current prompt instead.
            hero_path = None
        else:
            hero_path = _carry_forward_upload("hero_image", uploads_dir, prior_job_dir, prior_form_state)
    hero_provided = hero_path is not None

    # Size-specific PSD templates -- up to MAX_PSD_TEMPLATES rows of
    # (psd_size_N, psd_file_N) fields. Each row with a file attaches a
    # flattened PSD as the background for that exact output size, and
    # forces that size into the batch even if it wasn't otherwise
    # checked/typed above.
    psd_templates: dict = {}
    psd_template_paths: dict = {}
    psd_file_paths: dict = {}  # {row index: Path} -- for form_state.json, see below
    # (label, Path) for every PSD actually uploaded *this request* (not
    # carried forward from a prior edit) -- scanned for profanity in their
    # text layers below, once content_psd's own fresh-upload is known too.
    fresh_psd_uploads = []
    for i in range(1, MAX_PSD_TEMPLATES + 1):
        psd_size_raw = (request.form.get(f"psd_size_{i}") or "").strip()
        psd_file = request.files.get(f"psd_file_{i}")
        psd_file_fresh = psd_file is not None and bool(psd_file.filename)
        # The "x" button next to a row on the Edit page (see index.html)
        # sets this hidden field so a carried-forward template can be
        # cancelled outright -- otherwise there'd be no way to say "stop
        # using a template here, go back to the hero image for this size"
        # short of overwriting it with a different .psd.
        psd_cleared = bool(request.form.get(f"psd_size_{i}_clear"))
        if psd_file_fresh:
            if not _allowed(psd_file.filename, ALLOWED_PSD_TEMPLATE_EXTENSIONS):
                flash(
                    f"PSD template row {i}: '{psd_file.filename}' isn't a supported file type. Accepted: "
                    + ", ".join(ALLOWED_PSD_TEMPLATE_EXTENSIONS)
                )
                return redirect(url_for("index"))
            psd_path = _save_upload(psd_file, uploads_dir)
            fresh_psd_uploads.append((f"PSD template row {i}", psd_path))
        elif psd_cleared:
            psd_path = None
        else:
            psd_path = _carry_forward_upload(f"psd_file_{i}", uploads_dir, prior_job_dir, prior_form_state)
        psd_file_provided = psd_path is not None
        if psd_path is not None:
            psd_file_paths[i] = psd_path
        if not psd_size_raw and not psd_file_provided:
            continue
        if psd_file_provided and not psd_size_raw:
            flash(f"PSD template row {i}: choose a target size (e.g. 728x480) for the uploaded PSD file.")
            return redirect(url_for("index"))
        if psd_size_raw and not psd_file_provided:
            flash(f"PSD template row {i}: you entered a size ({psd_size_raw}) but didn't attach a .psd file.")
            return redirect(url_for("index"))
        try:
            psd_width, psd_height = parse_size(psd_size_raw)
        except ValueError as exc:
            flash(f"PSD template row {i}: {exc}")
            return redirect(url_for("index"))
        try:
            psd_templates[(psd_width, psd_height)] = open_as_rgb(psd_path)
        except ValueError as exc:
            flash(f"PSD template row {i}: {exc}")
            return redirect(url_for("index"))
        psd_template_paths[(psd_width, psd_height)] = psd_path

    # "Quick campaign" single-input mode: upload just the one flagship
    # 728x480 PSD and disregard the Output sizes / Custom sizes selections
    # entirely -- the exported batch becomes this size plus whatever's
    # already saved in default_templates/, nothing else.
    content_psd_file = request.files.get("content_psd")
    content_psd_fresh = content_psd_file is not None and bool(content_psd_file.filename)
    if content_psd_fresh:
        if not _allowed(content_psd_file.filename, ALLOWED_PSD_TEMPLATE_EXTENSIONS):
            flash(
                f"728x480 content PSD: '{content_psd_file.filename}' isn't a supported file type. "
                "Accepted: " + ", ".join(ALLOWED_PSD_TEMPLATE_EXTENSIONS)
            )
            return redirect(url_for("index"))
        content_psd_path = _save_upload(content_psd_file, uploads_dir)
        fresh_psd_uploads.append(("728x480 content PSD", content_psd_path))
    else:
        content_psd_path = _carry_forward_upload("content_psd", uploads_dir, prior_job_dir, prior_form_state)
    content_psd_provided = content_psd_path is not None
    content_psd_image = None
    if content_psd_provided:
        try:
            content_psd_image = open_as_rgb(content_psd_path)
        except ValueError as exc:
            flash(f"728x480 content PSD: {exc}")
            return redirect(url_for("index"))
        # The upload renders as its own size (see the size_templates merge
        # below) *and* pulls in whatever's already saved in
        # default_templates/ -- "Output sizes"/"Custom sizes" and the
        # general hero image are still ignored in this mode either way.
        sizes = []

    # The Upload Creative generator makes the campaign's backdrop.
    # Generated at the content PSD's own size, since it plays that role:
    # source artwork the saved templates are built from, fed in as a
    # background-layer override further down, which is what carries it
    # onto every template size.
    #
    # It runs whether or not a content PSD was uploaded. With no PSD it
    # stands in for one entirely, so a campaign can be built before the
    # flagship 728x480 exists. With a PSD it replaces just that file's
    # background, which is the point of picking it -- every other layer
    # the PSD carries, and everything the saved templates carry, stays
    # exactly where it was designed.
    upload_ai_image = None
    upload_ai_path = None
    if upload_ai_enabled:
        upload_ai_prompt_text = upload_ai_prompt or (
            f"professional studio product photo of {product_name or 'the product'}, clean background"
        )
        try:
            upload_ai_image = get_provider(upload_ai_provider).generate(
                upload_ai_prompt_text, width=CONTENT_PSD_SIZE[0], height=CONTENT_PSD_SIZE[1]
            )
            background_notes_pending = (
                f"Campaign artwork generated with AI ({upload_ai_provider}) in place of a content "
                f"PSD -- prompt: \"{upload_ai_prompt_text}\"."
            )
        except ImageProviderError as exc:
            # Same resilience as the hero generator: a flaky free API
            # degrades to the offline placeholder rather than failing the
            # whole run, and says so instead of quietly looking worse.
            upload_ai_image = MockImageProvider().generate(upload_ai_prompt_text)
            background_notes_pending = (
                f"Campaign artwork: the '{upload_ai_provider}' AI provider failed ({exc}) -- used the "
                f"offline placeholder generator instead. Prompt: \"{upload_ai_prompt_text}\"."
            )
        upload_ai_path = uploads_dir / AI_GENERATED_CAMPAIGN_FILENAME
        upload_ai_image.save(upload_ai_path)
    else:
        background_notes_pending = None

    # Profanity check, PSD text layers -- same hard gate as the typed
    # form fields above, just sourced from whatever's actually typed into
    # a text layer inside a freshly uploaded PSD (e.g. a template's
    # "description" layer). Only PSDs uploaded *this* request are
    # scanned -- one already carried forward from a prior edit was
    # already checked the first time it came in, and default_templates/
    # saved templates aren't a fresh "upload" at all.
    for psd_label, psd_path_to_scan in fresh_psd_uploads:
        for layer_name, layer_text in get_psd_text_layers(psd_path_to_scan).items():
            if check_profanity(layer_text):
                flash(
                    f"{psd_label}: the '{layer_name}' text layer contains language we can't allow "
                    "through -- please edit it in the PSD and re-upload."
                )
                return redirect(url_for("index"))

    background_notes = []  # shown on the results page -- flash() only survives a redirect, and this path doesn't redirect
    background_warnings = []  # same idea, but rendered in red -- for things worth flagging (e.g. a missing brand color), not just FYI context
    if background_notes_pending:
        background_notes.append(background_notes_pending)

    # Saved default templates (default_templates/) define the batch, but
    # only for a templated campaign -- one where the quick-campaign
    # content PSD field was used. That upload is the flagship design the
    # rest of the set is built from, so the folder is what it gets built
    # against.
    #
    # A campaign WITHOUT that upload is the plain path: a hero image and
    # the sizes asked for, nothing more. Scanning the folder there would
    # hand every such campaign the same seven saved templates -- which is
    # how a second campaign card, cloned blank with its file inputs
    # reset, ended up previewing a set indistinguishable from the first
    # one's.
    if content_psd_provided or upload_ai_image is not None:
        default_templates, default_template_paths = _default_size_templates()
    else:
        default_templates, default_template_paths = {}, {}
    if content_psd_provided and not default_templates:
        background_notes.append(
            "default_templates/ doesn't have any saved templates yet -- only the "
            "uploaded 728x480 content PSD's own size was exported."
        )
    size_templates = dict(default_templates)
    size_template_paths = dict(default_template_paths)
    content_psd_size = None
    if content_psd_provided:
        # Snapped rather than added: an upload a few pixels off a saved
        # template's size is a new version of that creative, not an extra
        # one, so it takes over that slot and the preview count stays
        # equal to the number of saved templates.
        content_psd_size = _snap_to_template_size(content_psd_image.size, default_templates)
        if content_psd_size != content_psd_image.size:
            background_notes.append(
                f"The uploaded content PSD is {content_psd_image.size[0]}x{content_psd_image.size[1]} -- close "
                f"enough to the saved {size_label(*content_psd_size)} template that it replaced that creative "
                "instead of exporting as an extra size of its own."
            )
        size_templates[content_psd_size] = content_psd_image
        size_template_paths[content_psd_size] = content_psd_path
    size_templates.update(psd_templates)
    size_template_paths.update(psd_template_paths)

    # Every template in play this request -- whether a per-request PSD
    # template row, the content_psd trigger's saved defaults, or both --
    # must have "logo", "description", and "product" named layers. The
    # layer-override feature (and anyone editing these templates going
    # forward) depends on all three being present and consistently named;
    # a template missing one wouldn't fail loudly on its own -- it would
    # just silently skip that layer's override -- so this catches it
    # upfront with a clear error instead.
    for (template_width, template_height), template_path in size_template_paths.items():
        template_layer_boxes = get_psd_layer_boxes(template_path)
        missing_layers = [
            name for name in REQUIRED_PSD_LAYERS if name not in template_layer_boxes
        ]
        if missing_layers:
            flash(
                f"{size_label(template_width, template_height)} template "
                f"({Path(template_path).name}) is missing required layer(s): "
                + ", ".join(missing_layers)
                + ". Every PSD template needs 'logo', 'description', and 'product' layers "
                "(named exactly that, case-insensitive)."
            )
            return redirect(url_for("index"))

    # Layer overrides -- lives in the PSD section only: swap the
    # description text, logo image, CTA image, or product image, applied
    # to EVERY template-covered size that has a matching named layer (each
    # size's own PSD has its own layout, so the same override lands at a
    # different position/scale per size, driven by that size's own layer
    # bbox). Only meaningful when there's at least one template-covered
    # size to apply them to; parsed once, applied per-size in the render
    # loop below via get_psd_layer_boxes()/apply_layer_*_override().
    layer_header_text = (request.form.get("layer_header_text") or "").strip() or None
    # Same idea as the description override just below -- these three let
    # the user override the PSD's own font family/size/color for the
    # "header" text layer specifically.
    layer_header_font_family = (request.form.get("layer_header_font_family") or "").strip()
    if layer_header_font_family not in VALID_FONT_FAMILIES:
        layer_header_font_family = ""
    layer_header_font_size = _parse_optional_font_size(request.form.get("layer_header_font_size"))
    layer_header_use_custom_color = bool(request.form.get("layer_header_use_custom_color"))
    layer_header_text_color = _parse_hex_color(
        request.form.get("layer_header_text_color"), default=(26, 26, 26)
    )
    layer_header_glow = bool(request.form.get("layer_header_glow"))
    layer_header_glow_color = _parse_hex_color(
        request.form.get("layer_header_glow_color"), default=(255, 255, 255)
    )
    layer_header_glow_size = _parse_glow_size(request.form.get("layer_header_glow_size"))
    layer_header_glow_opacity = _parse_glow_opacity(request.form.get("layer_header_glow_opacity"))
    layer_header_align = _parse_align(request.form.get("layer_header_align"), default="left")
    layer_header_background = bool(request.form.get("layer_header_background"))
    layer_header_background_color = _parse_hex_color(
        request.form.get("layer_header_background_color"), default=(0, 0, 0)
    )
    layer_header_background_opacity = _parse_glow_opacity(
        request.form.get("layer_header_background_opacity")
    )

    layer_description_text = (request.form.get("layer_description_text") or "").strip() or None
    # By default the description override matches whatever font family,
    # size, and color the PSD's own "description" text layer was set to
    # in Photoshop (see get_psd_layer_text_style()) -- these three fields
    # let the user override any of that per request. "" for family means
    # "match the PSD", same idea as leaving font size blank; the color
    # picker always has *some* value (browsers can't leave <input
    # type=color> blank), so a separate checkbox marks whether to actually
    # use it instead of the PSD's own color.
    layer_description_font_family = (request.form.get("layer_description_font_family") or "").strip()
    if layer_description_font_family not in VALID_FONT_FAMILIES:
        layer_description_font_family = ""
    layer_description_font_size = _parse_optional_font_size(request.form.get("layer_description_font_size"))
    layer_description_use_custom_color = bool(request.form.get("layer_description_use_custom_color"))
    layer_description_text_color = _parse_hex_color(
        request.form.get("layer_description_text_color"), default=(26, 26, 26)
    )
    layer_description_glow = bool(request.form.get("layer_description_glow"))
    layer_description_glow_color = _parse_hex_color(
        request.form.get("layer_description_glow_color"), default=(255, 255, 255)
    )
    layer_description_glow_size = _parse_glow_size(request.form.get("layer_description_glow_size"))
    layer_description_glow_opacity = _parse_glow_opacity(
        request.form.get("layer_description_glow_opacity")
    )
    layer_description_align = _parse_align(request.form.get("layer_description_align"), default="left")
    layer_description_background = bool(request.form.get("layer_description_background"))
    layer_description_background_color = _parse_hex_color(
        request.form.get("layer_description_background_color"), default=(0, 0, 0)
    )
    layer_description_background_opacity = _parse_glow_opacity(
        request.form.get("layer_description_background_opacity")
    )

    # The CTA layer's own text override. No position field, unlike the
    # hero-image tool's CTA: a template's button sits in the box its
    # designer drew, and that box is what gets filled.
    layer_cta_text = (request.form.get("layer_cta_text") or "").strip() or None
    layer_cta_font_family = (request.form.get("layer_cta_font_family") or "").strip()
    if layer_cta_font_family not in VALID_FONT_FAMILIES:
        layer_cta_font_family = "sans"
    layer_cta_font_size = _parse_optional_font_size(request.form.get("layer_cta_font_size"))
    layer_cta_button_color = _parse_hex_color(
        request.form.get("layer_cta_button_color"), default=(0, 87, 184)
    )
    layer_cta_text_color = _parse_hex_color(
        request.form.get("layer_cta_text_color"), default=(255, 255, 255)
    )
    layer_cta_glow = bool(request.form.get("layer_cta_glow"))
    layer_cta_glow_color = _parse_hex_color(
        request.form.get("layer_cta_glow_color"), default=(255, 255, 255)
    )
    layer_cta_glow_size = _parse_glow_size(request.form.get("layer_cta_glow_size"))
    layer_cta_glow_opacity = _parse_glow_opacity(request.form.get("layer_cta_glow_opacity"))

    layer_image_overrides: dict = {}  # {"logo"/"cta"/"product"/"background": Image.Image}
    layer_upload_paths: dict = {}  # {field_name: Path} -- for form_state.json, see below
    for layer_name, field_name in (
        ("logo", "layer_logo_image"),
        ("cta", "layer_cta_image"),
        ("product", "layer_product_image"),
        ("background", "layer_background_image"),
    ):
        layer_file = request.files.get(field_name)
        layer_fresh = layer_file is not None and bool(layer_file.filename)
        if layer_fresh:
            if not _allowed(layer_file.filename, ALLOWED_LAYER_IMAGE_EXTENSIONS):
                flash(
                    f"'{layer_file.filename}' isn't a supported file type for the {layer_name} layer update. "
                    "Accepted: " + ", ".join(ALLOWED_LAYER_IMAGE_EXTENSIONS)
                )
                return redirect(url_for("index"))
            layer_path = _save_upload(layer_file, uploads_dir)
        elif request.form.get(f"{field_name}_clear"):
            # The (x) next to a carried-forward image. Without an explicit
            # signal there'd be no way to take one back off: a file input
            # can't be emptied on the user's behalf, so "left blank" has
            # to keep meaning "keep what's there".
            layer_path = None
        else:
            layer_path = _carry_forward_upload(field_name, uploads_dir, prior_job_dir, prior_form_state)
        if layer_path is None:
            continue
        layer_upload_paths[field_name] = layer_path
        try:
            layer_image = Image.open(layer_path).convert("RGBA")
        except Exception as exc:
            flash(f"Couldn't read the {layer_name} layer update image: {exc}")
            return redirect(url_for("index"))
        # Every layer-update image gets the same best-effort background
        # removal -- logo, CTA image, and product image are all commonly
        # exported flat (a solid background behind the mark/product)
        # rather than as a proper cutout, and a flat rectangle looks wrong
        # composited into any of these layers, not just the logo. Not for
        # "background" itself, though -- that upload IS the intended
        # full-frame content, not a cutout with an unwanted backdrop to
        # strip away.
        if layer_name != "background":
            layer_image = auto_transparent_background(layer_image)
        layer_image_overrides[layer_name] = layer_image

    # The uploaded content PSD's own layers become overrides for every
    # other template size -- the point of the quick-campaign field is
    # "upload one flagship PSD and get the campaign", which means the
    # other sizes have to actually take on its artwork instead of
    # rendering from their saved templates untouched. A layer the user
    # uploaded by hand above wins; this only fills the gaps.
    propagated_layer_names = set()
    if upload_ai_image is not None and "background" not in layer_image_overrides:
        # The generated artwork reaches the templates the same way an
        # uploaded content PSD's own background layer does -- as a
        # background override, fitted to each size's background box.
        #
        # Set before the content PSD's own layers are propagated below,
        # and that loop skips any layer already overridden, so the
        # generated backdrop wins over the PSD's. Deliberate: ticking the
        # generator with a PSD uploaded can only mean "keep this design,
        # change the backdrop". A background image uploaded by hand
        # outranks both -- an explicit file beats a generated one.
        layer_image_overrides["background"] = upload_ai_image.convert("RGBA")
        layer_upload_paths["layer_background_image"] = upload_ai_path
    if content_psd_provided:
        for layer_name, layer_image in _content_psd_layer_images(content_psd_path).items():
            if layer_name in layer_image_overrides:
                continue
            layer_image_overrides[layer_name] = layer_image
            propagated_layer_names.add(layer_name)

    # For a templated campaign, default_templates/ is the source of
    # truth: the batch is exactly those sizes plus anything explicitly
    # uploaded on this request (a size-specific PSD template row, or the
    # content PSD itself), and the "Output sizes"/"Custom sizes"
    # selections are dropped so the preview count always reflects what's
    # in the folder. Everywhere else those selections are what drive the
    # batch, exactly as they always did.
    if default_templates:
        sizes = sorted(set(size_templates.keys()))
    else:
        sizes = sorted(set(sizes) | set(size_templates.keys()))

    # AI-generated hero image, if the box was checked and there's an
    # actual gap for it to fill (a hero image was uploaded, or every
    # requested size already has a matching PSD template -- either way,
    # nothing to generate). Generated once, reused for every size that
    # needs it, exactly like an uploaded hero image would be -- saved
    # into uploads/ as hero_path so nothing downstream needs to treat it
    # any differently, including the Edit page's carry-forward.
    if not hero_provided and ai_hero_enabled and any((w, h) not in size_templates for w, h in sizes):
        prompt = ai_hero_prompt or (
            f"professional studio product photo of {product_name or headline or 'the product'}, "
            "clean background"
        )
        try:
            generated_image = get_provider(ai_hero_provider).generate(prompt)
            background_notes.append(
                f"Hero image generated with AI ({ai_hero_provider}) -- prompt: \"{prompt}\"."
            )
        except ImageProviderError as exc:
            # Mirrors src/pipeline.py's own resilience (never let a flaky
            # free API turn into a hard failure) -- falls back to the
            # offline placeholder generator and says so plainly, so it
            # reads as "this specific provider had a bad moment," not as
            # a silent quality regression.
            generated_image = MockImageProvider().generate(prompt)
            background_warnings.append(
                f"Hero image: the '{ai_hero_provider}' AI provider failed ({exc}) -- used the offline "
                f"placeholder generator instead. Prompt: \"{prompt}\"."
            )
        hero_path = uploads_dir / AI_GENERATED_HERO_FILENAME
        generated_image.save(hero_path)
        hero_provided = True

    missing_sizes = [
        (width, height)
        for width, height in sizes
        if (width, height) not in size_templates and not hero_provided
    ]
    if missing_sizes:
        # Reaching this means the AI-hero fallback above either wasn't
        # checked or wasn't applicable -- point directly at that checkbox
        # rather than leaving "upload something" as the only way forward,
        # since it's the one-click fix for exactly this situation.
        missing_labels = ", ".join(size_label(width, height) for width, height in missing_sizes)
        flash(
            f"These sizes need either a hero image or a matching PSD template: {missing_labels}. "
            "Or check \"Generate a hero image with AI\" under the Hero image field below to have "
            "one generated automatically instead."
        )
        return redirect(url_for("index"))

    video_frame_seconds = None
    if hero_provided and hero_path.suffix.lower() in VIDEO_EXTENSIONS:
        raw_seconds = (request.form.get("video_frame_seconds") or "").strip()
        if raw_seconds:
            try:
                video_frame_seconds = float(raw_seconds)
            except ValueError:
                flash(f"'{raw_seconds}' isn't a valid number of seconds -- using the middle of the video instead.")

    hero_image = None
    if hero_provided:
        try:
            hero_image = open_as_rgb(hero_path, frame_seconds=video_frame_seconds)
        except ValueError as exc:
            flash(str(exc))
            return redirect(url_for("index"))

    logo_position = request.form.get("logo_position", "top-right")
    if logo_position not in VALID_LOGO_POSITIONS:
        logo_position = "top-right"
    logo_scale = _parse_percent(
        request.form.get("logo_scale"), default=DEFAULT_LOGO_SCALE_PERCENT, min_value=4, max_value=60
    ) / 100.0
    logo_opacity = _parse_percent(
        request.form.get("logo_opacity"), default=DEFAULT_LOGO_OPACITY_PERCENT, min_value=0, max_value=100
    ) / 100.0
    logo_offset_x = _parse_signed_int(request.form.get("logo_offset_x"), default=0)
    logo_offset_y = _parse_signed_int(request.form.get("logo_offset_y"), default=0)
    logo_image = None
    logo_path = None
    logo_file = request.files.get("logo")
    logo_fresh = logo_file is not None and bool(logo_file.filename)
    if logo_fresh:
        if not _allowed(logo_file.filename, ALLOWED_LOGO_EXTENSIONS):
            flash(
                f"Logo file '{logo_file.filename}' isn't a supported type -- use PNG or WEBP "
                "(needs transparency to composite cleanly)."
            )
            return redirect(url_for("index"))
        logo_path = _save_upload(logo_file, uploads_dir)
    else:
        logo_path = _carry_forward_upload("logo", uploads_dir, prior_job_dir, prior_form_state)
    if logo_path is not None:
        logo_image = Image.open(logo_path).convert("RGBA")

    badge_position = request.form.get("badge_position", "top-right")
    if badge_position not in VALID_BADGE_POSITIONS:
        badge_position = "top-right"
    badge_scale = _parse_percent(
        request.form.get("badge_scale"), default=DEFAULT_BADGE_SCALE_PERCENT, min_value=5, max_value=100
    ) / 100.0
    badge_opacity = _parse_percent(
        request.form.get("badge_opacity"), default=DEFAULT_BADGE_OPACITY_PERCENT, min_value=0, max_value=100
    ) / 100.0
    badge_image_obj = None
    badge_path = None
    badge_file = request.files.get("badge_image")
    badge_fresh = badge_file is not None and bool(badge_file.filename)
    if badge_fresh:
        if not _allowed(badge_file.filename, ALLOWED_BADGE_EXTENSIONS):
            flash(
                f"Badge file '{badge_file.filename}' isn't a supported type. Accepted: "
                + ", ".join(ALLOWED_BADGE_EXTENSIONS)
            )
            return redirect(url_for("index"))
        badge_path = _save_upload(badge_file, uploads_dir)
    else:
        badge_path = _carry_forward_upload("badge_image", uploads_dir, prior_job_dir, prior_form_state)
    if badge_path is not None:
        badge_image_obj = Image.open(badge_path).convert("RGBA")

    # Trademark/brand-name check -- optional bonus, not a requirement:
    # OCRs each uploaded image and flags any well-known brand name it
    # finds printed as text in it. Purely a warning (shown in red on the
    # results page, like a missing brand color), never blocks generation,
    # and silently finds nothing if the system doesn't have the
    # `tesseract` OCR binary installed -- see check_trademark_text().
    trademark_images = [("Hero image", hero_image), ("Logo", logo_image), ("Badge", badge_image_obj)]
    trademark_images += [
        (f"{layer_name.replace('_', ' ').title()} update image", layer_image)
        for layer_name, layer_image in layer_image_overrides.items()
    ]
    for image_label, image_obj in trademark_images:
        if image_obj is None:
            continue
        found_brands = check_trademark_text(image_obj)
        if found_brands:
            background_warnings.append(
                f"{image_label}: looks like it may contain the brand name "
                + ", ".join(found_brands)
                + " as text -- worth a second look before this goes out."
            )

    creatives = []
    for width, height in sizes:
        background_image = size_templates.get((width, height), hero_image)
        is_template_size = (width, height) in size_templates
        # Only set for the render_creative() path below -- a PSD template
        # size is already a complete, hand-built creative (see
        # is_template_size below), so there's nothing generic to re-export
        # as an editable layer stack for it.
        psd_filename = None
        # The unmodified source template, kept beside the rendered PSD
        # whenever a layer override forced a rebuild. The rebuild is a
        # stack of rasterized pixel layers -- psd-tools can only author
        # those (create_pixel_layer is its one layer-writing API, and
        # TypeLayer.text has no setter), so the header/description in a
        # rebuilt file are pictures of words, not Photoshop type layers.
        # The source template still has the real, live type layers, so
        # offering it alongside is the difference between "you can move
        # this text" and "you can retype this text".
        source_psd_filename = None
        if (width, height) in psd_templates:
            background_notes.append(
                f"{size_label(width, height)} used your uploaded PSD template as-is -- "
                "no header/message/logo/badge/CTA overlay is added on top of a template."
            )
        elif (width, height) in default_templates:
            background_notes.append(
                f"{size_label(width, height)} used the saved default template as-is -- "
                "no header/message/logo/badge/CTA overlay is added on top of a template."
            )

        if is_template_size:
            # Offer the original uploaded/saved template PSD itself as
            # this size's "Download PSD" -- a real, fully Photoshop-
            # editable multi-layer file, already using the exact layer
            # names (logo/description/product/cta) the app's own upload
            # flow expects, since that's what got it recognized as a
            # template in the first place. This is the file as uploaded,
            # not a rebuild with the description/logo/CTA/product
            # overrides below baked in -- get_psd_layer_boxes() and
            # friends read a template's layer *names and boxes*
            # reliably, but not reliably enough to safely reconstruct a
            # new multi-layer PSD with edited pixel content in each
            # named layer, so re-packaging overrides into a fresh PSD
            # here isn't attempted. Best-effort like the layered-PSD
            # export below: a copy failure never blocks an otherwise-
            # successful render, it just leaves no PSD link for this size.
            psd_source_path = size_template_paths.get((width, height))
            if psd_source_path is not None:
                try:
                    psd_candidate_filename = f"{file_name_prefix}_{size_label(width, height)}.psd"
                    shutil.copy(psd_source_path, job_dir / psd_candidate_filename)
                    psd_filename = psd_candidate_filename
                except Exception:
                    psd_filename = None

            # A PSD template is a complete, already-designed creative for
            # this exact size (headline, logo, CTA, etc. all baked into
            # its flattened pixels by whoever built it in Photoshop) --
            # drawing a second, generic overlay on top of that would
            # cover/duplicate work the template already did. Just fit it
            # to the exact canvas (it should already match; this is a
            # safety net for a template whose own pixel size doesn't
            # exactly equal its filename-derived size) and use it as-is.
            if fit_mode == "contain":
                final_image = resize_to_contain(background_image, (width, height))
            else:
                final_image = center_crop_to_ratio(background_image, (width, height))

            # PSD-section layer overrides -- swap the description text,
            # logo, CTA image, and/or product image, each applied at THIS
            # size's own layer bounding box (a different position/scale
            # per size, since every template size has its own layout).
            # A size whose PSD doesn't have a given named layer just skips
            # that one override rather than erroring the whole request.
            if (layer_header_text or layer_description_text or layer_image_overrides) and (width, height) in size_template_paths:
                psd_path_for_size = size_template_paths.get((width, height))
                layer_boxes = get_psd_layer_boxes(psd_path_for_size)
                applied_layers = []
                # Each overridden layer's own isolated RGBA patch (box-
                # positioned, transparent everywhere else) -- keyed by
                # lowercased layer name, e.g. "background"/"logo"/
                # "header". Populated below as each override is applied,
                # and reused when building the downloadable layered PSD
                # (see the "if applied_layers" block further down) so
                # that export shows exactly the new content on its own
                # layer instead of a single flattened image.
                export_layer_patches: dict = {}
                # A pristine copy of this size's template, exactly as it
                # renders with zero overrides -- i.e. Pillow's own
                # embedded/flattened PSD composite (the same source used
                # everywhere else in this app), not a psd-tools
                # recomposite. psd-tools is the only way to toggle a
                # layer's visibility (Pillow can't isolate layers at all
                # -- see get_psd_layer_boxes()'s docstring), but its own
                # from-scratch re-render of text/effects can come out
                # visibly different from Photoshop's own flattened
                # preview (different font hinting, missing layer
                # effects, etc.). So psd-tools is used below only to work
                # out *where* other layers draw (an alpha mask), and the
                # actual pixels restored always come from this pristine
                # copy -- guaranteeing a background-only change leaves
                # everything else pixel-identical to an unedited render.
                pristine_final_image = final_image.copy()

                # A layer box is read straight from the PSD's own pixel
                # space (see get_psd_layer_boxes()), but `final_image` is
                # `background_image` after being fit to (width, height)
                # via resize_to_contain()/center_crop_to_ratio() -- a
                # no-op only when the PSD's own saved canvas size exactly
                # equals this size's (width, height). A template PSD
                # normally IS saved at its nominal size, but a
                # user-uploaded PSD assigned to a size slot by filename
                # can be off by a handful of pixels (e.g. a 728x480 file
                # used for the "720x480" slot) -- without remapping, every
                # layer box silently drifts from where that content
                # actually lands in final_image, which shows up as things
                # like a background patch missing a layer's true edge by
                # a few pixels and leaving a sliver of the PSD's original
                # content (e.g. placeholder text) visible right at the
                # edge of an otherwise-correct override.
                psd_canvas_size = get_psd_canvas_size(psd_path_for_size) if psd_path_for_size else None
                if psd_canvas_size:
                    layer_boxes = {
                        name: map_box_through_fit(box, psd_canvas_size, (width, height), fit_mode)
                        for name, box in layer_boxes.items()
                    }

                def _fit_rgba_like_final_image(rgba_image):
                    # Map a full-canvas RGBA PSD composite (psd-tools'
                    # own coordinate space) into final_image's own
                    # coordinate space with the *same* transform that
                    # produced final_image itself (see the comment above
                    # psd_canvas_size) -- center_crop_to_ratio() preserves
                    # alpha untouched (it's just crop+resize), but
                    # resize_to_contain() always flattens to RGB for its
                    # letterboxed-blur look, which would silently throw
                    # away the transparency this needs, so "contain" is
                    # handled by hand here instead: scale to fit,
                    # centered, onto a fully transparent canvas the
                    # target size.
                    if rgba_image.size == final_image.size:
                        return rgba_image
                    if fit_mode == "contain":
                        target_w, target_h = final_image.size
                        src_w, src_h = rgba_image.size
                        scale = min(target_w / src_w, target_h / src_h)
                        new_w = max(int(round(src_w * scale)), 1)
                        new_h = max(int(round(src_h * scale)), 1)
                        fitted = rgba_image.resize((new_w, new_h), Image.LANCZOS)
                        canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
                        offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
                        canvas.alpha_composite(fitted, offset)
                        return canvas
                    return center_crop_to_ratio(rgba_image, final_image.size)

                # Was "background" also uploaded in this same request? If
                # so it's processed first (see the reordering below) and
                # this box's *own* clean-up needs to know that, so it
                # doesn't reintroduce the old (just-replaced) background
                # underneath whatever it's about to redraw -- see the
                # branch inside _clean_layer_box() just below.
                background_replaced_this_request = "background" in layer_image_overrides and not (
                    "background" in propagated_layer_names and (width, height) == content_psd_size
                )
                # Every layer actually being replaced on THIS size. The
                # masked restore in _clean_layer_box() below reaches back
                # into the original composite, so it has to know which
                # layers are no longer supposed to come from there.
                overridden_layer_names = {
                    name
                    for name in layer_image_overrides
                    if not (name in propagated_layer_names and (width, height) == content_psd_size)
                }
                # final_image as it stood with the new background painted
                # in but before every other layer was composited back on
                # top -- the "clear the whole box" case below needs a
                # backdrop-only image to wipe to, and once the background
                # has been replaced this request the PSD's own backdrop is
                # the wrong one to use. Stays None unless that happens.
                background_only_image = None

                def _clean_layer_box(target_box, layer_name, full_box=False):
                    # Patch in the PSD's own true pixels for this box with
                    # the named layer hidden (see get_psd_layer_background())
                    # before drawing anything new there -- this is real
                    # background data straight from the file (e.g. the
                    # ad's actual gradient/photo), not a guess, so
                    # whatever the new content doesn't fully cover reads
                    # correctly with zero leftover trace of the old layer.
                    nonlocal final_image
                    if psd_path_for_size is None:
                        return
                    if full_box and background_replaced_this_request:
                        # Same "wipe the whole box" intent as below, but
                        # against the background this request just put
                        # there rather than the PSD's original one.
                        if background_only_image is None:
                            return
                        final_image.paste(background_only_image.crop(target_box), target_box[:2])
                        return
                    if background_replaced_this_request and layer_name != "background":
                        # The background this request started with is
                        # already gone (replaced further up in this same
                        # loop). Hiding just `layer_name` here (like the
                        # plain case below) would recompose against the
                        # PSD's *original* background and paste that back
                        # in -- undoing part of the background override.
                        # Hiding `layer_name` *and* "background" together
                        # instead marks both transparent in the mask, so
                        # pasting pristine_final_image through it (see the
                        # comment where that's captured, above) restores
                        # every OTHER layer's real pixels and leaves
                        # `layer_name`'s own box (and the new background
                        # elsewhere) exactly as final_image already has
                        # them -- ready for apply_layer_image_override()
                        # to draw the new content into a clean box.
                        # Wipe the box back to the new background FIRST.
                        # The background step above restored every
                        # original foreground layer on top of the new
                        # backdrop, this layer's own old pixels included,
                        # and the masked restore below deliberately
                        # doesn't touch this layer's own area -- so
                        # without this the old artwork survives inside
                        # the box and the new override just draws over
                        # it. That showed up as the previous template's
                        # plate framing a smaller replacement cutout,
                        # worst in the extreme aspect ratios where a
                        # fitted cutout leaves the most margin.
                        if background_only_image is not None:
                            final_image.paste(
                                background_only_image.crop(target_box), target_box[:2]
                            )
                        # Hide every overridden layer, not just this one
                        # and the background. The restore below pulls
                        # from the ORIGINAL composite, so leaving another
                        # override's layer visible in the mask paints its
                        # old artwork back over the replacement drawn a
                        # moment ago -- with several overrides in play the
                        # last one processed would resurrect all the ones
                        # before it. That was the old template's plate
                        # reappearing behind a new product cutout.
                        mask_source = get_psd_layer_foreground(
                            psd_path_for_size,
                            sorted({layer_name, "background"} | overridden_layer_names),
                        )
                        if mask_source is None:
                            return
                        mask_source = _fit_rgba_like_final_image(mask_source)
                        final_image.paste(pristine_final_image, mask=mask_source.split()[3])
                        return
                    clean_bg = None
                    if full_box:
                        # Replace, don't overprint. Hiding just this one
                        # layer is enough when it's the only thing in its
                        # box, but a text layer's box routinely overlaps
                        # other artwork -- a header box parked across the
                        # logo, say -- and hiding only the text layer
                        # leaves that artwork sitting under the new words.
                        # Compositing everything except the background
                        # away gives the box's true backdrop to wipe to,
                        # so the new text owns the space the old text had.
                        clean_bg = get_psd_backdrop(psd_path_for_size)
                    if clean_bg is None:
                        clean_bg = get_psd_layer_background(psd_path_for_size, layer_name)
                    if clean_bg is None:
                        return
                    if clean_bg.size != final_image.size:
                        # Fit clean_bg into final_image's coordinate space
                        # with the *same* transform that produced
                        # final_image itself (see above) -- a plain
                        # stretch-resize here would use a different
                        # transform than center_crop_to_ratio()'s
                        # crop-then-resize whenever the aspect ratios
                        # don't exactly match, subtly misaligning the
                        # patch from the (now correctly mapped) target_box
                        # it's about to be cropped/pasted with.
                        if fit_mode == "contain":
                            clean_bg = resize_to_contain(clean_bg, final_image.size)
                        else:
                            clean_bg = center_crop_to_ratio(clean_bg, final_image.size)
                    final_image.paste(clean_bg.crop(target_box), target_box[:2])

                # Process "background" first, no matter which order the
                # form fields were uploaded in -- a background override
                # fills its (whole-canvas) box completely, which would
                # otherwise wipe out any logo/cta/product override this
                # same request just drew if background ran after them.
                # Running it first, then restoring every other PSD layer
                # on top (see get_psd_layer_foreground()), means the
                # logo/cta/product/text steps below land on the *new*
                # background exactly like they would on the original one.
                ordered_layer_names = sorted(
                    layer_image_overrides.keys(), key=lambda name: name != "background"
                )
                for layer_name in ordered_layer_names:
                    override_image = layer_image_overrides[layer_name]
                    box = layer_boxes.get(layer_name)
                    if box is None:
                        continue
                    if layer_name in propagated_layer_names and (width, height) == content_psd_size:
                        # This size IS the uploaded PSD -- re-applying its
                        # own layers back onto itself would round-trip
                        # them through a fit/crop for no gain.
                        continue
                    if layer_name == "background":
                        final_image = apply_layer_background_override(final_image, box, override_image)
                        background_only_image = final_image.copy()
                        foreground_mask_source = (
                            get_psd_layer_foreground(psd_path_for_size, layer_name)
                            if psd_path_for_size is not None
                            else None
                        )
                        if foreground_mask_source is not None:
                            foreground_mask_source = _fit_rgba_like_final_image(foreground_mask_source)
                            # Only the alpha channel is used, as a mask --
                            # see the comment above pristine_final_image
                            # for why the *pixels* being restored come
                            # from there instead of this RGBA composite.
                            final_image.paste(pristine_final_image, mask=foreground_mask_source.split()[3])
                        export_layer_patches[layer_name] = apply_layer_background_override(
                            Image.new("RGBA", final_image.size, (0, 0, 0, 0)),
                            box,
                            override_image,
                            keep_alpha=True,
                        )
                    else:
                        _clean_layer_box(box, layer_name)
                        final_image = apply_layer_image_override(final_image, box, override_image)
                        export_layer_patches[layer_name] = apply_layer_image_override(
                            Image.new("RGBA", final_image.size, (0, 0, 0, 0)),
                            box,
                            override_image,
                            keep_alpha=True,
                        )
                    applied_layers.append(layer_name)
                def _apply_text_layer_override(
                    layer_key,
                    text,
                    font_family,
                    font_size,
                    use_custom_color,
                    text_color,
                    glow=False,
                    glow_color=(255, 255, 255),
                    glow_size=GLOW_SIZE_DEFAULT,
                    glow_opacity=100,
                    align="left",
                    show_background=False,
                    background_color=(0, 0, 0),
                    background_opacity=60,
                ):
                    # Shared by every text-layer override (description,
                    # header, ...) -- reads that named layer's own PSD
                    # font settings as the default, lets font
                    # family/size/color be overridden per field, and
                    # shrink-to-fits the text into the layer's box. See
                    # apply_layer_text_override() for the actual
                    # shrink-to-fit/leading-scaling behavior.
                    nonlocal final_image
                    box = layer_boxes.get(layer_key)
                    if box is None:
                        return
                    if not text:
                        # Restyling, not rewriting. Someone who picked a
                        # colour or a font without retyping the words
                        # means "this layer, in that colour" -- so the
                        # layer's own text is read back out of the PSD and
                        # redrawn. Without this the whole override was
                        # gated behind the text box, and changing only the
                        # colour did nothing at all.
                        # visible_only: a layer switched off in Photoshop
                        # has no words to restyle. Without this, styling
                        # one of them redrew hidden text onto every
                        # creative -- and since a hidden header's box
                        # tends to sit over the logo, clearing that box
                        # first wiped most of the logo out with it.
                        text = (
                            (get_psd_text_layers(psd_path_for_size, visible_only=True) or {}).get(layer_key)
                            if psd_path_for_size
                            else None
                        )
                        if not text:
                            return
                    # full_box: a text override replaces what was in the
                    # box rather than printing over it -- see
                    # _clean_layer_box() for why a text layer needs this
                    # and an image layer doesn't.
                    _clean_layer_box(box, layer_key, full_box=True)
                    psd_text_style = (
                        get_psd_layer_text_style(psd_path_for_size, layer_key)
                        if psd_path_for_size is not None
                        else None
                    ) or {}
                    if not psd_text_style:
                        # Surfaced on the results page rather than just
                        # silently falling back -- if this shows up
                        # unexpectedly (the PSD clearly has a real text
                        # layer with this name), that's the signal
                        # something's off in *this* environment specifically
                        # (e.g. psd-tools missing/outdated here vs. wherever
                        # this was last verified), not a text-sizing bug.
                        background_notes.append(
                            f"{size_label(width, height)}: {layer_key} -- couldn't read this "
                            "template's own font settings from the PSD (font/size/color/leading "
                            "not read from a real text layer) -- using autofit sizing instead."
                        )
                    effective_family = font_family or psd_text_style.get("family") or "sans"
                    effective_bold = psd_text_style.get("bold", True)
                    # A user-typed font size wins as the *ceiling* for the
                    # same shrink-to-fit search the PSD's own font size
                    # otherwise drives -- see apply_layer_text_override()'s
                    # `exact_font_size` param. It's still just a ceiling,
                    # not a literal demand: the text is always shrunk
                    # further if it doesn't actually fit the box, so an
                    # explicit size can never push text past the box's
                    # edges (see the "clamped" note appended below when
                    # that happens, so it's visible rather than a silent
                    # "why didn't my font size change anything").
                    exact_font_size = font_size
                    ceiling_font_size = None if exact_font_size else psd_text_style.get("font_size")
                    # The PSD's own leading (line spacing) is scaled
                    # proportionally against the PSD's own font size
                    # (leading_reference_size) to whatever size actually
                    # ends up rendering -- see apply_layer_text_override()'s
                    # `leading` / `leading_reference_size` params. This
                    # still applies even when the user typed an explicit
                    # font size: the PSD's *relative* line spacing is
                    # still the best estimate available, and tracks the
                    # requested size far better than a generic
                    # ~1.2x-of-font-size approximation would.
                    effective_leading = psd_text_style.get("line_height")
                    leading_reference_size = psd_text_style.get("font_size")
                    if use_custom_color:
                        effective_color = text_color
                    else:
                        effective_color = psd_text_style.get("color", (26, 26, 26))
                    text_debug: dict = {}
                    final_image = apply_layer_text_override(
                        final_image,
                        box,
                        text,
                        text_color=effective_color,
                        font_family=effective_family,
                        font_size=ceiling_font_size,
                        exact_font_size=exact_font_size,
                        bold=effective_bold,
                        leading=effective_leading,
                        leading_reference_size=leading_reference_size,
                        glow=glow,
                        glow_color=glow_color,
                        glow_size=glow_size,
                        glow_opacity=glow_opacity,
                        align=align,
                        show_background=show_background,
                        background_color=background_color,
                        background_opacity=background_opacity,
                        debug=text_debug,
                    )
                    # Same call again, but onto a transparent canvas with
                    # keep_alpha=True -- isolates just the new glyphs as
                    # their own layer (see export_layer_patches above),
                    # for the downloadable PSD.
                    export_layer_patches[layer_key] = apply_layer_text_override(
                        Image.new("RGBA", final_image.size, (0, 0, 0, 0)),
                        box,
                        text,
                        text_color=effective_color,
                        font_family=effective_family,
                        font_size=ceiling_font_size,
                        exact_font_size=exact_font_size,
                        bold=effective_bold,
                        leading=effective_leading,
                        leading_reference_size=leading_reference_size,
                        glow=glow,
                        glow_color=glow_color,
                        glow_size=glow_size,
                        glow_opacity=glow_opacity,
                        align=align,
                        show_background=show_background,
                        background_color=background_color,
                        background_opacity=background_opacity,
                        keep_alpha=True,
                    )
                    applied_layers.append(layer_key)
                    background_notes.append(
                        f"{size_label(width, height)}: {layer_key} debug -- "
                        f"read from PSD: {psd_text_style or 'none'} | "
                        f"used: {text_debug.get('font_size')}px {text_debug.get('family')} "
                        f"(bold={text_debug.get('bold')}), leading {text_debug.get('line_height')}px, "
                        f"{text_debug.get('lines')} line(s), color {effective_color}."
                    )
                    if text_debug.get("clamped"):
                        # The text is always shrunk to actually fit the
                        # box (see apply_layer_text_override()) -- if that
                        # meant using less than what was requested (the
                        # PSD's own size, or an explicit override), say so
                        # explicitly here rather than leaving a "why
                        # didn't my font size change anything" silently
                        # unanswered.
                        background_notes.append(
                            f"{size_label(width, height)}: {layer_key} -- requested "
                            f"{text_debug.get('requested_font_size')}px didn't fit this size's "
                            f"box at this text length -- used {text_debug.get('font_size')}px "
                            "instead so the text stays inside the box."
                        )

                # Any of these on their own is a reason to redraw the
                # layer: new words, a colour, a family, a size.
                if (
                    layer_header_text
                    or layer_header_glow
                    or layer_header_background
                    or layer_header_use_custom_color
                    or layer_header_font_family
                    or layer_header_font_size
                ):
                    _apply_text_layer_override(
                        "header",
                        layer_header_text,
                        layer_header_font_family,
                        layer_header_font_size,
                        layer_header_use_custom_color,
                        layer_header_text_color,
                        glow=layer_header_glow,
                        glow_color=layer_header_glow_color,
                        glow_size=layer_header_glow_size,
                        glow_opacity=layer_header_glow_opacity,
                        align=layer_header_align,
                        show_background=layer_header_background,
                        background_color=layer_header_background_color,
                        background_opacity=layer_header_background_opacity,
                    )
                if (
                    layer_description_text
                    or layer_description_glow
                    or layer_description_background
                    or layer_description_use_custom_color
                    or layer_description_font_family
                    or layer_description_font_size
                ):
                    _apply_text_layer_override(
                        "description",
                        layer_description_text,
                        layer_description_font_family,
                        layer_description_font_size,
                        layer_description_use_custom_color,
                        layer_description_text_color,
                        glow=layer_description_glow,
                        glow_color=layer_description_glow_color,
                        glow_size=layer_description_glow_size,
                        glow_opacity=layer_description_glow_opacity,
                        align=layer_description_align,
                        show_background=layer_description_background,
                        background_color=layer_description_background_color,
                        background_opacity=layer_description_background_opacity,
                    )
                if layer_cta_text and "cta" not in layer_image_overrides:
                    # Skipped when a CTA image was uploaded for this run:
                    # that upload IS the button, and drawing one over it
                    # would bury what the user just supplied.
                    cta_box = layer_boxes.get("cta")
                    if cta_box is not None:
                        _clean_layer_box(cta_box, "cta", full_box=True)
                        cta_kwargs = dict(
                            button_color=layer_cta_button_color,
                            text_color=layer_cta_text_color,
                            font_size=layer_cta_font_size,
                            font_family=layer_cta_font_family,
                            glow=layer_cta_glow,
                            glow_color=layer_cta_glow_color,
                            glow_size=layer_cta_glow_size,
                            glow_opacity=layer_cta_glow_opacity,
                        )
                        final_image = apply_layer_cta_override(
                            final_image, cta_box, layer_cta_text, **cta_kwargs
                        )
                        export_layer_patches["cta"] = apply_layer_cta_override(
                            Image.new("RGBA", final_image.size, (0, 0, 0, 0)),
                            cta_box,
                            layer_cta_text,
                            keep_alpha=True,
                            **cta_kwargs,
                        )
                        applied_layers.append("cta")

                if applied_layers:
                    background_notes.append(
                        f"{size_label(width, height)}: updated layer(s) -- " + ", ".join(applied_layers) + "."
                    )
                    # The "Download PSD" above is a copy of the *original*
                    # uploaded template -- once a layer override actually
                    # changed pixels in final_image (logo/CTA/product/
                    # description), that original copy silently stops
                    # matching what the results page just showed as the
                    # preview. Rebuild it as a real layered PSD instead of
                    # a single flattened image: every layer this request
                    # did NOT touch comes straight from the original file
                    # (see get_psd_layer_stack()), and every layer it DID
                    # touch is swapped for that override's own isolated
                    # RGBA patch (export_layer_patches, built alongside
                    # each override above) -- so the download still opens
                    # in Photoshop with logo/CTA/product/background/
                    # header/description as separate, transparency-intact
                    # layers, not one baked-together image. Falls back to
                    # the single-flattened-layer file only if the original
                    # PSD's layer stack can't be read at all. Best-effort
                    # like the copy above: a failure here just leaves
                    # that last-copied (now-stale) file in place rather
                    # than blocking an otherwise-successful render.
                    try:
                        layer_stack = (
                            get_psd_layer_stack(psd_path_for_size) if psd_path_for_size is not None else None
                        )
                        export_layers = []
                        for name, layer_img in layer_stack or []:
                            key = name.strip().lower()
                            if key in export_layer_patches:
                                export_layers.append((name, export_layer_patches[key]))
                            else:
                                export_layers.append((name, _fit_rgba_like_final_image(layer_img)))
                        if not export_layers:
                            export_layers = [("Background", final_image)]
                        # Saved first: the rebuild below writes over the
                        # copy of the template made further up (same
                        # filename), and that copy is the only file in
                        # this job with live text layers in it.
                        if psd_path_for_size is not None:
                            source_candidate_filename = (
                                f"{file_name_prefix}_{size_label(width, height)}_source-template.psd"
                            )
                            shutil.copy(psd_path_for_size, job_dir / source_candidate_filename)
                            source_psd_filename = source_candidate_filename
                            # Carry a custom text colour into the live
                            # text too. The rendered PSD beside this one
                            # has the colour baked into pixels; here it
                            # stays an editable type layer that simply
                            # opens in the right colour.
                            live_text_colors = {}
                            if layer_header_use_custom_color:
                                live_text_colors["header"] = layer_header_text_color
                            if layer_description_use_custom_color:
                                live_text_colors["description"] = layer_description_text_color
                            if live_text_colors:
                                recoloured = set_type_layer_colors(
                                    job_dir / source_candidate_filename, live_text_colors
                                )
                                if recoloured:
                                    background_notes.append(
                                        f"{size_label(width, height)}: source PSD's live text recoloured -- "
                                        + ", ".join(recoloured)
                                        + "."
                                    )
                        psd_candidate_filename = f"{file_name_prefix}_{size_label(width, height)}.psd"
                        save_layered_psd(
                            export_layers,
                            (width, height),
                            job_dir / psd_candidate_filename,
                            layer_names={},
                        )
                        psd_filename = psd_candidate_filename
                    except Exception:
                        pass
        else:
            # Built once and reused for both calls below so render_creative()
            # (the flattened PNG preview) and render_creative_layers() (the
            # per-size layered PSD download) can never quietly drift apart --
            # see render_creative_layers()'s own docstring for why that
            # matters.
            render_kwargs = dict(
                message=message,
                headline=headline,
                fit_mode=fit_mode,
                logo=logo_image,
                logo_position=logo_position,
                logo_scale=logo_scale,
                logo_opacity=logo_opacity,
                logo_offset_x=logo_offset_x,
                logo_offset_y=logo_offset_y,
                header_text_color=header_text_color,
                header_show_background=header_show_background,
                header_glow=header_glow,
                header_glow_color=header_glow_color,
                header_align=header_align,
                header_font_size=header_font_size,
                message_text_color=message_text_color,
                message_show_background=message_show_background,
                message_glow=message_glow,
                message_glow_color=message_glow_color,
                message_align=message_align,
                message_font_size=message_font_size,
                badge_image=badge_image_obj,
                badge_position=badge_position,
                badge_scale=badge_scale,
                badge_opacity=badge_opacity,
                cta_text=cta_text,
                cta_position=cta_position,
                cta_button_color=cta_button_color,
                cta_text_color=cta_text_color,
                cta_font_size=cta_font_size,
                cta_font_family=cta_font_family,
                cta_glow=cta_glow,
                cta_glow_color=cta_glow_color,
                cta_above_message=cta_above_message,
            )
            final_image, _logo_composited = render_creative(background_image, (width, height), **render_kwargs)

            # A layered PSD download alongside the flattened PNG preview,
            # for this same size -- best-effort: a PSD write failing
            # (a corrupt logo/badge upload triggering some edge case in
            # psd-tools, disk space, etc.) should never fail a render that
            # otherwise already succeeded, so this never blocks or bubbles
            # up to the user -- it just leaves psd_filename as None and the
            # results page simply won't show a PSD link for this size.
            try:
                psd_layers = render_creative_layers(background_image, (width, height), **render_kwargs)
                psd_candidate_filename = f"{file_name_prefix}_{size_label(width, height)}.psd"
                save_layered_psd(psd_layers, (width, height), job_dir / psd_candidate_filename)
                psd_filename = psd_candidate_filename
            except Exception:
                psd_filename = None
        label = size_label(width, height)
        if brand_colors:
            missing_colors = find_missing_brand_colors(
                final_image, brand_colors, tolerance=BRAND_COLOR_MATCH_TOLERANCE
            )
            if missing_colors:
                missing_hex = ", ".join(
                    "#%02x%02x%02x" % color for color in missing_colors
                )
                background_warnings.append(
                    f"{label}: brand color check -- not all brand colors are in this creative "
                    f"({len(missing_colors)} of {len(brand_colors)} missing): {missing_hex}."
                )
        filename = f"{file_name_prefix}_{label}.png"
        out_path = job_dir / filename
        final_image.save(out_path)
        creatives.append(
            {
                "filename": filename,
                "label": label,
                "ratio": ratio_label(width, height),
                "name": size_name(width, height),
                "psd_filename": psd_filename,
                "source_psd_filename": source_psd_filename,
            }
        )

    zip_stem = (
        f"{product_name_slug}_{campaign_label}_creatives"
        if product_name_slug
        else f"{campaign_label}_creatives"
    )
    zip_path = job_dir / f"{zip_stem}.zip"
    # Everything inside the zip is nested under its campaign, and under
    # the product name too when there is one -- so unzipping drops a
    # self-contained "<Product Name>/campaign1/" tree wherever the user
    # extracts to. Several campaigns from the same session can then be
    # unzipped side by side without their same-named sizes colliding,
    # which is the whole reason the campaign is in the path.
    zip_entry_prefix = (
        f"{product_name_slug}/{campaign_label}/" if product_name_slug else f"{campaign_label}/"
    )
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for creative in creatives:
            zf.write(
                job_dir / creative["filename"],
                arcname=f"{zip_entry_prefix}{creative['filename']}",
            )
            # The per-size layered PSD (see render_creative_layers()) --
            # bundling it into the same zip means a bulk download gets
            # the editable file too, not just the flattened PNG, without
            # a separate per-size click for each one.
            if creative.get("psd_filename"):
                zf.write(
                    job_dir / creative["psd_filename"],
                    arcname=f"{zip_entry_prefix}{creative['psd_filename']}",
                )
            # The source template next to it -- the only copy whose
            # header/description are still editable Photoshop type
            # layers rather than rendered pixels (see
            # source_psd_filename where it's set).
            if creative.get("source_psd_filename"):
                zf.write(
                    job_dir / creative["source_psd_filename"],
                    arcname=f"{zip_entry_prefix}{creative['source_psd_filename']}",
                )

    # A copy of the zip somewhere a person can actually find it, without
    # going through the browser's download folder or digging through
    # outputs/web/<random id>/. Re-running the same campaign overwrites
    # its own file rather than accumulating near-identical archives.
    # Best-effort: a failure here (a read-only checkout, say) must never
    # sink a render that already succeeded -- the download button still
    # works either way.
    try:
        DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(zip_path, DOWNLOADS_DIR / zip_path.name)
    except OSError:
        pass

    # Saved so the "Edit" button on the results page (see /edit/<job_id>)
    # can reload this form pre-filled, and so a file field the user
    # doesn't re-upload next time is carried forward as-is instead of
    # being dropped -- see _carry_forward_upload(). Best-effort: editing
    # is a convenience, never something that should fail a render that
    # already succeeded.
    form_state_fields = {name: (request.form.get(name) or "") for name in EDIT_TEXT_FIELD_NAMES}
    # These already have a validated/defaulted Python variable (the raw
    # form field could be missing or invalid) -- prefer that so a radio
    # group's default always round-trips into a real checked option
    # instead of landing on "" and leaving nothing checked.
    form_state_fields.update({
        "fit_mode": fit_mode,
        "header_align": header_align,
        "message_align": message_align,
        "cta_position": cta_position,
        "cta_font_family": cta_font_family,
        "logo_position": logo_position,
        "badge_position": badge_position,
    })
    form_state_fields["sizes"] = selected_presets
    for name in EDIT_CHECKBOX_FIELD_NAMES:
        form_state_fields[name] = bool(request.form.get(name))
    raw_file_paths = {
        "hero_image": hero_path if hero_provided else None,
        "content_psd": content_psd_path if content_psd_provided else None,
        "logo": logo_path,
        "badge_image": badge_path,
    }
    raw_file_paths.update({f"psd_file_{i}": psd_file_paths.get(i) for i in range(1, MAX_PSD_TEMPLATES + 1)})
    raw_file_paths.update(layer_upload_paths)
    form_state_files = {}
    for field_name, saved_path in raw_file_paths.items():
        if saved_path is None:
            continue
        try:
            form_state_files[field_name] = saved_path.relative_to(uploads_dir).as_posix()
        except ValueError:
            continue
    # Which multi-campaign page (if any) this job was generated from, and
    # which campaign card on it -- see _session_index_path()/
    # _load_session_campaigns() and the hidden session_id/campaign_slot
    # fields each campaign card's <form> carries. A session_id missing or
    # blank (an old cached page, or a non-browser client) just means this
    # job won't be grouped with any others on Edit -- never a hard error.
    session_id = (request.form.get("session_id") or "").strip() or uuid.uuid4().hex

    try:
        (job_dir / "form_state.json").write_text(
            json.dumps(
                {
                    "fields": form_state_fields,
                    "files": form_state_files,
                    "session_id": session_id,
                    "campaign_slot": campaign_slot,
                },
                indent=2,
            )
        )
    except OSError:
        pass

    # Best-effort, like the write above: record this job into its
    # session's {slot -> job_id} index so a later Edit on ANY campaign
    # generated alongside it (see _load_session_campaigns()) can bring
    # all of them back, not just this one.
    try:
        session_index_path = _session_index_path(session_id)
        session_index_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            session_state = json.loads(session_index_path.read_text())
        except (OSError, ValueError):
            session_state = {}
        slots = session_state.get("slots") or {}
        slots[str(campaign_slot)] = job_id
        session_state["slots"] = slots
        session_index_path.write_text(json.dumps(session_state, indent=2))
    except OSError:
        pass

    return render_template(
        "result.html",
        job_id=job_id,
        creatives=creatives,
        fit_mode=fit_mode,
        background_notes=background_notes,
        background_warnings=background_warnings,
        build_stamp=BUILD_STAMP,
        product_name=product_name,
        campaign_slot=campaign_slot,
        session_id=session_id,
        session_campaign_count=len(_session_campaign_jobs(session_id)),
        market=market,
        audience=audience,
        campaign_message=campaign_message,
    )


@app.route("/outputs/<job_id>/<path:filename>")
def serve_output(job_id, filename):
    job_id = secure_filename(job_id)
    filename = secure_filename(filename)
    file_path = JOBS_DIR / job_id / filename
    if not file_path.is_file():
        abort(404)
    return send_file(file_path)


@app.route("/uploads/<job_id>/<filename>")
def serve_upload(job_id, filename):
    """Serve one file out of a job's uploads/ folder.

    Browsers refuse to pre-populate a file input, so an image carried
    forward from a previous run has no way to show itself in the form
    it's still active in -- the edit page points a thumbnail here
    instead, which is the difference between "my logo is still set" and
    an input that reads "No file chosen" and looks empty.
    """
    job_id = secure_filename(job_id)
    filename = secure_filename(filename)
    file_path = JOBS_DIR / job_id / "uploads" / filename
    if not file_path.is_file():
        abort(404)
    return send_file(file_path)


@app.route("/download-campaigns/<session_id>")
def download_campaigns(session_id):
    """Every campaign in this session, as one zip of zips.

    Each campaign is its own job with its own zip, so grabbing a whole
    multi-campaign page otherwise means clicking through each results
    page in turn. The per-campaign zips go in unchanged, under a
    `campaigns/` folder -- already-compressed archives, so they're stored
    rather than deflated again.
    """
    session_id = secure_filename(session_id)
    jobs = _session_campaign_jobs(session_id)
    entries = []
    for _slot, job_id in jobs:
        zip_path = next(iter(sorted((JOBS_DIR / job_id).glob("*.zip"))), None)
        if zip_path is not None and zip_path.is_file():
            entries.append(zip_path)
    if not entries:
        abort(404)

    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_STORED) as zf:
        for zip_path in entries:
            zf.write(zip_path, arcname=f"campaigns/{zip_path.name}")
    bundle.seek(0)
    return send_file(
        bundle,
        mimetype="application/zip",
        as_attachment=True,
        download_name="campaigns.zip",
    )


@app.route("/download/<job_id>")
def download(job_id):
    job_id = secure_filename(job_id)
    job_dir = JOBS_DIR / job_id
    zip_path = next(iter(sorted(job_dir.glob("*.zip"))), None)
    if zip_path is None or not zip_path.is_file():
        abort(404)
    return send_file(zip_path, as_attachment=True, download_name=zip_path.name)


@app.route("/download-psd/<job_id>/<filename>")
def download_psd(job_id, filename):
    # A per-size layered PSD (see render_creative_layers()/
    # src/psd_export.py), saved alongside that size's PNG at generate()
    # time. Restricted to .psd specifically -- serve_output() above
    # already serves any file in a job's folder, so this isn't a wider
    # attack surface, just a clearer, download-forced, download_name'd
    # entry point for this one file type.
    job_id = secure_filename(job_id)
    filename = secure_filename(filename)
    if not filename.lower().endswith(".psd"):
        abort(404)
    file_path = JOBS_DIR / job_id / filename
    if not file_path.is_file():
        abort(404)
    return send_file(file_path, as_attachment=True, download_name=filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
