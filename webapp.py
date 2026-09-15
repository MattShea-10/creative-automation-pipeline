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
import math
import json
from fractions import Fraction
import os
import random
import sys
import threading
import time
from pathlib import Path

# Packaged (PyInstaller) build: .env, default_templates/, outputs/, downloads/ live beside
# the exe; read-only templates/ and fonts/ ship in the bundle's extraction dir.
FROZEN = bool(getattr(sys, "frozen", False))
BASE_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", BASE_DIR))

try:  # optional, same as src/main.py -- a missing python-dotenv isn't fatal
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None
else:
    # The web app never loaded .env, so a token set there was ignored and the form still
    # reported it unset. Explicit path: dotenv's search starts here, not at the user's .env.
    load_dotenv(BASE_DIR / ".env")
import re
import secrets
import shutil
import tempfile
import uuid
import zipfile

from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_file, url_for
from PIL import Image
from werkzeug.utils import secure_filename

from src.creative_render import render_creative, render_creative_layers
from psd_tools import PSDImage

from src.ad_split import split_ad
from src.psd_export import (
    write_whole_ad_psd,
    hidden_layer_names as psd_hidden_layer_names,
    save_layered_psd,
    save_layered_psd_preserving_type,
    replace_pixel_layers,
    refresh_flattened_preview,
    set_flattened_preview,
    set_shape_layer_style,
    set_type_layer_effects,
    set_type_layer_font_size,
    pair_type_layers_with_pixels,
    set_type_layer_raster,
    set_layer_visibility,
    set_type_layer_text,
    set_type_layer_colors,
)
from src.compliance import check_profanity, check_trademark_text
from src.localization import localize_message, take_last_failure
from src.image_ops import (
    get_psd_text_layers,
    DEFAULT_SIZES,
    SIZE_NAMES,
    VALID_BADGE_POSITIONS,
    VALID_CTA_POSITIONS,
    VALID_FONT_FAMILIES,
    VALID_LOGO_POSITIONS,
    VALID_TEXT_ALIGNMENTS,
    VIDEO_EXTENSIONS,
    apply_layer_background_override,
    apply_layer_image_override,
    apply_layer_cta_override,
    upscale_to_cover,
    apply_layer_styled_lines,
    apply_layer_text_override,
    auto_transparent_background,
    center_crop_to_ratio,
    find_missing_brand_colors,
    get_psd_canvas_size,
    get_psd_backdrop,
    carry_flattened_effects,
    get_psd_layer_background,
    get_psd_layer_boxes,
    get_psd_layer_effect_reach,
    font_covers_text,
    get_psd_layer_foreground,
    get_psd_composite_rgba,
    get_psd_layer_rgba,
    draw_layer_effects,
    _reconstruct_box_background,
    get_psd_group_text,
    get_psd_group_text_box,
    get_psd_layer_stack,
    get_psd_layer_names,
    get_psd_pixel_layer_names,
    get_psd_visible_layers,
    get_psd_buried_layers,
    get_psd_layer_text_style,
    map_box_through_fit,
    find_font_file,
    open_as_rgb,
    parse_size,
    parse_sizes,
    resize_to_contain,
    ratio_label,
    size_label,
    size_name,
)
from src.providers import (
    ALL_PROVIDER_NAMES,
    DEFAULT_PROVIDER_NAME,
    PROVIDER_NAMES,
    ImageProviderError,
    MockImageProvider,
    get_provider,
)
from src.storage import SUPPORTED_EXTENSIONS
from src.text_check import TextCheckResult, detector_description, find_text, ocr_available, remove_text, scrub_text

# Safety valve against nonsense input. The autofit path (size left blank) ignores these.
MIN_CUSTOM_FONT_SIZE = 4
MAX_CUSTOM_FONT_SIZE = 2000

DEFAULT_LOGO_SCALE_PERCENT = 16
DEFAULT_LOGO_OPACITY_PERCENT = 100

DEFAULT_BADGE_SCALE_PERCENT = 35
DEFAULT_BADGE_OPACITY_PERCENT = 100

JOBS_DIR = BASE_DIR / "outputs" / "web"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
_REAL_JOBS_DIR = JOBS_DIR


def _refuse_the_real_jobs_dir_under_test() -> None:
    """Stop a test writing into the real outputs/web.

    A test that posts to /generate without pointing JOBS_DIR at a temp
    folder leaves job and draft folders in the user's own outputs -- ten
    of them turned up looking like real runs, carrying fixture values
    (product "HydroBoost", market "UK"), and they are indistinguishable
    from a batch someone actually made. Silent, so it went unnoticed for
    as long as the tests have existed. Loud from here.
    """
    if app.config.get("TESTING") and JOBS_DIR == _REAL_JOBS_DIR:
        raise RuntimeError(
            "a test is about to write into the real outputs/web -- point "
            "webapp.JOBS_DIR at a temp folder in setUp"
        )

# Templates here apply automatically to their matching output size on every /generate.
# downloads/ is the browsable copy of each run's zip; job folders are named by random id.
DOWNLOADS_DIR = BASE_DIR / "downloads"

DEFAULT_TEMPLATES_DIR = BASE_DIR / "default_templates"
DEFAULT_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
BRIEFS_DIR = BASE_DIR / "briefs"


def _product_folder_name(product_name) -> str:
    """The folder a product's templates live in under default_templates/:
    the product name as typed -- "HydroBoost Sports Drink" -- with only
    what a folder name can't hold taken out (slashes, colons and the
    like) and runs of spaces collapsed. Empty when nothing usable is
    left."""
    name = re.sub(r'[\\/:*?"<>|]+', " ", (product_name or "")).strip()
    name = re.sub(r"\s+", " ", name).strip(". ")
    # A name with no letter or digit in it ("---") is no name.
    if not re.search(r"[A-Za-z0-9]", name):
        return ""
    return name


_brief_campaign_cache: dict = {}


def brief_campaigns_by_product() -> dict:
    """{product name, lowercased: every campaign that briefs it}.

    A product can appear in more than one brief -- HydroBoost Sports
    Drink is in both Winter Glow 2026 and Summer Refresh 2026 -- and each
    one has its own template folder.
    """
    table = {}
    for choice in _brief_choices():
        name = (choice.get("product_name") or "").strip()
        campaign = (choice.get("campaign") or "").strip()
        if not name or not campaign:
            continue
        seen = table.setdefault(name.lower(), [])
        if campaign not in seen:
            seen.append(campaign)
    return table


def brief_campaign_by_product() -> dict:
    """{product name, lowercased: the campaign its brief files it under}.

    Handed to the page so typing a product the briefs know fills Campaign
    in, the way typing a Market fills Copy language. A blank Campaign is
    not a harmless empty field: it sends that product's templates to a
    folder of their own, which is invisible until a restore appears to do
    nothing.

    Only products exactly one brief claims are in here. A product in two
    campaigns has no right answer to fill in, and picking the first file
    alphabetically is worse than filling nothing: it silently points the
    run at the other campaign's templates. Those are offered as a hint
    instead -- see brief_campaigns_by_product().
    """
    return {
        name: campaigns[0]
        for name, campaigns in brief_campaigns_by_product().items()
        if len(campaigns) == 1
    }


def default_campaign_name() -> str:
    """What a brand new card opens with: the campaign last used, else the
    first one the briefs offer. A card created by "Create Campaign" used
    to open blank, which is where a stray campaign-less folder came from."""
    remembered = (_load_preferences().get("campaign_name") or "").strip()
    if remembered:
        return remembered
    for choice in _brief_choices():
        campaign = (choice.get("campaign") or "").strip()
        if campaign:
            return campaign
    return ""


def campaign_for_product(product_name, market=None) -> str:
    """The campaign a brief files this product under, or "".

    A blank Campaign field is not a different product -- it is the same
    product with a field nobody filled in. Left alone it sends that
    product's templates to default_templates/<product>/ while everything
    chosen from a brief goes to default_templates/<campaign>/<product>/,
    so a person can edit one set and restore the other and never see why.
    Looked up from briefs/, cached against the folder's timestamps.

    `market` breaks a tie. HydroBoost Sports Drink is in both Winter Glow
    2026 and Summer Refresh 2026, so its name alone has no answer -- but a
    saved batch also remembers the market it ran for, and Summer Refresh
    is the only one targeting Mexico. Without it the product name alone
    picked whichever brief file sorted first, which reopened a Mexico
    batch pointing at the other campaign's templates.
    """
    product = (product_name or "").strip().lower()
    if not product:
        return ""
    region = (market or "").strip().lower()
    if region:
        matches = []
        for choice in _brief_choices():
            if (choice.get("product_name") or "").strip().lower() != product:
                continue
            if (choice.get("market") or "").strip().lower() != region:
                continue
            campaign = (choice.get("campaign") or "").strip()
            if campaign and campaign not in matches:
                matches.append(campaign)
        if len(matches) == 1:
            return matches[0]
    try:
        stamp = tuple(sorted(
            (p.name, p.stat().st_mtime_ns) for p in BRIEFS_DIR.iterdir()
            if p.suffix.lower() in (".json", ".yaml", ".yml")
        ))
    except OSError:
        return ""
    table = _brief_campaign_cache.get(stamp)
    if table is None:
        # Ambiguous products are left out rather than resolved to
        # whichever brief file sorts first. This feeds the template
        # folder (see _campaign_folder_parts), so a wrong answer here
        # edits one campaign's templates while the person thinks they
        # are editing the other's -- the exact failure this fallback
        # exists to prevent.
        table = {
            name: campaigns[0]
            for name, campaigns in brief_campaigns_by_product().items()
            if len(campaigns) == 1
        }
        _brief_campaign_cache.clear()
        _brief_campaign_cache[stamp] = table
    return table.get(product, "")


def _campaign_folder_parts(product_name, campaign_name=None) -> tuple:
    """The path under default_templates/ for a campaign's product:
    ("Winter Glow 2026", "HydroBoost Sports Drink") when the brief names a
    campaign, ("HydroBoost Sports Drink",) when it doesn't, () for no
    product at all. Each dropdown entry in "From a brief file" is one
    such pair, and gets its own folder."""
    product = _product_folder_name(product_name)
    if not product:
        return ()
    campaign = _product_folder_name(campaign_name)
    if not campaign:
        # Campaign empty: take it from the briefs so the folder, the remembered form and the
        # per-size restore agree instead of opening a second template set for the product.
        campaign = _product_folder_name(campaign_for_product(product_name))
    return (campaign, product) if campaign else (product,)


def product_templates_dir(product_name, create: bool = False, campaign_name=None) -> Path:
    """Where this campaign's product's saved templates are --
    default_templates/<campaign>/<product>/ (or <product>/ alone with no
    campaign name) -- or the shared default_templates/ when there is no
    product. With `create`, a missing folder is made and seeded from the
    backup zip, so it starts from the same templates and keeps its own
    from then on: every drop and every style saved back goes into its
    folder, not the shared one."""
    parts = _campaign_folder_parts(product_name, campaign_name)
    if not parts:
        return DEFAULT_TEMPLATES_DIR
    folder = DEFAULT_TEMPLATES_DIR.joinpath(*parts)
    if create and not folder.is_dir():
        try:
            folder.mkdir(parents=True, exist_ok=True)
            # The backup zip is the master set every Reset returns to;
            # loose shared PSDs are copied only when there is no zip.
            restored, _untouched, error = restore_templates_from_backup(dest_dir=folder)
            if error or not restored:
                for path in sorted(DEFAULT_TEMPLATES_DIR.iterdir()):
                    if path.is_file() and path.suffix.lower() == ".psd" and SIZE_IN_NAME_RE_LOOSE.search(path.name):
                        shutil.copy(path, folder / path.name)
        except OSError:
            return DEFAULT_TEMPLATES_DIR
    return folder


def templates_dir() -> Path:
    """The templates folder for the request in hand: the product's own
    (set by generate() once it knows the product), else the shared one.
    Everything that scans, promotes into or saves back to "the
    templates" reads this rather than DEFAULT_TEMPLATES_DIR directly."""
    try:
        from flask import g, has_request_context
        if has_request_context():
            chosen = getattr(g, "templates_dir", None)
            if chosen is not None:
                return chosen
    except Exception:  # noqa: BLE001
        pass
    return DEFAULT_TEMPLATES_DIR
# Template edits overwrite a hand-made file and can't be undone from the results page, so
# the version being replaced is kept here, timestamped.
TEMPLATE_BACKUPS_DIR = BASE_DIR / "_template_backups"

# Logos need alpha to composite, so only formats that carry it are allowed here.
ALLOWED_LOGO_EXTENSIONS = (".png", ".webp")

# The badge can be a full-frame tint or texture, not just a badge, so JPG is allowed too.
ALLOWED_BADGE_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg")

# For the PSD layer-override fields (logo/CTA/product image) -- same
# tolerance as the badge image, since these can be flat JPGs too.
ALLOWED_LAYER_IMAGE_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg")
# Picture fields: hero, logo/product layer updates, mood board. A PSD is flattened on the way
# in (transparency kept) and only its composite is used; layered designs go in PSD templates.
PICTURE_UPLOAD_EXTENSIONS = ALLOWED_LAYER_IMAGE_EXTENSIONS + (".psd",)

# Each PSD template is the background for its paired size only, as a flattened Pillow preview:
# no layer extraction. Separate from SUPPORTED_EXTENSIONS so the hero field stays image/video.
ALLOWED_PSD_TEMPLATE_EXTENSIONS = (".psd",)
MAX_PSD_TEMPLATES = 12
# Rows open on the form to begin with; the rest sit hidden behind the
# "+ Add another size" button (see index.html).
PSD_TEMPLATE_ROWS_SHOWN = 4

# Quick campaign: this one flagship size is uploaded, the rest come from default_templates/.
# 1920x1080 is what the saved templates are drawn at and the floor for generated backdrops.
CONTENT_PSD_SIZE = (1920, 1080)
CONTENT_PSD_LABEL = f"{CONTENT_PSD_SIZE[0]}x{CONTENT_PSD_SIZE[1]}"

# Every PSD template used in a request needs layers with these exact names, case-insensitive
# (read by get_psd_layer_boxes()).
REQUIRED_PSD_LAYERS = ("logo", "description", "product")

# A quick-campaign content PSD is the flagship design, so these layers restyle every other
# size in the batch. Text is excluded: header/description already apply to all sizes.
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

# Plain fields captured into form_state.json for /edit/<job_id>. The multi-value "sizes"
# checkboxes need request.form.getlist(), and the checkboxes below are stored as booleans.
EDIT_TEXT_FIELD_NAMES = (
    # campaign_name belongs here with the rest of the brief. It was
    # missing, so every run wrote a form_state.json without it: the batch
    # named its files and its template folder after the campaign, and
    # then forgot which one it was. Reopening through /edit handed back a
    # blank Campaign, which _fields_with_campaign() has been guessing at
    # from the product name ever since -- a guess that is wrong whenever
    # two briefs share a product.
    "campaign_name",
    "product_name", "market", "audience", "campaign_message",
    "brand_color_1", "brand_color_2", "brand_color_3",
    "ai_hero_prompt", "ai_hero_provider",
    "upload_ai_prompt", "upload_ai_provider", "upload_ai_speed", "upload_ai_headline",
    "upload_ai_background_style",
    "layer_header_glow_color", "layer_header_glow_size", "layer_header_glow_opacity",
    "layer_header_align", "layer_header_background_color", "layer_header_background_opacity",
    "layer_header_background_blur",
    "layer_description_glow_color", "layer_description_glow_size", "layer_description_glow_opacity",
    "layer_description_align", "layer_description_background_color", "layer_description_background_opacity",
    "layer_description_background_blur",
    "layer_legal_glow_color", "layer_legal_glow_size", "layer_legal_glow_opacity",
    "layer_header_stroke_size", "layer_header_stroke_color",
    "layer_header_shadow_color", "layer_header_shadow_opacity", "layer_header_shadow_distance",
    "layer_header_shadow_spread", "layer_header_shadow_size", "layer_header_shadow_angle",
    "layer_description_shadow_color", "layer_description_shadow_opacity", "layer_description_shadow_distance",
    "layer_description_shadow_spread", "layer_description_shadow_size", "layer_description_shadow_angle",
    "layer_logo_glow_color", "layer_logo_glow_size", "layer_logo_glow_opacity",
    "layer_logo_shadow_color", "layer_logo_shadow_opacity", "layer_logo_shadow_distance",
    "layer_logo_shadow_spread", "layer_logo_shadow_size", "layer_logo_shadow_angle",
    "layer_product_glow_color", "layer_product_glow_size", "layer_product_glow_opacity",
    "layer_product_shadow_color", "layer_product_shadow_opacity", "layer_product_shadow_distance",
    "layer_product_shadow_spread", "layer_product_shadow_size", "layer_product_shadow_angle",
    "layer_logo_stroke_color", "layer_logo_stroke_size",
    "layer_product_stroke_color", "layer_product_stroke_size",
    "layer_description_stroke_size", "layer_description_stroke_color",
    "layer_legal_stroke_size", "layer_legal_stroke_color",
    "layer_cta_stroke_size", "layer_cta_stroke_color", "layer_cta_radius",
    "layer_cta_text_stroke_size", "layer_cta_text_stroke_color",
    "layer_legal_align", "layer_legal_background_color", "layer_legal_background_opacity",
    "layer_legal_background_blur",
    "layer_cta_text", "layer_cta_font_family", "layer_cta_font_size",
    "layer_cta_button_color", "layer_cta_text_color",
    "layer_cta_glow_color", "layer_cta_glow_size", "layer_cta_glow_opacity",
    "header", "description", "custom_sizes",
    "fit_mode", "upload_hero_fit",
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
    "layer_legal_text",
    "layer_legal_font_family", "layer_legal_font_size", "layer_legal_text_color",
    "psd_as_is", "psd_as_is_hero", "copy_language", "psd_make_saved",
) + tuple(f"psd_size_{i}" for i in range(1, MAX_PSD_TEMPLATES + 1))
# Not "background": hiding it leaves a hole, not a cleaner creative. Backdrops get replaced.
HIDEABLE_LAYER_NAMES = ("header", "description", "legal", "logo", "cta", "product")

EDIT_CHECKBOX_FIELD_NAMES = (
    "header_no_background", "header_glow",
    "message_no_background", "message_glow",
    "cta_glow",
    "cta_above_message",
    "update_saved_templates",
    "layer_header_use_custom_color",
    "layer_description_use_custom_color",
    "layer_legal_use_custom_color",
    "brand_color_1_enabled", "brand_color_2_enabled", "brand_color_3_enabled",
    "ai_hero_enabled",
    "upload_custom_hero_enabled",
    "upload_hero_from_template",
    "upload_ai_enabled", "upload_ai_send_references",
    "upload_ai_keep",
    "upload_ai_allow_text",
    "upload_ai_full_ad",
    "layer_header_glow", "layer_header_shadow", "layer_description_shadow",
    "layer_logo_glow", "layer_logo_shadow", "layer_product_glow", "layer_product_shadow",
    "layer_logo_stroke", "layer_product_stroke",
    "layer_description_glow",
    "layer_header_background",
    "layer_description_background",
    "layer_legal_glow",
    "layer_legal_background",
    "layer_cta_glow",
) + tuple(f"layer_{name}_hidden" for name in HIDEABLE_LAYER_NAMES)

# RGB distance under which a pixel matches a brand colour in find_missing_brand_colors().
# The CTA default is also the test for a deliberate colour choice; a group CTA is left alone.
CTA_BUTTON_COLOR_DEFAULT = (0, 87, 184)
# What a CTA label is in every shipped template, so "not this" is the did-anyone-choose test.
CTA_TEXT_COLOR_DEFAULT = (255, 255, 255)

BRAND_COLOR_MATCH_TOLERANCE = 30

# Fixed name for the AI hero fallback; also how the Edit page tells a carried-forward
# generated hero from an uploaded one.
AI_GENERATED_HERO_FILENAME = "ai_generated_hero.png"
# Upload Creative's output. Named apart from the hero so a job using both never overwrites one.
AI_GENERATED_CAMPAIGN_FILENAME = "ai_generated_campaign.png"

# Size can sit anywhere in the name: "tester-728x480.psd", "hero_970x90_v2.psd".
_SIZE_IN_FILENAME_RE = re.compile(r"(\d+)\s*[xX]\s*(\d+)")
SIZE_IN_NAME_RE_LOOSE = re.compile(r"\d{2,5}x\d{2,5}")


# Nothing useful comes of asking a provider for more than this, and the
# wait grows with the pixels.
MAX_GENERATED_EDGE = 2048

# Appended to every generated prompt. Models are worst at faces, hands, crowds and lettering,
# which is what reads as distortion behind a template that has its own product, logo and text.
# No defocus request: "softly out of focus" was obeyed. Compliance is verified, not trusted.
NO_TEXT_CLAUSE = (
    "no text, no words, no lettering, no numbers, no watermarks, no signage, "
    # A brand name invites a wordmark, so the exclusion has to name that case explicitly.
    "no brand name, no wordmark, no logo, no packaging text, no labels"
)

# Used on retry once text has come back. Blunter on purpose: the polite phrasing already failed.
NO_TEXT_ESCALATION = (
    "absolutely no text anywhere in the image, no words, no letters, no numbers, "
    "no captions, no labels, no signs, no posters, no packaging text, no watermark, "
    "a completely textless photographic background"
)

# "no text, no words" in a positive prompt hands the model those tokens (one run painted
# "NEVFT WORDS"). It must also match its own `[excluded: ...]` output, pasted back in.
_NEGATION_TERM = (
    r"(?:text|words?|letters?|lettering|typography|type|writing|captions?|"
    r"labels?|numbers?|digits?|watermarks?|signage|signs?|posters?|logos?|"
    r"wordmarks?|brand(?:ing|\s*names?)?|titles?|headlines?|headers?|headings?|"
    r"slogans?|taglines?|copy|characters?|fonts?|packaging text|"
    r"ctas?|call[- ]to[- ]actions?|buttons?|badges?|stickers?|price tags?|prices?)"
)
_NEGATION_QUALIFIER = r"(?:any\s+|visible\s+|written\s+|painted\s+|rendered\s+)?"
PROMPT_NEGATION_LEAD = re.compile(
    r"\[?\s*\b(?:excluded?|exclusions?|negatives?(?:\s+prompt)?|avoid|"
    r"do\s+not\s+include|don'?t\s+include|not\s+allowed|never)\s*[:=\-\u2013\u2014]\s*([^.\n\]]+?)\.?\s*\]?\s*$",
    re.IGNORECASE | re.DOTALL,
)
# Any noun, not just text words: "no bottles or drink" came back with four bottles. A term is
# a short run of plain words up to the next comma, full stop, "just"/"only", or the end.
_ANY_TERM = r"(?:[A-Za-z][A-Za-z'-]*)(?:\s+(?!no\b|just\b|only\b|with\b)[A-Za-z][A-Za-z'-]*){0,3}"
PROMPT_NEGATION_PHRASE = re.compile(
    r"\b(?:no|without|free\s+of|zero|not\s+any|absolutely\s+no)\s+" + _NEGATION_QUALIFIER + _ANY_TERM
    # "no X or Y" continues the exclusion; after a comma only another explicit "no ..." does, so
    # "no bottles, just people having fun" keeps the wanted half.
    + r"(?:\s*(?:(?:/|&|\bor\b|\band\b|\bnor\b)\s*(?:no\s+)?|,\s*no\s+)" + _NEGATION_QUALIFIER + _ANY_TERM + r")*"
    + r"(?:\s+(?:anywhere|at\s+all|in\s+the\s+(?:image|picture|frame|shot)|on\s+(?:it|the\s+image|the\s+picture)))?\b",
    re.IGNORECASE,
)
_TEXT_TERM_RE = re.compile(r"\b" + _NEGATION_TERM + r"\b", re.IGNORECASE)
PROMPT_TEXTLESS_WORD = re.compile(r"\b(?:text-?free|textless|wordless|un-?lettered)\b", re.IGNORECASE)
_TEXT_NEGATION_TERMS = (
    "text", "word", "letter", "typograph", "type", "writing", "caption",
    "label", "number", "digit", "sign", "wordmark", "title", "headline",
    "header", "heading", "slogan", "tagline", "copy", "character", "font",
    "packaging", "cta", "call to action", "call-to-action", "button",
    "badge", "sticker", "price",
)


def split_prompt_negations(prompt):
    """Take the exclusions out of a typed Image prompt.

    Returns (positive, negations, about_text): `positive` is the prompt
    with every "no text", "without words", "excluded: ..." clause
    removed (None if nothing is left), `negations` the clauses as typed,
    for the negative channel and the results-page note, and
    `about_text` whether any of them was about lettering -- in which
    case the run wants a text-free picture whatever the checkboxes say.
    """
    if not prompt:
        return None, [], False
    text = prompt
    negations = []
    lead = PROMPT_NEGATION_LEAD.search(text)
    if lead:
        tail = lead.group(1).strip().strip("[]").strip()
        if tail:
            negations.append(tail)
        text = text[: lead.start()]
    for match in PROMPT_NEGATION_PHRASE.finditer(text):
        negations.append(match.group(0).strip())
    text = PROMPT_NEGATION_PHRASE.sub(" ", text)
    if PROMPT_TEXTLESS_WORD.search(text):
        negations.append("no text")
        text = PROMPT_TEXTLESS_WORD.sub(" ", text)
    # Drop remaining clauses that mention lettering: "no bottles or drink with text on it"
    # would otherwise leave "with text on it" standing, which is an order for text.
    kept_clauses = []
    for clause in re.split(r"\s*[,;.]\s*", text):
        if not clause.strip():
            continue
        if _TEXT_TERM_RE.search(clause):
            negations.append(clause.strip())
            continue
        # A clause that lost its ending to a removal ("we can add a
        # bottle but", "just image with") gets the dangling word cut.
        clause = re.sub(
            r"(?:^|\s+)(?:but|with|and|or|of|for|in|on|just|only|then)\s*$", "", clause.strip(),
            flags=re.IGNORECASE,
        ).strip()
        if clause and clause.lower() not in ("just image", "an image", "image", "a picture", "picture"):
            kept_clauses.append(clause)
    text = ", ".join(kept_clauses)
    # Tidy what the removals left behind: doubled commas, an orphaned
    # "and", empty brackets, stray spaces.
    text = re.sub(r"\(\s*\)|\[\s*\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([,;])\s*(?:[,;]\s*)+", r"\1 ", text)
    text = re.sub(r"(?:^|[,;])\s*(?:and|or|nor|with|but)\s*(?=[,;]|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*([,;])\s*(?:[,;]\s*)+", r"\1 ", text)
    text = text.strip(" ,;:-\u2013\u2014\t\n")
    text = re.sub(r"\s+([,;.])", r"\1", text)
    about_text = any(_TEXT_TERM_RE.search(n) for n in negations)
    return (text or None), negations, about_text


def _env_int(name: str, default: int) -> int:
    """An integer setting from the environment, falling back to `default`
    for anything unusable. `or default` on the raw value, not a get()
    default: a .env written from the example ships bare "NAME=" lines,
    and an empty string is present as far as os.environ is concerned."""
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


# Each retry is another paid API call, so keep it small. 0 disables retries, warning stays.
AI_TEXT_RETRY_LIMIT = max(0, _env_int("AI_TEXT_RETRIES", 2))
# ...and how many of those a PAID provider gets (each one is an image).
AI_PAID_TEXT_RETRIES = max(0, _env_int("AI_PAID_TEXT_RETRIES", 2))

# Added on retry: painted objects come with markings unless told otherwise, a ball with a
# maker's name, a can with a label.
NO_TEXT_RETRY_CLAUSE = (
    "every object plain and unbranded with no printed markings, no lettering on any "
    "surface, no logos on equipment or clothing"
)


def _text_amount(result) -> float:
    """How much lettering a text check found -- the area of its boxes,
    or the count when there are no boxes -- to rank attempts by."""
    findings = getattr(result, "findings", None) or []
    area = 0.0
    for finding in findings:
        box = getattr(finding, "box", None)
        try:
            _left, _top, box_w, box_h = box  # (left, top, width, height)
            area += max(0, box_w) * max(0, box_h)
        except Exception:  # noqa: BLE001
            area += 1.0
    return area or float(len(findings))

# No mention of text, not even room for it: "space for overlaid text" came back captioned.
BACKGROUND_PROMPT_GUIDANCE = (
    "sharp focus, crisp fine detail, high resolution, professional photography, "
    "even lighting, plenty of clean empty negative space, uncluttered composition"
)

# Exclusions belong on the negative channel: in the positive prompt "no faces" just adds that
# token, and Ideogram documents positive as taking precedence (asked for no people, got two).
# Short on purpose: only what a backdrop can never want. Lettering has its own clause.
BACKDROP_NEGATIVE_CLAUSE = (
    "cluttered composition, busy distracting background, collage, "
    "picture frame, border, vignette"
)


def _keep_the_product_out_of_a_backdrop_prompt(prompt: str, product_name: str):
    """A backdrop prompt with the BRAND NAME in it, minus the name.

    A model asked for "HydroBoost sports drink" draws a bottle with
    HydroBoost written on it, and the lettering is what the text check
    then has to smear out -- so the name goes. The rest of the prompt
    does not: it comes from the campaign brief, it is the instruction,
    and an earlier version of this that also deleted "bottle", "can" and
    "product" turned "a chilled blue sports drink bottle, winter theme"
    into "a chilled blue, winter theme" -- an adjective with no noun.
    Handed a prompt with no subject in it, the model kept the colour and
    the mood and invented the rest, which is where the stray people came
    from. Returns (prompt, what was taken out)."""
    if not prompt:
        return prompt, []
    removed = []
    names = []
    if product_name:
        compact = re.sub(r"\s+", "", product_name)
        names = [re.escape(product_name.strip())]
        if compact.lower() != product_name.strip().lower():
            names.append(re.escape(compact))
    for pattern in names:
        for match in re.finditer(pattern, prompt, flags=re.IGNORECASE):
            removed.append(match.group(0))
        prompt = re.sub(pattern, " ", prompt, flags=re.IGNORECASE)
    # Tidy what the removals leave behind: doubled spaces, empty
    # comma-separated parts, a leading comma.
    parts = [part.strip() for part in re.split(r"[,;]", prompt)]
    parts = [re.sub(r"\s{2,}", " ", part) for part in parts if part and re.search(r"[A-Za-z]", part)]
    # Removing the name from "a bottle of Hydro Boost" leaves "a bottle of", an unfinished
    # sentence, so the dangling joining word goes with it.
    parts = [
        re.sub(
            r"\s+(?:of|for|with|by|from|in|on|at|and|or|a|an|the)$", "", part,
            flags=re.IGNORECASE,
        ).strip()
        for part in parts
    ]
    parts = [part for part in parts if part and re.search(r"[A-Za-z]", part)]
    return ", ".join(parts), removed

# For runs that want type in the picture: the no-logos/room-for-text half above protects the
# template and works against this brief.
BACKGROUND_PROMPT_GUIDANCE_WITH_TEXT = (
    "sharp focus, crisp fine detail, high resolution, professional graphic design, "
    "clean legible typography, balanced composition"
)


# A bare hex triplet reads as text, not colour, so each is sent as "#0057b8 (blue)": a word
# the model steers on, exact value still readable.
_COLOUR_WORDS = (
    ((0, 0, 0), "black"), ((255, 255, 255), "white"), ((128, 128, 128), "grey"),
    ((255, 0, 0), "red"), ((0, 128, 0), "green"), ((0, 0, 255), "blue"),
    ((255, 255, 0), "yellow"), ((255, 165, 0), "orange"), ((128, 0, 128), "purple"),
    ((255, 192, 203), "pink"), ((165, 42, 42), "brown"), ((0, 255, 255), "cyan"),
    ((0, 128, 128), "teal"), ((245, 245, 220), "cream"), ((25, 25, 112), "navy"),
    # One anchor per hue misnames brand colours: mid blue #0057b8 is numerically nearer
    # pure teal, and came back "teal".
    ((0, 87, 184), "blue"), ((65, 105, 225), "blue"), ((135, 206, 235), "light blue"),
    ((50, 205, 50), "bright green"), ((255, 122, 0), "orange"), ((220, 20, 60), "red"),
    # #ffd100 is 44 from pure orange, 46 from pure yellow, so one anchor called it "orange" and
    # orange is what got painted across a blue-and-yellow campaign.
    ((255, 209, 0), "yellow"), ((255, 223, 0), "yellow"), ((255, 215, 0), "gold"),
    # Same failure elsewhere: brand green #00a651 came back "teal", silver #c0c0c0 came back
    # "pink", being nearer the pink anchor than grey or white.
    ((0, 166, 81), "green"), ((0, 200, 100), "green"),
    ((192, 192, 192), "silver"), ((211, 211, 211), "light grey"),
    ((139, 195, 74), "light green"), ((154, 205, 50), "yellow green"),
    ((96, 125, 139), "blue grey"),
)


def _colour_word(rgb) -> str:
    """The nearest word from _COLOUR_WORDS to `rgb`, by plain RGB
    distance. Approximate on purpose -- it only has to be close enough to
    steer a prompt, not to name a paint."""
    red, green, blue = rgb
    return min(
        _COLOUR_WORDS,
        key=lambda entry: (entry[0][0] - red) ** 2
        + (entry[0][1] - green) ** 2
        + (entry[0][2] - blue) ** 2,
    )[1]


# Given only a product name, Ideogram returned one idea on every seed: the name as a wordmark
# on a flat field. This names that idea so it can be excluded.
LOGO_NEGATIVE_CLAUSE = (
    "logo design, wordmark, lettermark, brand mark, typographic poster, "
    "plain flat background, text-only layout"
)

# One composition per run. Changing the seed varies the rendering, not the idea.
BACKDROP_SCENES = (
    "a dramatic product hero shot on a reflective surface with a splash frozen mid-air",
    "a wide lifestyle photograph of the audience in motion outdoors at golden hour",
    "an extreme close-up of the product's texture and droplets, shallow depth of field",
    "an energetic action scene with the audience mid-effort, motion blur in the background",
    "a bold studio still life with hard directional light and long shadows",
    "an aerial or high-angle scene suggesting the campaign's setting, with open space",
    "a moody low-key scene with a single shaft of light on the product",
    "a bright, airy scene with the product surrounded by fresh natural elements",
)


# What is left after these is the subject matter, usable as a scene without the sentence a
# typography model would set.
_THEME_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "your",
    "you", "our", "we", "is", "are", "be", "it", "its", "this", "that", "new",
    "now", "get", "at", "by", "from", "into", "as", "up", "out", "all", "every",
}


def _theme_keywords(message: str, limit: int = 5) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", message or "")
    kept = []
    for word in words:
        low = word.lower()
        if low in _THEME_STOPWORDS or low in kept:
            continue
        kept.append(low)
    return ", ".join(kept[:limit])


def _backdrop_scene(product_name, campaign_message, audience, rng=None, textless=False) -> str:
    """One composition for this run's automatic backdrop, with the brief
    folded into it.

    The product name alone is what made every run the same picture. The
    campaign message says what the ad is ABOUT, the audience says who is
    in it, and the scene says how it is shot -- none of which were being
    sent. The scene rotates per run so two runs of the same brief are two
    ideas rather than two renderings of one.
    """
    chooser = rng or random
    scene = chooser.choice(BACKDROP_SCENES)
    pictured = "the audience" in scene
    # Every distinctive word has come back as lettering at some point (ADULTS, HYDRO BOOST). The
    # backdrop sits under the product and copy layers, so it needs a scene, not names.
    who = "people" if textless else (audience or "the target audience")
    scene = scene.replace("the audience", who)
    if textless:
        scene = scene.replace("the product's", "the object's").replace("the product", "a plain unlabeled object")
    parts = [scene]
    if product_name and not textless:
        parts.append(f"featuring {product_name}")
    if audience and not pictured and not textless:
        # A scene without people still has an audience: it sets the styling. Left out on
        # text-free runs: "styled for Active Adults 18-34" came back with ADULTS lettered.
        parts.append(f"styled for {audience}")
    if campaign_message:
        if textless:
            # Not even as keywords: "mood: rehydrate, refreshing, summer" came back with
            # REHYDRATE and SUMMER set as type.
            pass
        else:
            # Unquoted on purpose. Quoted text in an Ideogram prompt is a
            # request to letter it; this is a theme, not copy to set.
            parts.append(f"on the theme of {campaign_message.rstrip('.')}")
    return ", ".join(parts)


# How many pictures a mood board holds: Ideogram takes up to three
# style references for one style.
REFERENCE_LIMIT = 3
REFERENCE_SLOTS = ("upload_ai_reference", "upload_ai_reference_2", "upload_ai_reference_3")


def _mood_board(images) -> Image.Image:
    """The reference pictures side by side at equal height, so one
    description (reference_look_phrase) covers the board as a whole --
    its palette is the palette of all of them, weighted by area."""
    if len(images) == 1:
        return images[0]
    height = 256
    tiles = []
    for im in images:
        w = max(1, round(im.width * height / max(im.height, 1)))
        tiles.append(im.convert("RGB").resize((w, height)))
    board = Image.new("RGB", (sum(t.width for t in tiles), height))
    x = 0
    for t in tiles:
        board.paste(t, (x, 0))
        x += t.width
    return board


# A style reference carries layout and typography too, so an ad with a headline returns a
# backdrop with a headline. Text-free runs paint the words out, or crop when too big.
REFERENCE_CROP_MIN_FRACTION = 0.35


def _textless_reference(image, name: str):
    """(image, note): `image` with its lettering painted out or cropped
    off, and a sentence saying what was done -- or (image, None) when
    there was nothing to do or no way to check, and (None, note) for a
    reference that is mostly lettering and must not be sent at all."""
    from src.text_check import (
        MAX_REMOVABLE_AREA_FRACTION, build_text_mask, masked_area_fraction, remove_text,
    )

    found = find_text(image)
    if not found.available or not found.found_text:
        return image, None
    mask = build_text_mask(image, found)
    if masked_area_fraction(image, mask) <= MAX_REMOVABLE_AREA_FRACTION:
        cleaned, count, reason = remove_text(image, found)
        if reason is None:
            return cleaned, (
                f"{name} had text in it ({count} word{'s' if count != 1 else ''}), "
                "painted out before it went to the model as a style reference."
            )
        return image, None
    # Too much to paint out (a headline, a lockup): use the tallest text-free band, if it is
    # enough of the picture to match on.
    try:
        import numpy as np
    except ImportError:
        return image, None
    rows = np.asarray(mask).max(axis=1) > 0
    best = (0, 0)
    start = None
    for y, has_text in enumerate(list(rows) + [True]):
        if not has_text and start is None:
            start = y
        elif has_text and start is not None:
            if y - start > best[1] - best[0]:
                best = (start, y)
            start = None
    top, bottom = best
    if bottom - top < image.height * REFERENCE_CROP_MIN_FRACTION:
        return None, (
            f"{name} is mostly text ({masked_area_fraction(image, mask):.0%} of it), "
            "which couldn't be taken out, so it was NOT sent to the model -- a style "
            "reference like that makes it copy the lettering. A photo without a "
            "headline works as a reference."
        )
    cropped = image.crop((0, top, image.width, bottom))
    # Anything small that survived in the band goes the cheap way.
    leftover = find_text(cropped)
    if leftover.found_text:
        painted, _, reason = remove_text(cropped, leftover)
        if reason is None:
            cropped = painted
    return cropped, (
        f"{name} had a headline across it, so only the text-free band "
        f"({(bottom - top) / image.height:.0%} of its height) went to the model as a "
        "style reference."
    )


def reference_look_phrase(image) -> str:
    """A dropped reference picture described in words -- its dominant
    colours, brightness, warmth, saturation and contrast -- as art
    direction for the prompt.

    This is the reference for providers that can't take the file
    (Pollinations' GET endpoint, the offline placeholder), and it rides
    along with the real style reference on Ideogram too, since a prompt
    that agrees with the picture steers better than one that doesn't.
    It is a description of the LOOK, not the subject: what is in the
    picture stays out of the prompt on purpose, or a photo of a beach
    would turn every backdrop into a beach. Words, never hex codes --
    see _brand_palette_phrase() for what hex codes turn into.
    """
    small = image.convert("RGB")
    small.thumbnail((96, 96))
    # Dominant colours: quantise to a handful and take them by area,
    # skipping ones that name the same word twice.
    quantised = small.quantize(colors=6, method=Image.Quantize.MEDIANCUT)
    palette = quantised.getpalette()[: 6 * 3]
    counts = sorted(quantised.getcolors(), reverse=True)
    words = []
    for _, index in counts:
        rgb = tuple(palette[index * 3 : index * 3 + 3])
        word = _colour_word(rgb)
        if word not in words:
            words.append(word)
        if len(words) == 3:
            break
    pixels = list(small.getdata())
    n = max(len(pixels), 1)
    lum = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in pixels]
    mean_lum = sum(lum) / n
    contrast = (sum((v - mean_lum) ** 2 for v in lum) / n) ** 0.5
    sat = sum((max(p) - min(p)) / (max(p) or 1) for p in pixels) / n
    warmth = sum(r - b for r, _, b in pixels) / n

    tone = "dark and moody" if mean_lum < 80 else "bright and airy" if mean_lum > 170 else "evenly lit"
    temp = "warm" if warmth > 25 else "cool" if warmth < -25 else "neutral"
    colour = "vivid" if sat > 0.45 else "muted" if sat < 0.18 else "natural"
    depth = "high-contrast lighting with deep shadows" if contrast > 60 else "soft low-contrast lighting" if contrast < 30 else "balanced lighting"

    named = words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]
    return (
        f"in the look of the reference picture: a palette of {named}, "
        f"{tone}, {temp} {colour} colours, {depth}"
    )


# What a palette request must NOT turn into. Sent as a negative prompt
# whenever brand colours are ticked -- see _brand_palette_phrase().
PALETTE_NEGATIVE_CLAUSE = (
    "color swatches, colour chips, palette strip, hex codes, color codes, "
    "style guide, color reference chart"
)


_SUBJECT_COLOUR_WORDS = frozenset(
    [word for _rgb, word in _COLOUR_WORDS] + [
        "gold", "golden", "silver", "bronze", "copper", "beige", "ivory", "charcoal",
        "turquoise", "lime", "magenta", "maroon", "olive", "amber", "lavender",
        "emerald", "aqua", "mint", "peach", "coral", "burgundy", "tan", "clear",
        "transparent", "frosted", "matte", "translucent",
    ]
)


def _prompt_names_its_own_colour(prompt: str) -> bool:
    """Does this prompt already say what colour its subject is?

    "a green body wash bottle" does; "winter theme, clean bright
    background" does not. Multi-word entries ("light blue") are matched
    whole, so "light" alone never counts."""
    if not prompt:
        return False
    lowered = prompt.lower()
    return any(
        re.search(r"\b%s\b" % re.escape(word), lowered) for word in _SUBJECT_COLOUR_WORDS
    )


def _brand_palette_phrase(brand_colors, subject_has_its_own_colour: bool = False) -> str:
    """The ticked brand colours as a prompt clause, or "" when none are.

    Colour WORDS only, phrased as art direction for the scene. The first
    version sent hex codes too, for precision -- and Ideogram rendered
    exactly what it was given: a style-guide strip across the bottom of
    the ad with "#D00606  #1332CD  #FFFFFF" printed under three swatches.
    To an image model a hex code is not a colour, it is six characters
    to letter, and "palette" beside it is a request for a chart. So the
    hex stays in the results page for people, the words go to the model,
    and the clause says where the colours go: into the picture. What it
    must not become is in PALETTE_NEGATIVE_CLAUSE, on the channel that
    actually works for "not this".

    Only the ticked swatches arrive here -- an unticked one still holds a
    colour in the form, and sending it would be steering the picture by
    a value the user switched off.
    """
    if not brand_colors:
        return ""
    words = []
    for rgb in brand_colors:
        word = _colour_word(rgb)
        if word not in words:
            words.append(word)
    # A brief naming its subject's colour collides with the campaign palette: asked for
    # green and blue/yellow dominance, it painted the scene. Palette then steers light only.
    if len(words) == 1:
        if subject_has_its_own_colour:
            return (
                f"{words[0]} in the lighting and the surroundings, with the subject "
                "keeping the colour the prompt gives it"
            )
        return (
            f"{words[0]} as the dominant colour of the scene, in the lighting, "
            "surfaces and background"
        )
    named = ", ".join(words[:-1]) + " and " + words[-1]
    if subject_has_its_own_colour:
        return (
            f"{named} in the lighting and the surroundings, with the subject "
            "keeping the colour the prompt gives it"
        )
    return (
        f"{named} as the dominant colours of the scene, in the lighting, "
        "surfaces and background"
    )


def _build_full_ad_prompt(
    product_name, campaign_message, header_text, cta_text, audience, market,
    brand_colors=None,
) -> str:
    """Compose a brief for a COMPLETE ad -- headline, hero, call to
    action -- out of what the campaign brief already says.

    Short, and with as little quoted text as an ad can carry. A
    typography model sets one or two short quoted strings cleanly;
    asked for the product name AND a headline AND a supporting line
    AND a button, in a prompt that also told it what NOT to write, it
    set most of them as gibberish ("the expensive one isn't creating
    words"). So: the headline (the campaign message when there is no
    headline), the button label if there is one, and the product named
    as the subject rather than as a third piece of type. Audience and
    market are left out of the prompt altogether -- they were only ever
    art direction, and "(not written on the ad)" is exactly the kind of
    aside a model letters.
    """
    headline = (header_text or campaign_message or "").strip()
    parts = ["a polished advertising poster"]
    if product_name:
        parts.append(f"for {product_name}")
    if headline:
        parts.append(f'with large bold headline text "{headline}"')
    else:
        parts.append(f'with the product name "{product_name}" as large bold headline text' if product_name else "with a large bold headline")
    if cta_text:
        parts.append(f'and a button with the text "{cta_text.strip()}"')
    # Unlabelled on purpose. Nobody asked for lettering on the packaging --
    # the model puts it there because product shots have labels, and then
    # improvises the letters, which is where "HROPPBOOST" and "HYDRO
    # BOOOST" came from. A model will leave text out far more reliably
    # than it will spell it, so the bottle is described as blank and the
    # brand name is carried by the headline, which is the one string worth
    # spending the model's spelling on. The real label goes back on in the
    # split PSD, where the actual product layer sits under the painted one.
    parts.append("a hero shot of the product as the focus, its packaging plain and unlabelled")
    palette = _brand_palette_phrase(brand_colors)
    if palette:
        parts.append(palette)
    parts.append(
        "clean modern layout, crisp legible typography, only the quoted text appears, "
        "generous margins, nothing touching the edges"
    )
    return ", ".join(parts)


# A headline longer than this comes back misspelled more often than not:
# the model sets six words cleanly and starts inventing letters after.
FULL_AD_HEADLINE_WORDS = 6


# A whole-ad run otherwise letters the audience and market as copy and fills the bottom edge
# with fine print, which comes out as gibberish since there are no real words to set.
FULL_AD_NEGATIVE_CLAUSE = (
    "extra text, fine print, disclaimer, lorem ipsum, gibberish lettering, "
    "misspelled words, cropped text, "
    # The packaging specifically: it is the one surface the model letters
    # without being asked, and the one place a misspelling reads as a
    # broken brand rather than as a rough draft.
    "text on the packaging, label text, writing on the bottle, brand name on the product, "
    # And the quote marks themselves. They are in the prompt to mark which
    # words are literal, and the model was painting them into the artwork
    # -- four of nine sizes came back with the headline in quotes.
    "quotation marks, quote marks around text"
)


# The margin every whole-ad size is asked to keep, in pixels of the
# final creative -- on top of whatever the provider's crop takes.
FULL_AD_EDGE_PADDING_PX = 10

# Same padding the app gives its own text boxes: a designer box flush to the canvas edge (the
# skyscraper description starts at x=0) would put every line's first letter against the edge.
TEXT_EDGE_PADDING_PX = 10


def _inset_boxes_to_canvas(boxes: dict, canvas: tuple, pad: int = TEXT_EDGE_PADDING_PX) -> dict:
    """Copies of the text-layer boxes clamped to sit at least `pad`
    pixels inside the canvas. Boxes already inside are untouched; a box
    that reaches an edge is trimmed, never moved, so the designer's
    alignment survives. Only the layers the app sets type in."""
    width, height = canvas
    out = dict(boxes)
    for name in ("header", "description", "legal", "cta"):
        box = boxes.get(name)
        if not box:
            continue
        left, top, right, bottom = box
        left2, top2 = max(left, pad), max(top, pad)
        right2, bottom2 = min(right, width - pad), min(bottom, height - pad)
        if right2 - left2 >= 8 and bottom2 - top2 >= 8:
            out[name] = (left2, top2, right2, bottom2)
    return out



def _full_ad_safe_area(width: int, height: int, provider_name: str) -> tuple:
    """How far in from each edge, as a fraction of the FINAL creative,
    the model has to keep everything for none of it to be cut off.

    Ideogram renders a fixed set of ratios and the app centre-crops the
    nearest one to the size asked for -- a 160x600 skyscraper comes
    back as 1:3 and loses 10% off each side, which is exactly the
    margin the model gave the headline. The model can't know that, so
    the crop is worked out here and said in the prompt, plus
    FULL_AD_EDGE_PADDING_PX of breathing room.

    Returns (left_right_fraction, top_bottom_fraction).
    """
    crop_x = crop_y = 0.0
    if provider_name == "ideogram":
        from src.providers.ideogram_provider import _closest_aspect

        aw, ah = (int(v) for v in _closest_aspect(width, height).split("x"))
        rendered = aw / ah
        wanted = width / height
        if rendered > wanted:
            # Wider than needed: the sides go. Fraction of the final
            # width lost on EACH side.
            crop_x = (rendered / wanted - 1) / 2
        elif rendered < wanted:
            crop_y = (wanted / rendered - 1) / 2
    pad_x = FULL_AD_EDGE_PADDING_PX / max(width, 1)
    pad_y = FULL_AD_EDGE_PADDING_PX / max(height, 1)
    return crop_x + pad_x, crop_y + pad_y


def _full_ad_margin_clause(width: int, height: int, provider_name: str) -> str:
    lr, tb = _full_ad_safe_area(width, height, provider_name)
    # Never less than a designer's margin, however big the canvas.
    lr_pct, tb_pct = max(3, round(lr * 100)), max(3, round(tb * 100))
    return (
        f"every element -- text, product, logo -- kept at least {lr_pct}% of the width "
        f"in from the left and right edges and {tb_pct}% of the height in from the top "
        "and bottom, with nothing touching or cut off by an edge"
    )


IDEOGRAM_SPEED_CHOICES = (
    ("TURBO", "Turbo -- $0.03 an image, rougher (drafts)"),
    ("DEFAULT", "Default -- $0.06 an image"),
    ("QUALITY", "Quality -- $0.09 an image (the keeper)"),
)
DEFAULT_IDEOGRAM_SPEED = "TURBO"


class _MeteredProvider:
    """A provider with a running bill. Every generate() adds one image
    at the provider's own price to `meter`, so the results page can say
    what the run cost -- and the retry that a text check triggers is
    counted like any other image, because it is one."""

    def __init__(self, provider, meter: dict):
        self._provider = provider
        self._meter = meter

    def __getattr__(self, name):
        return getattr(self._provider, name)

    def generate(self, *args, **kwargs):
        image = self._provider.generate(*args, **kwargs)
        self._meter["images"] += 1
        self._meter["cost"] += float(getattr(self._provider, "cost_per_image", 0.0) or 0.0)
        speeds = self._meter.setdefault("speeds", set())
        speed = getattr(self._provider, "rendering_speed", None)
        if speed:
            speeds.add(speed)
        return image


def _spend_note(meter: dict) -> str:
    """One line for the results page: what this run cost, or "" if it
    was free."""
    if not meter.get("cost"):
        return ""
    images = meter["images"]
    speeds = ", ".join(sorted(s.title() for s in meter.get("speeds", ())))
    return (
        f"Ideogram spend this run: about ${meter['cost']:.2f} "
        f"({images} image{'s' if images != 1 else ''}{f' at {speeds}' if speeds else ''})."
    )


def _generate_text_free(
    provider, prompt: str, width: int, height: int, allow_text: bool = False,
    negative_extra: str = None, style_reference: bytes = None,
):
    """Generate `prompt`, and regenerate if the result has text baked in.

    `style_reference` -- a dropped picture whose look the backdrop should
    share -- goes to the provider as a file when it can take one
    (Ideogram), and is left out otherwise: the caller has already put
    that picture into the prompt in words for providers that can't (see
    reference_look_phrase()).

    With `allow_text`, none of that happens: no negative prompt, no OCR
    check, no retry. Suppressing lettering is right for a backdrop the
    template's own header and CTA sit on top of, and exactly wrong when
    someone has asked the model for a logo, a headline or a laid-out
    creative -- which is the one thing Ideogram is here for. The caller
    decides which of the two it wants; this only stops fighting the
    result once it has.

    Returns (image, prompt_used, attempts, leftover_findings) where
    `leftover_findings` is a TextCheckResult describing what's still in
    the image that came back -- empty when the picture came out clean, and
    `available=False` when Tesseract isn't installed and nothing could be
    checked at all.

    Note the ordering: the image returned is always the LAST one
    generated, not the cleanest one seen. Each retry asks harder, so the
    last is the best-steered attempt; picking a "winner" across attempts
    would mean holding several full-size images in memory to choose
    between shades of wrong.
    """
    attempts = 0
    result = TextCheckResult(available=False)
    image = None
    used = prompt
    # A real negative-prompt field is where the no-lettering instruction goes: negation in a
    # positive prompt is weak, and Ideogram ranks positive higher. Pollinations' GET folds it in.
    negative = NO_TEXT_CLAUSE
    if negative_extra:
        negative = f"{negative}, {negative_extra}"
    # Only providers that declare support get the keyword at all, so a
    # provider (or a test stub) with the plain signature keeps working.
    extra = (
        {"style_reference": style_reference}
        if style_reference and getattr(provider, "supports_style_reference", False)
        else {}
    )
    if not allow_text and getattr(provider, "supports_render_mode", False):
        # Photographic, not design mode, which is a poster generator and sets type.
        # rewrite_prompt off: MagicPrompt has added a caption to a prompt asking for none.
        extra.update({"photographic": True, "rewrite_prompt": False})
    if allow_text:
        # Text is wanted here: nothing to verify, no retry budget spent, no-text clause out.
        # A caller's own exclusion still goes, on the negative channel.
        if negative_extra:
            image = provider.generate(
                prompt, width=width, height=height, negative_prompt=negative_extra, **extra
            )
            shown = (
                f"{prompt}  [excluded: {negative_extra}]"
                if getattr(provider, "supports_negative_prompt", False)
                else f"{prompt}, {negative_extra}"
            )
            return image, shown, 1, TextCheckResult(available=False)
        image = provider.generate(prompt, width=width, height=height, **extra)
        return image, prompt, 1, TextCheckResult(available=False)
    # The offline placeholder draws the prompt on purpose, so verifying it fails every time and
    # burns the whole retry budget plus OCR.
    verify = getattr(provider, "name", "") != "mock"
    retry_limit = AI_TEXT_RETRY_LIMIT if verify else 0
    if getattr(provider, "cost_per_image", 0):
        # Retries here are paid images, so cap them: a fresh generation beats a
        # painted-out one, and paint-out is the last resort.
        retry_limit = min(retry_limit, AI_PAID_TEXT_RETRIES)
    # Every attempt is kept: if none comes back clean, the one with the
    # least lettering is the one to paint out, not simply the last.
    tried = []
    while attempts <= retry_limit:
        # First attempt asks politely, retries escalate. Escalation goes into the prompt too:
        # the negative prompt alone did not take the brand off a volleyball.
        used = prompt if attempts == 0 else f"{prompt}, {NO_TEXT_RETRY_CLAUSE}"
        negative_used = (
            negative if attempts == 0 else f"{negative}, {NO_TEXT_ESCALATION}"
        )
        image = provider.generate(
            used, width=width, height=height, negative_prompt=negative_used, **extra
        )
        attempts += 1
        if not verify:
            break
        result = find_text(image)
        if not result.available or not result.found_text:
            break
        tried.append((image, result, used, negative_used))
    else:
        image, result, used, negative_used = min(tried, key=lambda t: _text_amount(t[1]))
    if getattr(provider, "supports_negative_prompt", False):
        shown = f"{used}  [excluded: {negative_used}]"
    else:
        # Folded into the prompt by the provider -- report it that way,
        # since that is what was actually sent.
        shown = f"{used}, {negative_used}"
    return image, shown, attempts, result


def _text_detector_problem() -> str:
    """Why the scene-text detector isn't running, for a warning."""
    from src import text_check

    text_check.ensure_text_detector(install=False)
    return text_check._detector_state or f"{text_check.DETECTOR_PACKAGE} isn't installed"


def _clean_text_out(image, result, label: str, attempts: int):
    """Last resort once the retries are spent: paint the text out.

    Returns (image, note, warning). Free -- no API call -- which is why
    it runs before giving up and warning, and why it is worth attempting
    even on a detection that might be a false alarm.

    scrub_text() does whatever it takes short of a new image: it paints
    words out, re-reads, paints again, and crops to the largest
    text-free band when the lettering covers so much of the frame that
    painting would replace the picture. So a warning here means
    something readable SURVIVED all of that, not that nothing was tried.
    """
    attempt_word = f"{attempts} attempt{'s' if attempts != 1 else ''}"
    # Paint out, re-read, paint again, crop if painting can't finish. Report what is left.
    cleaned, what, after = scrub_text(image)
    done = " ".join(what) if what else "nothing could be done"
    if after.found_text:
        return (
            cleaned,
            None,
            f"The generated {label} had text in it after {attempt_word} ({result.summary()}); "
            f"{done} Some is still readable ({after.summary()}). Worth a look before shipping, "
            "or run again for a fresh picture.",
        )
    from src.text_check import detector_description

    return (
        cleaned,
        f"Text was found in the generated {label} after {attempt_word} "
        f"({result.summary()}) and removed: {done} (checked with {detector_description()}.)",
        None,
    )


def _flagship_template_path():
    """The saved template at the flagship size, or failing that the
    largest saved template; None when there are none."""
    # Matches the render's tie-break: with two files for one size, the later name wins, so its
    # backdrop is the one taken.
    best, best_area = None, -1
    if not templates_dir().is_dir():
        return None
    by_size = {}
    for path in sorted(templates_dir().iterdir()):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_PSD_TEMPLATE_EXTENSIONS:
            continue
        match = _SIZE_IN_FILENAME_RE.search(path.name)
        if not match:
            continue
        by_size[(int(match.group(1)), int(match.group(2)))] = path
    if CONTENT_PSD_SIZE in by_size:
        return by_size[CONTENT_PSD_SIZE]
    for size, path in by_size.items():
        if size[0] * size[1] > best_area:
            best, best_area = path, size[0] * size[1]
    return best


def template_background_as_hero(dest_dir: Path):
    """The flagship template's own `background` layer, written to
    `dest_dir` as a PNG the size of that template, for use as the hero
    image -- so a batch can run on the template's own backdrop (a plain
    white, a brand gradient) without anyone exporting it by hand.
    Returns (path, note) or raises ValueError."""
    template = _flagship_template_path()
    if template is None:
        raise ValueError("there are no saved templates in default_templates/ to take a background from")
    try:
        psd = PSDImage.open(template)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"couldn't open {template.name}: {exc}") from exc
    layer = next((l for l in psd if (l.name or "").strip().lower() == "background"), None)
    if layer is None:
        raise ValueError(f"{template.name} has no layer named 'background'")
    try:
        image = layer.composite()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"couldn't read the background layer of {template.name}: {exc}") from exc
    if image is None:
        raise ValueError(f"the background layer of {template.name} is empty")
    canvas = Image.new("RGBA", (psd.width, psd.height), (255, 255, 255, 255))
    canvas.alpha_composite(image.convert("RGBA"), (max(0, layer.left), max(0, layer.top)))
    dest = dest_dir / f"hero_from_{template.stem}_background.png"
    canvas.convert("RGB").save(dest)
    return dest, (
        f"Hero image: the background layer of {template.name} ({psd.width}x{psd.height}), "
        "used as the backdrop of every size."
    )


def _off_canvas_layers(psd_path, names) -> dict:
    """{name: description} for each of `names` that IS a top-level layer
    in the PSD but has no pixels inside the canvas."""
    try:
        psd = PSDImage.open(psd_path)
    except Exception:  # noqa: BLE001
        return {}
    wanted = {n.lower() for n in names}
    out = {}
    for layer in psd:
        key = (layer.name or "").strip().lower()
        if key not in wanted:
            continue
        try:
            x0, y0, x1, y1 = layer.bbox
        except Exception:  # noqa: BLE001
            continue
        if x1 <= x0 or y1 <= y0:
            out[key] = "it has no pixels"
        elif x0 >= psd.width or y0 >= psd.height or x1 <= 0 or y1 <= 0:
            out[key] = f"its box is at x {x0}..{x1}, y {y0}..{y1}"
    return out


def _default_template_sizes() -> list:
    """The sizes saved in default_templates/, read from the filenames
    alone.

    _default_size_templates() below answers the same question but opens
    every PSD to do it -- several seconds and a lot of memory for seven
    multi-megabyte files. Callers that only need the dimensions (sizing a
    generated image to fit them, say) shouldn't pay that.
    """
    sizes = []
    if not templates_dir().is_dir():
        return sizes
    for path in sorted(templates_dir().iterdir()):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_PSD_TEMPLATE_EXTENSIONS:
            continue
        match = _SIZE_IN_FILENAME_RE.search(path.name)
        if match:
            sizes.append((int(match.group(1)), int(match.group(2))))
    return sizes


def _generation_size(sizes, fallback) -> tuple:
    """How big to ask a provider for an image that has to cover `sizes`.

    A generated image is cropped to fill each output size, so it needs to
    be at least as wide as the widest and as tall as the tallest -- taken
    independently, since a 1920x1080 and a 1080x1920 in the same batch
    together demand 1920x1920. Generating at the old flat 728x480 meant
    every size above that was an upscale, which is exactly what a blurry
    background looks like.

    Capped per edge, and never smaller than the fallback.
    """
    widths = [w for w, _h in sizes if w > 0]
    heights = [h for _w, h in sizes if h > 0]
    if not widths or not heights:
        return fallback
    return (
        max(fallback[0], min(max(widths), MAX_GENERATED_EDGE)),
        max(fallback[1], min(max(heights), MAX_GENERATED_EDGE)),
    )


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
    for size, path in _default_template_paths().items():
        try:
            templates[size] = _open_template_cached(path)
        except Exception:
            continue
        template_paths[size] = path
    return templates, template_paths


# Saved templates are named for their size: tester-720x480.psd.
SAVED_TEMPLATE_PREFIX = "tester-"


def _files_claiming_size(size) -> list:
    """Every PSD in default_templates/ whose name carries `size`, in
    folder order -- the ones _default_template_paths() chooses between."""
    if not templates_dir().is_dir():
        return []
    found = []
    for path in sorted(templates_dir().iterdir()):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_PSD_TEMPLATE_EXTENSIONS:
            continue
        match = _SIZE_IN_FILENAME_RE.search(path.name)
        if match and (int(match.group(1)), int(match.group(2))) == tuple(size):
            found.append(path)
    return found


def _default_template_paths() -> dict:
    """{(width, height): path} for the saved templates, without opening
    any of them. Everything that only needs to know which files exist
    (the form's layer lists) reads this."""
    template_paths: dict = {}
    if not templates_dir().is_dir():
        return template_paths
    for path in sorted(templates_dir().iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_PSD_TEMPLATE_EXTENSIONS:
            continue
        match = _SIZE_IN_FILENAME_RE.search(path.name)
        if not match:
            continue
        template_paths[(int(match.group(1)), int(match.group(2)))] = path
    return template_paths


# Keyed by (path, size, mtime), so a re-saved template is re-read. Flattening a large PSD
# takes about a second and a page load asks for each template several times.
_template_image_cache: dict = {}


def _open_template_cached(path: Path):
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    image = _template_image_cache.get(key)
    if image is None:
        image = open_as_rgb(path)
        # One entry per path: an older version of the same file is stale.
        for old in [k for k in _template_image_cache if k[0] == key[0]]:
            del _template_image_cache[old]
        _template_image_cache[key] = image
    return image.copy()


# How far a content PSD may sit from a saved template's size and still update that slot rather
# than export a near-duplicate (728x480 against 720x480). The ratio check keeps it honest.
CONTENT_PSD_SNAP_RATIO_TOLERANCE = 0.05   # aspect ratio within 5%
CONTENT_PSD_SNAP_SIZE_TOLERANCE = 0.10    # each dimension within 10%


SAME_RATIO_TOLERANCE = 0.005  # 1080x1080 and 1200x1200 are the same shape; 1200x1200 and 1200x1250 are not


def _same_ratio_source(size, sources):
    """The uploaded size whose proportions match `size`, or None.

    Exact shape, not "close": a template scaled onto a size of the same
    ratio lands every layer where the designer put it, while one a few
    percent off would need a crop or a squash. Ties go to the nearest
    pixel size, so a 1200x1200 takes the 1080x1080 over a 300x300.
    """
    width, height = size
    if not width or not height:
        return None
    ratio = width / height
    best = None
    for candidate in sources:
        candidate_width, candidate_height = candidate
        if not candidate_width or not candidate_height:
            continue
        if abs(candidate_width / candidate_height - ratio) / ratio > SAME_RATIO_TOLERANCE:
            continue
        distance = abs(candidate_width - width) + abs(candidate_height - height)
        if best is None or distance < best[0]:
            best = (distance, candidate)
    return best[1] if best else None


NEAR_MISS_SIZE_TOLERANCE = 0.12  # each side within 12%: catches a transposed digit, not a different format


def _near_miss_size(size, known):
    """The known size `size` is probably a typo of, or None.

    A different aspect ratio is what makes a typo expensive here -- same-
    ratio near misses already snap or carry -- so this asks for both
    sides to be close and the shape to differ, e.g. 3480x2160 vs
    3840x2160, or 1290x1080 vs 1920x1080.
    """
    width, height = size
    if not width or not height:
        return None
    best = None
    for candidate in known:
        cw, ch = candidate
        if not cw or not ch or candidate == size:
            continue
        if abs(cw / ch - width / height) / (width / height) <= SAME_RATIO_TOLERANCE:
            continue  # same shape: it carries over as designed, nothing to flag
        # Two near misses: both sides close (3480x2160 for 3840x2160), or one side exact and the
        # other's digits shuffled (1290x1080 for 1920x1080). Shuffled ranks first.
        shuffled = (cw == width and sorted(str(ch)) == sorted(str(height))) or (
            ch == height and sorted(str(cw)) == sorted(str(width))
        )
        close = abs(cw - width) / cw <= NEAR_MISS_SIZE_TOLERANCE and abs(ch - height) / ch <= NEAR_MISS_SIZE_TOLERANCE
        # One side exactly right and the other off by a digit (1920x1280
        # for 1920x1080) is the other common slip.
        one_side = (cw == width and abs(ch - height) / ch <= 0.25) or (ch == height and abs(cw - width) / cw <= 0.25)
        if not shuffled and not close and not one_side:
            continue
        distance = (0 if shuffled else 1, abs(cw - width) + abs(ch - height))
        if best is None or distance < best[0]:
            best = (distance, candidate)
    return best[1] if best else None


def _ratio_label(width: int, height: int) -> str:
    return ratio_label(width, height)


def _template_scale(canvas_size, target_size, fit_mode: str) -> float:
    """How much a template's pixels grow (or shrink) when its canvas is
    fitted onto the output size -- the same factor map_box_through_fit()
    applies to its boxes, so type sizes read from the PSD can follow."""
    if not canvas_size:
        return 1.0
    src_w, src_h = canvas_size
    target_w, target_h = target_size
    if src_w <= 0 or src_h <= 0 or (src_w, src_h) == (target_w, target_h):
        return 1.0
    if fit_mode == "contain":
        return min(target_w / src_w, target_h / src_h)
    return max(target_w / src_w, target_h / src_h)


ROW_SNAP_TOLERANCE = 0.03  # a canvas a few pixels off a known size is that size; 1200 vs 1080 is not


def _snap_row_size(size, known) -> tuple:
    """A row's size snapped to a known size it is within 3% of on both
    sides (728x480 -> 720x480), else unchanged. Tighter than the content
    PSD's snap on purpose: 1200x1200 is 10% off 1080x1080 and is its own
    size."""
    width, height = size
    if not width or not height or size in known:
        return size
    best = None
    for candidate in known:
        cw, ch = candidate
        if not cw or not ch:
            continue
        if abs(cw - width) / width > ROW_SNAP_TOLERANCE or abs(ch - height) / height > ROW_SNAP_TOLERANCE:
            continue
        distance = abs(cw - width) + abs(ch - height)
        if best is None or distance < best[0]:
            best = (distance, candidate)
    return best[1] if best else size


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
    Path(sys.executable) if FROZEN else Path(__file__),
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
# Startup and footer build stamp. If it lags an edit you just made, the process is serving
# stale code.
print(f"[webapp] code build stamp: {BUILD_STAMP} (auto-reload on unless FLASK_RELOAD=0)")

SIZE_PRESET_CHOICES = [
    ("default", "Social defaults -- 1080x1080, 1080x1920, 1920x1080"),
    ("web-top7", "Web ad sizes (9) -- Leaderboard, Medium Rectangle, Skyscraper, etc."),
    ("broadcast", "Broadcast/video frame sizes (3) -- 1080p, 720p, 4K UHD"),
]

# static/ is read-only and ships in the bundle, so it resolves off BUNDLE_DIR like templates/;
# Flask's default derives from __name__ and points somewhere useless in a frozen build.
app = Flask(
    __name__,
    template_folder=str(BUNDLE_DIR / "templates"),
    static_folder=str(BUNDLE_DIR / "static"),
)
app.secret_key = os.environ.get("WEBAPP_SECRET_KEY", secrets.token_hex(16))
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB -- generous enough for a short product video
# Jinja caches compiled templates for the process lifetime and the reloader only watches .py,
# so edits to templates/index.html silently did nothing in a running server.
app.config["TEMPLATES_AUTO_RELOAD"] = True


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


def _parse_layer_align(value: str) -> str:
    """A template text layer's alignment choice: left/center/right, or
    "template" -- the paragraph alignment the layer has in Photoshop,
    which is the default. It used to default to left, which quietly
    un-centred every centred description."""
    return value if value in VALID_TEXT_ALIGNMENTS else "template"


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


def _download_folder_parts(product_name, campaign_name, campaign_label: str) -> tuple:
    """Where a run's files go, in the zip and under downloads/:
    ("<Campaign Name>", "<Product Name>") as typed, minus characters a
    filesystem refuses. A card with no campaign name uses its slot
    ("campaign1"); one with no product name has just the campaign."""
    campaign = _product_folder_name(campaign_name) if campaign_name else ""
    product = _product_folder_name(product_name) if product_name else ""
    parts = (campaign or campaign_label,)
    if product:
        parts += (product,)
    return parts


GLOW_SIZE_MIN, GLOW_SIZE_MAX, GLOW_SIZE_DEFAULT = 1, 100, 12


def _parse_stroke_size(raw) -> int:
    """A text outline's width as a percentage of the font size.

    Relative for the same reason the glow radius is (see below): one
    setting has to read the same on a 160x600 and a 1920x1080, and a
    fixed pixel stroke cannot. 0 -- the default -- means no outline at
    all, so the control is inert until somebody asks for it.
    """
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, value))


def _template_glow_percent(template_glow: dict, fx_scale: float, design_font_px: float) -> int:
    """A template's own Outer Glow as the form's glow size -- a percentage
    of the font size -- for the renderer to redraw it from. Photoshop's
    Size is the halo's full reach; the renderer's radius is the blur it
    starts from, and the two meet at Size = 2.7 x radius (the fit the
    live-text export uses in the other direction), so a glow the app
    wrote into a template comes back at the strength it went in."""
    size_px = float(template_glow.get("size", 0) or 0) * fx_scale
    radius_px = size_px / 2.7
    return max(1, int(round(radius_px / max(float(design_font_px), 1.0) * 100)))


FORM_LAYER_STYLING_FIELDS = tuple(
    [f"layer_{layer}_{control}" for layer in ("header", "description", "legal", "cta")
     for control in ("glow", "shadow", "use_custom_color", "stroke_size", "font_size", "font_family")]
    + ["layer_cta_text_stroke_size"]
    + [f"layer_{layer}_{control}" for layer in ("logo", "product") for control in ("glow", "shadow", "stroke")]
)


def _switch_off_form_layer_styling() -> list:
    """Blank the form's layer-styling switches and values for this
    request (see FORM_LAYER_STYLING_FIELDS), so the parse below sees
    them off and _remember_form_fields() saves them off. Returns plain
    names of what was actually on, for the note."""
    form = request.form.copy()
    was_on = []
    for name in FORM_LAYER_STYLING_FIELDS:
        if (form.get(name) or "").strip():
            was_on.append(name[len("layer_"):].replace("_", " "))
            form[name] = ""
    if was_on:
        request.form = form
        # The form was remembered at the top of the request, before the drops:
        # overwrite those entries so the next form opens with the switches off.
        prefs = _load_preferences()
        key = _product_memory_key(request.form.get("product_name"), request.form.get("campaign_name"))
        targets = [prefs]
        if key and key in _product_memories(prefs):
            targets.append(_product_memories(prefs)[key])
        for target in targets:
            for name in FORM_LAYER_STYLING_FIELDS:
                if name in REMEMBERED_FIELD_NAMES:
                    target[name] = ""
        _save_preferences(prefs)
    return was_on


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


def _parse_band_blur(raw) -> int:
    """How soft a text band's edges are, 0-100 as a percentage of its
    height. 0 (the default, and what unusable input falls back to) keeps
    the hard rectangle edge.
    """
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, value))


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


def _flatten_psd_upload(path: Path) -> Path:
    """A PSD dropped on a picture field becomes a PNG of its composite,
    transparency intact, in place; any other file is returned as is."""
    if path.suffix.lower() != ".psd":
        return path
    try:
        from psd_tools import PSDImage

        psd = PSDImage.open(path)
        # force=True composites the layers rather than returning the cached preview,
        # which is flat RGB; that is what keeps a cut-out logo's transparency.
        image = psd.composite(force=True)
        if image is None:
            raise ValueError("the file has no composite image")
        image = image.convert("RGBA")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Couldn't flatten '{path.name}': {exc}") from exc
    out = path.with_suffix(".png")
    image.save(out)
    try:
        path.unlink()
    except OSError:
        pass
    return out


def _save_upload(file_storage, dest_dir: Path) -> Path:
    """Save an uploaded file, preserving its extension (open_as_rgb() and
    render_creative() both branch on it -- e.g. to detect a video)."""
    safe_name = secure_filename(file_storage.filename) or "upload"
    dest = dest_dir / safe_name
    file_storage.save(dest)
    return dest


REFERENCE_FETCH_LIMIT = 25 * 1024 * 1024  # Ideogram's per-image cap


def _save_data_url_image(data_url: str, name: str, dest_dir: Path) -> Path:
    """Save a picture the browser sent as a data: URL (a file dropped on
    the mood board -- not every browser lets a script put a dropped file
    into a file input, so the page reads it and sends the bytes itself).
    Raises ValueError with a message fit for the form."""
    import base64

    match = re.match(r"data:([a-z]+/[a-z0-9.+-]+)?;base64,(.+)$", data_url or "", re.S)
    if not match:
        raise ValueError(f"Reference image '{name or 'dropped picture'}' couldn't be read from the drop.")
    mime = (match.group(1) or "").lower()
    ext_by_mime = {
        "image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/webp": ".webp",
        "image/heic": ".heic", "image/heif": ".heif", "image/tiff": ".tif", "image/gif": ".gif", "image/bmp": ".bmp",
        "video/mp4": ".mp4", "video/quicktime": ".mov", "video/x-m4v": ".m4v", "video/webm": ".webm",
    }
    ext = ext_by_mime.get(mime) or (Path(name or "").suffix.lower() if Path(name or "").suffix.lower() in REFERENCE_EXTENSIONS else None)
    if ext is None:
        raise ValueError(
            f"Reference image '{name or 'dropped picture'}' isn't a picture or video the board can use "
            f"(it came in as {mime or 'an unknown type'})."
        )
    try:
        data = base64.b64decode(match.group(2), validate=False)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Couldn't read the dropped picture '{name}': {exc}") from exc
    if len(data) > REFERENCE_FETCH_LIMIT * 4:
        raise ValueError(f"'{name}' is over 100 MB -- choose it with Choose Files instead of dropping it.")
    stem = secure_filename(Path(name or "dropped").stem) or "dropped"
    dest = dest_dir / f"{stem}{ext}"
    counter = 2
    while dest.exists():
        dest = dest_dir / f"{stem}-{counter}{ext}"
        counter += 1
    dest.write_bytes(data)
    return _normalise_reference(dest)


# Beyond Ideogram's PNG/JPEG/WebP: video (middle frame used), HEIC, TIFF, GIF, BMP, PSD, all
# converted to PNG on the way in.
REFERENCE_EXTRA_EXTENSIONS = (".heic", ".heif", ".tif", ".tiff", ".gif", ".bmp", ".psd") + VIDEO_EXTENSIONS
REFERENCE_EXTENSIONS = ALLOWED_LAYER_IMAGE_EXTENSIONS + REFERENCE_EXTRA_EXTENSIONS


def _normalise_reference(path: Path) -> Path:
    """Turn a reference picture the board accepted into a PNG/JPEG/WebP
    Ideogram will take, in place. A video becomes its middle frame; a
    HEIC (an iPhone photo) or TIFF/GIF/BMP is converted. Returns the
    path to use. Raises ValueError with a message fit for the form."""
    suffix = path.suffix.lower()
    if suffix in ALLOWED_LAYER_IMAGE_EXTENSIONS:
        return path
    if suffix == ".psd":
        return _flatten_psd_upload(path)
    try:
        if suffix in VIDEO_EXTENSIONS:
            from src.image_ops import extract_video_frame

            image = extract_video_frame(path)  # the middle frame
        else:
            if suffix in (".heic", ".heif"):
                try:
                    import pillow_heif

                    pillow_heif.register_heif_opener()
                except ImportError as exc:
                    raise ValueError(
                        f"'{path.name}' is an iPhone HEIC photo; converting it needs the pillow-heif "
                        "package (pip install pillow-heif), or export it as JPEG first."
                    ) from exc
            image = Image.open(path)
            image.load()
            image = image.convert("RGB")
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Couldn't read '{path.name}' as a picture or video: {exc}") from exc
    out = path.with_suffix(".png")
    image.save(out)
    try:
        path.unlink()
    except OSError:
        pass
    return out


def _fetch_web_image(url: str, dest_dir: Path) -> Path:
    """Save the picture at `url` into `dest_dir` and return its path.

    For a picture dragged in from a web page: the browser hands over the
    image's address rather than a file, so the app goes and gets it.
    http(s) only, must come back as an image, capped at what Ideogram
    will accept. Raises ValueError with a message fit for the form.
    """
    import requests
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("The reference image address must start with http:// or https://.")
    try:
        resp = requests.get(
            url, timeout=20, stream=True,
            # A browser's own user agent: some image hosts (Wikimedia
            # among them) answer anything else with a 400 or 403.
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
                ),
                "Accept": "image/avif,image/webp,image/png,image/jpeg,*/*;q=0.8",
            },
        )
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Couldn't fetch the reference image: {exc}") from exc
    if resp.status_code != 200:
        raise ValueError(f"The reference image address returned HTTP {resp.status_code}.")
    ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(ctype)
    if ext is None:
        raise ValueError(
            f"That address isn't a PNG, JPEG or WebP image (it came back as {ctype or 'unknown'}). "
            "Right-click the picture and copy the image address itself, not the page's."
        )
    data = b""
    for chunk in resp.iter_content(64 * 1024):
        data += chunk
        if len(data) > REFERENCE_FETCH_LIMIT:
            raise ValueError("The reference image is over 25 MB, which is more than Ideogram accepts.")
    stem = secure_filename(Path(parsed.path).stem) or "web-reference"
    dest = dest_dir / f"{stem}{ext}"
    dest.write_bytes(data)
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
    # Let go of by a per-size restore: carrying it forward here is what
    # put the old template back on the very next submit.
    prior_job_id = Path(prior_job_dir).name
    if f"{prior_job_id}:{field_name}" in dropped_file_markers():
        return None
    prior_path = prior_job_dir / "uploads" / prior_rel
    if not prior_path.is_file():
        return None
    dest_path = uploads_dir / prior_rel
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # Link, not copy: 4K templates run 30-50 MB and re-copying per Edit filled 108 GB.
    # Uploads get a new name, so a linked file is never rewritten; copy where linking fails.
    try:
        if dest_path.exists():
            dest_path.unlink()
        os.link(prior_path, dest_path)
    except OSError:
        shutil.copy2(prior_path, dest_path)
    return dest_path


# Working space, not an archive: keep the newest JOB_KEEP_COUNT runs, fewer over
# JOB_DISK_BUDGET_GB. Indexed sessions and the run a fresh form draws files from survive.
JOB_KEEP_COUNT = 40
JOB_KEEP_MIN = 10
JOB_DISK_BUDGET_GB = 8.0
_last_prune_at = 0.0


def _folder_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def _protected_job_ids() -> set:
    keep = set()
    prefs = _load_preferences()
    if prefs.get("last_job_id"):
        keep.add(prefs["last_job_id"])
    sessions_dir = JOBS_DIR / "_sessions"
    if sessions_dir.is_dir():
        # Only the newest few sessions' jobs: an index for every session
        # ever would protect everything.
        recent = sorted(sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
        for path in recent:
            try:
                index = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(index, dict):
                keep.update(str(v) for v in index.values() if isinstance(v, str))
    return keep


# Image + caption pairs with an index, the layout a LoRA fine-tune wants, since job folders
# are pruned. backdrops/ is what the provider returned, the only thing a style tune may see;
# training on the finished ad teaches it to paint lettering, which returns as gibberish.
IMAGE_LIBRARY_DIR = BASE_DIR / "image_library"
IMAGE_LIBRARY_BACKDROPS = "backdrops"
IMAGE_LIBRARY_CREATIVES = "creatives"
IMAGE_LIBRARY_SIZES = ((1200, 1200), (1200, 627))


# Appended clauses are instructions, not descriptions. Left in a caption, a fine-tune learns
# to paint swatches and hex codes; "no logos" teaches that a logo belongs here.
LIBRARY_CAPTION_DROP = (
    "color swatches", "colour swatches", "color chips", "colour chips",
    "palette strip", "hex codes", "color codes", "colour codes",
    "style guide", "color reference chart", "colour reference chart",
    "no watermarks", "no signage",
)


def _describing_clauses(prompt: str) -> str:
    """`prompt` with the instructions to the generator taken out."""
    kept = []
    for clause in (prompt or "").split(","):
        text = clause.strip()
        low = text.lower()
        if not text or low.startswith("no ") or low in LIBRARY_CAPTION_DROP:
            continue
        kept.append(text)
    return ", ".join(kept)


def _library_caption(record: dict) -> str:
    """What the image shows, in the order a caption wants it: the subject
    first, then the framing, then the description.

    The description is what the person typed, not the prompt the app sent.
    The sent one carries a tail of negative instructions -- "no faces, no
    logos, no text" and a run of colour-chart boilerplate -- which describe
    nothing in the picture and would be learned as if they did.

    Deliberately says nothing about the style either. A LoRA learns the
    look from what every image has in common; naming it in the caption
    teaches the model to treat it as optional instead.
    """
    bits = [b for b in (record.get("product"), record.get("campaign")) if b]
    subject = " -- ".join(bits) if bits else "advertising artwork"
    parts = [subject]
    if record.get("kind") == "creative":
        parts.append(f"{record.get('ratio') or record.get('size')} advertising creative")
    described = (record.get("prompt_typed") or "").strip() or _describing_clauses(record.get("prompt"))
    if described:
        parts.append(described)
    if record.get("market"):
        parts.append(f"market: {record['market']}")
    return ", ".join(parts)


def _library_write(folder: Path, stem: str, source: Path, entry: dict) -> bool:
    """One image, its caption and its index line. False if anything failed."""
    image_path = folder / f"{stem}.png"
    if image_path.exists():
        return False              # same run, same image: already kept
    caption = _library_caption(entry)
    entry["caption"] = caption
    entry["file"] = f"{folder.name}/{image_path.name}"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_path)
        # The caption beside the image: what every fine-tune tool reads.
        (folder / f"{stem}.txt").write_text(caption, encoding="utf-8")
        with open(IMAGE_LIBRARY_DIR / "index.jsonl", "a", encoding="utf-8") as index:
            index.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        return False
    return True


def save_to_image_library(job_dir: Path, creatives: list, record: dict, backdrop: Path = None) -> list:
    """Keep this run: the generated backdrop, and the library sizes built
    from it. Returns the paths written, relative to image_library/.

    Never raises. A library that cannot be written must not fail somebody's
    run -- the creatives are already saved by the time this is called.
    """
    written = []
    try:
        IMAGE_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return written

    base = dict(record)
    stamp = base.pop("stamp", "")
    slug = base.pop("slug", None) or "creative"
    short_job = (base.get("job_id") or "")[:8]

    # The training image: bare artwork, as the provider returned it.
    if backdrop is not None and Path(backdrop).is_file():
        entry = dict(base, kind="backdrop", size=None, ratio=None)
        stem = f"{stamp}_{slug}_backdrop_{short_job}"
        if _library_write(IMAGE_LIBRARY_DIR / IMAGE_LIBRARY_BACKDROPS, stem, Path(backdrop), entry):
            written.append(f"{IMAGE_LIBRARY_BACKDROPS}/{stem}.png")

    # What was built from it -- reference, not training material.
    wanted = {f"{w}x{h}" for w, h in IMAGE_LIBRARY_SIZES}
    for creative in creatives:
        label = creative.get("label")
        if label not in wanted:
            continue
        source = job_dir / creative.get("filename", "")
        if not source.is_file():
            continue
        entry = dict(
            base, kind="creative", size=label,
            ratio=creative.get("ratio") or label, name=creative.get("name"),
        )
        stem = f"{stamp}_{slug}_{label}_{short_job}"
        if _library_write(IMAGE_LIBRARY_DIR / IMAGE_LIBRARY_CREATIVES, stem, source, entry):
            written.append(f"{IMAGE_LIBRARY_CREATIVES}/{stem}.png")
    return written


def prune_job_folders(force: bool = False) -> list:
    """Delete old run folders (and their browsable zips) so the output
    folder stays a working set. Returns the ids removed. Runs at most
    every ten minutes unless forced."""
    global _last_prune_at
    now = time.time()
    if not force and now - _last_prune_at < 600:
        return []
    _last_prune_at = now
    if not JOBS_DIR.is_dir():
        return []
    jobs = []
    for path in JOBS_DIR.iterdir():
        if not path.is_dir() or not re.fullmatch(r"(draft_)?[0-9a-f]{12,32}", path.name):
            continue
        try:
            jobs.append((path.stat().st_mtime, path))
        except OSError:
            continue
    jobs.sort(key=lambda item: item[0], reverse=True)  # newest first
    protected = _protected_job_ids()
    removed = []
    keep, candidates = [], []
    for _mtime, path in jobs:
        if path.name in protected or len(keep) < JOB_KEEP_MIN:
            keep.append(path)
        elif len(keep) < JOB_KEEP_COUNT:
            keep.append(path)
        else:
            candidates.append(path)
    budget = JOB_DISK_BUDGET_GB * (1024 ** 3)
    kept_size = sum(_folder_size(p) for p in keep)
    # Over budget even after the count cut: the oldest kept go too, down
    # to the minimum and never the protected ones.
    while kept_size > budget and len(keep) > JOB_KEEP_MIN:
        oldest = keep.pop()
        if oldest.name in protected:
            keep.insert(0, oldest)
            break
        kept_size -= _folder_size(oldest)
        candidates.append(oldest)
    for path in candidates:
        try:
            shutil.rmtree(path)
            removed.append(path.name)
        except OSError:
            continue
        for stale in DOWNLOADS_DIR.glob(f"*{path.name[:12]}*") if DOWNLOADS_DIR.is_dir() else []:
            try:
                stale.unlink()
            except OSError:
                pass
    return removed


def _session_index_path(session_id: str) -> Path:
    """Where a campaign session's {slot -> job_id} index lives -- see
    _load_session_campaigns() and the session_id/campaign_slot handling in
    generate(). Deliberately a sibling of JOBS_DIR's job folders (not
    inside any one job's own folder), since one session covers several
    jobs, not just one."""
    return JOBS_DIR / "_sessions" / f"{secure_filename(session_id)}.json"


def drop_row_files_for_job(job_id: str, rows, product_key: str = "") -> None:
    """Let go of psd_file_<n> from one job, for the rows given.

    Put back pressed on an Edit page is about the job being edited, not
    the last run the preferences point at -- and the Edit page loads that
    job's files directly. Without this the row came back filled the moment
    the page reloaded, and put the template back on the next submit.
    """
    job_id = (job_id or "").strip()
    if not job_id or not rows:
        return
    prefs = _load_preferences()
    products = _product_memories(prefs)
    targets = [prefs]
    if product_key and product_key in products:
        targets.append(products[product_key])
    for memory in targets:
        dropped = [d for d in (memory.get("dropped_files") or []) if isinstance(d, str)]
        for i in rows:
            marker = f"{job_id}:psd_file_{i}"
            if marker not in dropped:
                dropped.append(marker)
        memory["dropped_files"] = dropped
    if products:
        prefs["products"] = products
    _save_preferences(prefs)


def dropped_file_markers() -> set:
    """Every "<job id>:<field>" a per-size restore has let go of, across
    the top-level memory and every product's.

    Put back records these; the remembered form, the Edit page and the
    carry-forward on submit all have to honour them. They did not, which
    is why putting a size back looked like it worked and then the same
    template came straight back on the next run: Edit reads a job's files
    directly, so the row still held the upload and re-promoted it.
    """
    prefs = _load_preferences()
    markers = set()
    for memory in [prefs] + list(_product_memories(prefs).values()):
        for marker in (memory.get("dropped_files") or []):
            if isinstance(marker, str):
                markers.add(marker)
    return markers


def _files_minus_dropped(job_id, files: dict) -> dict:
    """`files` without anything let go of for that job."""
    if not job_id or not files:
        return files or {}
    dropped = dropped_file_markers()
    return {name: value for name, value in files.items() if f"{job_id}:{name}" not in dropped}


def _fields_with_campaign(fields: dict) -> dict:
    """A job's saved fields, with Campaign filled in when it is missing.

    The Edit page rebuilds its cards from what a job saved, not from the
    remembered form -- a second card builder, and one the campaign fix
    missed. A batch generated before the field was required saved it
    empty, so reopening that batch handed back a blank Campaign and sent
    its templates to a folder of their own all over again.
    """
    fields = dict(fields or {})
    if not (fields.get("campaign_name") or "").strip():
        # The market too: it is saved alongside, and it is what tells a
        # product that two briefs claim which of them this batch was.
        known = campaign_for_product(fields.get("product_name"), fields.get("market"))
        if known:
            fields["campaign_name"] = known
    return fields


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
    # Fallback content is the job's saved form: a failed-submission draft, or a run predating
    # session tracking, has no session index but still has its fields.
    try:
        own = json.loads((JOBS_DIR / fallback_job_id / "form_state.json").read_text())
        fallback[0]["prefill"] = _fields_with_campaign(own.get("fields"))
        fallback[0]["prefill_files"] = _files_minus_dropped(fallback_job_id, own.get("files") or {})
    except (OSError, ValueError, TypeError):
        pass
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
            "prefill": _fields_with_campaign(slot_state.get("fields")),
            "prefill_files": _files_minus_dropped(slot_job_id, slot_state.get("files") or {}),
            "edit_job_id": slot_job_id,
        })
    campaigns = campaigns or fallback
    for card in campaigns:
        fields = card.get("prefill") or {}
        card["layers"] = card_layer_sets(fields.get("product_name"), fields.get("campaign_name"))
    return campaigns


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


def _template_paths_in(folder=None) -> dict:
    """_default_template_paths() for one folder -- a campaign's own
    (see card_layer_sets()) rather than the request's."""
    folder = templates_dir() if folder is None else Path(folder)
    template_paths: dict = {}
    if not folder.is_dir():
        return template_paths
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_PSD_TEMPLATE_EXTENSIONS:
            continue
        match = _SIZE_IN_FILENAME_RE.search(path.name)
        if not match:
            continue
        template_paths[(int(match.group(1)), int(match.group(2)))] = path
    return template_paths


def _editable_text_layers(folder=None) -> set:
    """Which of the named text layers are actually editable right now --
    i.e. present AND switched on in at least one saved template.

    A layer switched off in Photoshop can't be restyled: there are no
    visible words to recolour or resize, and the renderer skips it. The
    form greys its fields out rather than accepting settings that would
    quietly do nothing. Judged across all saved templates together, since
    one enabled somewhere is enough for the field to be worth offering.
    """
    return _scan_layer_sets(folder if folder is not None else templates_dir())["editable"]


def _switched_off_layers(folder=None) -> set:
    """Named layers the saved templates HAVE but that are switched off in
    every template carrying them.

    Not the same as absent, and not the same as hidden-by-this-form: the
    designer turned these off in Photoshop, so they are not drawn no
    matter what the form asks for. The "hide" checkbox for one of them is
    therefore already true and cannot be made false from here -- it is
    shown ticked and locked rather than left inviting a click that would
    change nothing.

    Every layer kind, not just text: logo, cta and product get a hide
    checkbox too, and a designer can switch any of them off.
    """
    return _scan_layer_sets(folder if folder is not None else templates_dir())["switched_off"]


def _present_text_layers(folder=None) -> set:
    """Every named text layer the saved templates have at all -- switched
    on or off.

    The companion to _editable_text_layers(), and the distinction matters
    for whether a section is offered: a layer that is PRESENT but off can
    be turned on in Photoshop, so its controls are worth showing greyed
    out with an explanation. A layer that isn't in any template has
    nothing to explain and no path to enabling it, so its controls aren't
    shown at all rather than sitting there permanently dead.
    """
    return _scan_layer_sets(folder if folder is not None else templates_dir())["present"]


# One scan per folder, keyed by folder plus file sizes and mtimes. The three layer sets each
# opened every PSD per card: seven cards of nine templates made a sixteen-second form.
_layer_sets_cache: dict = {}


def _folder_signature(folder: Path) -> tuple:
    try:
        return tuple(sorted(
            (p.name, p.stat().st_size, p.stat().st_mtime_ns)
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in ALLOWED_PSD_TEMPLATE_EXTENSIONS
        ))
    except OSError:
        return ()


def _layer_scan_cache_path() -> Path:
    return JOBS_DIR / "layer_scan_cache.json"


_template_scan_cache: dict | None = None


def _scan_template_layers(path: Path) -> dict | None:
    """One template's named layers -- present/editable text layers and
    every layer seen/visible -- read once and kept on disk keyed by the
    file's size and mtime. Opening a PSD is a quarter of a second, and
    the form asks about every template of every card on every load;
    with the answers kept, a fresh start reads only files that changed
    since the last run (a template re-saved in Photoshop, say)."""
    global _template_scan_cache
    try:
        stat = path.stat()
    except OSError:
        return None
    key = str(path.resolve())
    if _template_scan_cache is None:
        try:
            _template_scan_cache = json.loads(_layer_scan_cache_path().read_text(encoding="utf-8"))
            if not isinstance(_template_scan_cache, dict):
                _template_scan_cache = {}
        except (OSError, ValueError):
            _template_scan_cache = {}
    entry = _template_scan_cache.get(key)
    if entry and entry.get("size") == stat.st_size and entry.get("mtime_ns") == stat.st_mtime_ns:
        return entry
    from src.image_ops import layers_under_background
    try:
        psd = PSDImage.open(path)
    except Exception:
        return None
    editable, present, seen, visible = set(), set(), set(), set()
    # Same judgement as get_psd_text_layers(): a top-level type layer with words, on and not
    # buried under the background, is editable.
    buried = layers_under_background(psd)
    for layer in psd:
        name = (layer.name or "").strip()
        if not name or getattr(layer, "kind", None) != "type":
            continue
        try:
            text = layer.text
        except Exception:
            continue
        if not text:
            continue
        present.add(name.lower())
        if layer.visible and name.lower() not in buried:
            editable.add(name.lower())
    for layer in psd.descendants():
        name = layer.name.strip().lower()
        seen.add(name)
        if layer.visible:
            visible.add(name)
    entry = {
        "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "editable": sorted(editable), "present": sorted(present), "seen": sorted(seen), "visible": sorted(visible),
    }
    _template_scan_cache[key] = entry
    try:
        _layer_scan_cache_path().parent.mkdir(parents=True, exist_ok=True)
        _layer_scan_cache_path().write_text(json.dumps(_template_scan_cache), encoding="utf-8")
    except OSError:
        pass
    return entry


def _scan_layer_sets(folder: Path) -> dict:
    """{"editable", "present", "switched_off"} for one folder, opening
    each template once. editable: named text layers switched on (and
    not buried under the background) in at least one template; present:
    every named text layer, on or off; switched_off: named layers of
    any kind the templates have but that are off in every one."""
    key = (str(folder), _folder_signature(folder))
    cached = _layer_sets_cache.get(key)
    if cached is not None:
        return {k: set(v) for k, v in cached.items()}
    editable, present, seen, visible = set(), set(), set(), set()
    for path in _template_paths_in(folder).values():
        scan = _scan_template_layers(path)
        if scan is None:
            continue
        editable |= set(scan["editable"])
        present |= set(scan["present"])
        seen |= set(scan["seen"])
        visible |= set(scan["visible"])
    result = {"editable": editable, "present": present, "switched_off": seen - visible}
    for stale in [k for k in _layer_sets_cache if k[0] == key[0]]:
        del _layer_sets_cache[stale]
    _layer_sets_cache[key] = {k: set(v) for k, v in result.items()}
    return result


def card_layer_sets(product_name=None, campaign_name=None) -> dict:
    """The three layer sets above, judged on one campaign card's own
    templates -- default_templates/<campaign>/<product>/ -- not the
    shared folder. Each card's notes ("the description layer is switched
    off in your saved templates") must describe the templates that card
    will render with, otherwise a layer hidden in some other folder greys
    out fields on a card whose own templates have it on. A campaign whose
    folder does not exist yet (nothing generated for it) is judged on the
    shared folder, the set its folder will be seeded from."""
    return _scan_layer_sets(_layer_judging_dir(product_name, campaign_name))


def _page_layer_sets() -> dict:
    """The layer sets for a card with no product yet (the blank card a
    Create Campaign click clones): judged like any other card, on the
    shared folder or the seed set."""
    sets = card_layer_sets()
    return {
        "editable_text_layers": sets["editable"],
        "present_text_layers": sets["present"],
        "switched_off_layers": sets["switched_off"],
    }


def _has_saved_templates(folder: Path) -> bool:
    return folder.is_dir() and any(
        p.is_file() and p.suffix.lower() in ALLOWED_PSD_TEMPLATE_EXTENSIONS and _SIZE_IN_FILENAME_RE.search(p.name)
        for p in folder.iterdir()
    )


_seed_set_cache: dict = {}


def _seed_templates_dir() -> Path | None:
    """The backup zip's templates, unpacked once into a scratch folder --
    the set every new campaign folder is seeded from. Judged on when a
    card's folder does not exist yet and the shared folder holds nothing
    loose (the usual layout: only the zip and the campaign folders at
    the top of default_templates/). Re-unpacked when the zip changes."""
    zip_path = template_reset_zip()
    if zip_path is None:
        return None
    try:
        stat = zip_path.stat()
    except OSError:
        return None
    key = (str(zip_path), stat.st_size, stat.st_mtime_ns)
    cached = _seed_set_cache.get(key)
    if cached is not None and cached.is_dir():
        return cached
    dest = Path(tempfile.mkdtemp(prefix="seed-templates-"))
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                name = Path(info.filename).name
                if info.is_dir() or info.filename.startswith("__MACOSX/") or name.startswith("._"):
                    continue
                if name.lower().rsplit(".", 1)[-1] != "psd" or not SIZE_IN_NAME_RE_LOOSE.search(name):
                    continue
                (dest / name).write_bytes(zf.read(info))
    except (OSError, zipfile.BadZipFile):
        return None
    for old in [k for k in _seed_set_cache if k[0] == key[0]]:
        shutil.rmtree(_seed_set_cache.pop(old), ignore_errors=True)
    _seed_set_cache[key] = dest
    return dest


def _layer_judging_dir(product_name=None, campaign_name=None) -> Path:
    """The folder a card's layer notes are judged on: the campaign's own
    when it has templates; else the shared folder when anything loose is
    saved there; else the backup zip's set (what the folder will be
    seeded with on its first run). A wrong guess here is not cosmetic --
    a section greyed out is not posted, so its words never reach the
    layer."""
    folder = product_templates_dir(product_name, campaign_name=campaign_name)
    if _has_saved_templates(folder):
        return folder
    if _has_saved_templates(DEFAULT_TEMPLATES_DIR):
        return DEFAULT_TEMPLATES_DIR
    return _seed_templates_dir() or DEFAULT_TEMPLATES_DIR


ENV_FILE = BASE_DIR / ".env"


def _ideogram_key_status() -> dict:
    """What the page may say about the Ideogram key: whether one is set
    and its last four characters. Never the key itself."""
    key = (os.environ.get("IDEOGRAM_API_KEY") or "").strip()
    return {"set": bool(key), "hint": key[-4:] if len(key) >= 8 else ""}


# The month's DeepL allowance, remembered between renders. This function
# runs from a context processor -- on EVERY page -- and it used to ask
# DeepL over the network each time, with a 20 second timeout. One slow
# reply from a vendor and the form itself took 20 seconds to draw, for a
# number nobody was waiting on.
_DEEPL_USAGE = {"at": 0.0, "used": None, "limit": None}
_DEEPL_USAGE_TTL_SECONDS = 600
_DEEPL_USAGE_LOCK = threading.Lock()
_DEEPL_USAGE_REFRESHING = False


def _refresh_deepl_usage_in_the_background() -> None:
    """Ask DeepL what the month looks like, off the request. At most one
    of these is in flight; a failure leaves the last known figures
    alone rather than blanking them."""
    global _DEEPL_USAGE_REFRESHING

    with _DEEPL_USAGE_LOCK:
        if _DEEPL_USAGE_REFRESHING:
            return
        _DEEPL_USAGE_REFRESHING = True

    def work():
        global _DEEPL_USAGE_REFRESHING
        try:
            from src.translation_providers import DeepLProvider

            usage = DeepLProvider().usage()
            if usage:
                _DEEPL_USAGE.update({
                    "at": time.time(),
                    "used": usage.get("character_count"),
                    "limit": usage.get("character_limit"),
                })
            else:
                # Reachable but unhappy: back off for the full period
                # rather than retrying on every render.
                _DEEPL_USAGE["at"] = time.time()
        except Exception:  # noqa: BLE001
            _DEEPL_USAGE["at"] = time.time()
        finally:
            with _DEEPL_USAGE_LOCK:
                _DEEPL_USAGE_REFRESHING = False

    threading.Thread(target=work, daemon=True).start()


def _deepl_key_status() -> dict:
    """What the page may say about the DeepL key: whether one is set, its
    last four characters, and how much of the month's allowance is left.
    Never the key itself.

    The usage figure is the reason to have a key at all -- a number you
    can look at before a demo, rather than finding out mid-demo that an
    invisible quota ran out hours ago. It is read from the last answer
    DeepL gave, never fetched here: this runs on every page render, and
    a page must not wait on a vendor to draw. The first render after a
    start shows the key without a figure, and the number appears on the
    next one. The template already draws it that way."""
    from src.translation_providers import DeepLProvider

    provider = DeepLProvider()
    status = {"set": provider.is_configured(), "hint": provider.api_key[-4:] if len(provider.api_key) >= 8 else ""}
    if not status["set"]:
        return status
    if time.time() - _DEEPL_USAGE["at"] > _DEEPL_USAGE_TTL_SECONDS:
        _refresh_deepl_usage_in_the_background()
    # Best effort: a key that works but whose usage call fails should
    # still read as "set", not as a problem.
    if _DEEPL_USAGE["limit"] is not None:
        status["used"] = _DEEPL_USAGE["used"]
        status["limit"] = _DEEPL_USAGE["limit"]
    return status


def _brief_choices() -> list:
    """One entry per product in every brief file under briefs/ -- the
    campaign fields it fills (product, market, audience, message), the
    brand colours, the product's prompt hint and headline. Read fresh
    each time, so a brief edited while the app runs shows up on the next
    page load. A file that won't parse is skipped: the form must still
    open."""
    try:
        from src.brief_loader import load_brief
    except Exception:  # noqa: BLE001
        return []
    choices = []
    try:
        files = sorted(p for p in BRIEFS_DIR.iterdir() if p.suffix.lower() in (".json", ".yaml", ".yml"))
    except OSError:
        return []
    for path in files:
        try:
            brief = load_brief(str(path))
        except Exception:  # noqa: BLE001
            continue
        colors = [c for c in (brief.brand.colors or []) if isinstance(c, str) and c.startswith("#")][:3]
        for product in brief.products:
            choices.append({
                "id": f"{path.name}::{product.slug}",
                "label": f"{brief.name} -- {product.name}",
                "file": path.name,
                "campaign": brief.name,
                "product_name": product.name,
                "market": brief.target_region,
                "audience": brief.target_audience,
                "campaign_message": brief.message,
                "headline": product.headline or brief.headline or "",
                "prompt_hint": product.prompt_hint or "",
                "language": brief.language or market_copy_languages().get((brief.target_region or "").strip().lower(), ""),
                "colors": colors,
            })
    return choices


def seed_product_template_folders() -> list:
    """At start-up: a templates folder for every product in every brief
    file -- default_templates/HydroBoost Sports Drink/ and so on -- each
    unpacked from the backup zip, so the folders are there to look at
    (and to edit in Photoshop) before a single run. A folder that
    already exists is left exactly as it is: this never overwrites a
    product's own templates. Returns the folders it made."""
    made = []
    for choice in _brief_choices():
        name = choice.get("product_name") or ""
        campaign = choice.get("campaign") or ""
        parts = _campaign_folder_parts(name, campaign)
        if not parts:
            continue
        folder = DEFAULT_TEMPLATES_DIR.joinpath(*parts)
        if folder.is_dir():
            continue
        try:
            if product_templates_dir(name, create=True, campaign_name=campaign) == folder and folder.is_dir():
                made.append(folder)
        except Exception:  # noqa: BLE001
            continue
    return made


_seeded_product_folders = False


@app.before_request
def _seed_product_folders_once():
    """Run the seeding on the first request too, for launches that don't
    go through the __main__ block (flask run, a WSGI server). Skipped
    under test, where the folders would land in a scratch directory."""
    global _seeded_product_folders
    if _seeded_product_folders or app.config.get("TESTING"):
        return
    _seeded_product_folders = True
    try:
        seed_product_template_folders()
    except Exception:  # noqa: BLE001
        pass


@app.context_processor
def _inject_settings():
    return {
        "brief_choices": _brief_choices(),
        "backup_zip_sizes": backup_zip_sizes(),
        "template_sizes_status": template_sizes_status,
        "brief_campaign_by_product": brief_campaign_by_product(),
        "brief_campaigns_by_product": brief_campaigns_by_product(),
        "default_campaign_name": default_campaign_name(),
        "market_copy_languages": market_copy_languages(),
        "ideogram_key": _ideogram_key_status(),
        "deepl_key": _deepl_key_status(),
        "env_file": str(ENV_FILE),
        "ideogram_speeds": IDEOGRAM_SPEED_CHOICES,
        "default_ideogram_speed": DEFAULT_IDEOGRAM_SPEED,
        "reference_limit": REFERENCE_LIMIT,
        "reference_slots": REFERENCE_SLOTS,
        "content_psd_label": CONTENT_PSD_LABEL,
        "max_psd_templates": MAX_PSD_TEMPLATES,
        "known_sizes": sorted(
            {f"{w}x{h}" for w, h in list(SIZE_NAMES) + list(DEFAULT_SIZES) + list(_default_template_paths())}
        ),
        "copy_languages": COPY_LANGUAGES,
        "psd_rows_shown": PSD_TEMPLATE_ROWS_SHOWN,
    }


def save_env_value(name: str, value: str, env_file: Path = None) -> Path:
    """Write NAME=value into the .env beside the app, keeping every other
    line. A missing .env is started from .env.example so the comments
    that explain the other settings come along. The running process
    picks the value up at once via os.environ -- no restart."""
    env_file = env_file or ENV_FILE
    if not env_file.exists():
        example = env_file.parent / ".env.example"
        env_file.write_text(example.read_text() if example.is_file() else "")
    lines = env_file.read_text().splitlines()
    written = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{name}="):
            lines[i] = f"{name}={value}"
            written = True
            break
    if not written:
        lines.append(f"{name}={value}")
    env_file.write_text("\n".join(lines) + "\n")
    os.environ[name] = value
    return env_file


@app.route("/reference-thumb")
def reference_thumb():
    """A small preview of a web picture on the mood board, fetched by
    the app rather than the browser. Sites that block hotlinking (and
    browsers that block cross-site images) left the chip with a broken
    image; this way the thumbnail comes from here, and a bad address
    shows as one straight away instead of at Generate time."""
    import tempfile

    url = (request.args.get("url") or "").strip()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            path = _fetch_web_image(url, Path(tmp))
            image = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}, 422
        image.thumbnail((160, 160))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    response = send_file(buf, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "private, max-age=3600"
    return response


TEMPLATE_RESET_ZIP_NAME = "template-backup.zip"
# The names the reset looks for, in order. The first is the documented
# one; the others are what a zip of the folder is naturally called.
TEMPLATE_RESET_ZIP_NAMES = (TEMPLATE_RESET_ZIP_NAME, "default_templates.zip", "default-templates.zip")


def template_reset_zip() -> Path | None:
    """The zip the reset restores from: one of the known names in
    default_templates/, or -- when none of those is there -- the only
    .zip in the folder, whatever it is called. None when there isn't one."""
    for name in TEMPLATE_RESET_ZIP_NAMES:
        candidate = DEFAULT_TEMPLATES_DIR / name
        if candidate.is_file():
            return candidate
    zips = sorted(p for p in DEFAULT_TEMPLATES_DIR.glob("*.zip") if p.is_file())
    return zips[0] if len(zips) == 1 else None


_template_status_cache: dict = {}


def template_sizes_status(product_name=None, campaign_name=None) -> list:
    """Each size the backup zip carries, and whether this product's saved
    template for it still matches the zip's copy:

        [{"size": "720x1280", "state": "changed"}, ...]

    state is "same", "changed" (edited, or replaced by an upload the run
    promoted) or "missing" (no file for that size). It is what turns the
    restore picker from nine identical labels into a list that says which
    one you actually meant -- the size showing artwork you did not expect
    is the changed one.

    Compared on CRC32, which the zip already stores per entry, so nothing
    has to be decompressed. Cached against the zip's timestamp and each
    file's size and timestamp, since this runs on every page load.
    """
    zip_path = template_reset_zip()
    if zip_path is None:
        return []
    folder = product_templates_dir(product_name, campaign_name=campaign_name)
    try:
        files = sorted(folder.glob("*.psd"))
        stamp = (
            str(zip_path), zip_path.stat().st_mtime_ns, str(folder),
            tuple((f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in files),
        )
    except OSError:
        return []
    cached = _template_status_cache.get(stamp)
    if cached is not None:
        return cached

    import zlib
    by_name = {f.name: f for f in files}
    out = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                name = Path(info.filename).name
                if info.is_dir() or info.filename.startswith("__MACOSX/") or name.startswith("._"):
                    continue
                if name.lower().rsplit(".", 1)[-1] != "psd":
                    continue
                found = SIZE_IN_NAME_RE_LOOSE.search(name)
                if not found:
                    continue
                here = by_name.get(name)
                if here is None:
                    state = "missing"
                else:
                    try:
                        state = "same" if zlib.crc32(here.read_bytes()) == info.CRC else "changed"
                    except OSError:
                        state = "same"   # unreadable: don't cry wolf
                out.append({"size": found.group(0).lower(), "state": state})
    except (OSError, zipfile.BadZipFile):
        return []

    # Sizes that go back together, so the picker can say so before the
    # click rather than the flash message saying it afterwards.
    for entry in out:
        entry["with"] = [x for x in sizes_sharing_shape(entry["size"]) if x != entry["size"]]

    def area(entry):
        w, h = entry["size"].split("x")
        return int(w) * int(h)

    out.sort(key=area, reverse=True)
    # An empty folder has not been seeded, so nothing in it changed; marking every
    # size "missing" would be wrong. The list goes out unlabelled.
    if out and all(entry["state"] == "missing" for entry in out):
        for entry in out:
            entry["state"] = "same"
    _template_status_cache.clear()   # one product's page at a time; keep it small
    _template_status_cache[stamp] = out
    return out


def sizes_sharing_shape(size: str) -> list:
    """Every size in the backup zip with the same proportions as `size`,
    largest first -- 1080x1920 and 720x1280 are both 9:16.

    An uploaded template already spreads this way: it "carries onto any
    other size in the batch with exactly the same proportions that has no
    file of its own". Putting one back has to spread the same way, or a
    9:16 upload is removed from one size and left standing on the other,
    which reads as the restore not working.
    """
    try:
        w, h = (int(part) for part in size.lower().split("x"))
    except (AttributeError, ValueError):
        return []
    if not w or not h:
        return []
    shape = Fraction(w, h)
    out = []
    for label in backup_zip_sizes():
        lw, lh = (int(part) for part in label.split("x"))
        if lh and Fraction(lw, lh) == shape:
            out.append(label)
    return out or [size]


def backup_zip_sizes() -> list:
    """The size labels the backup zip carries ("720x480"), largest first
    -- what the per-size restore beside a card's Reset offers. Empty when
    there is no zip, which is what hides that control."""
    zip_path = template_reset_zip()
    if zip_path is None:
        return []
    sizes = set()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                name = Path(info.filename).name
                if info.is_dir() or info.filename.startswith("__MACOSX/") or name.startswith("._"):
                    continue
                if name.lower().rsplit(".", 1)[-1] != "psd":
                    continue
                found = SIZE_IN_NAME_RE_LOOSE.search(name)
                if found:
                    sizes.add(found.group(0).lower())
    except (OSError, zipfile.BadZipFile):
        return []
    def area(label):
        w, h = label.split("x")
        return int(w) * int(h)
    return sorted(sizes, key=area, reverse=True)


def restore_templates_from_backup(
    zip_path: Path | None = None,
    dest_dir: Path | None = None,
    only_sizes: set | None = None,
) -> tuple[list[str], list[str], str | None]:
    """Put the saved templates back to the copies in the backup zip.

    Every tester-WxH.psd in the zip replaces the one in `dest_dir` --
    default_templates/ itself, or a product's own folder under it (the
    one being replaced is moved to _template_backups/ first, stamped, so
    nothing is lost). Sizes the zip does not carry are left as they are.
    Returns (restored names, untouched sizes, error message).

    `only_sizes` narrows it to those size labels ("720x480"), for putting
    one size back after an edit went wrong without disturbing the other
    eight. None means every size in the zip, which is what the card's
    Reset does."""
    if zip_path is None:
        zip_path = template_reset_zip()
    if dest_dir is None:
        dest_dir = DEFAULT_TEMPLATES_DIR
    if zip_path is None or not zip_path.is_file():
        return [], [], (
            f"No backup zip in {DEFAULT_TEMPLATES_DIR.name}/ -- nothing to restore from. "
            f"Put the tester-WxH.psd files in a zip named {TEMPLATE_RESET_ZIP_NAME} "
            "(or default_templates.zip) in that folder."
        )
    restored: list[str] = []
    stamp = time.strftime("%Y%m%d-%H%M%S")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                name = Path(info.filename).name
                # Finder's zips carry a __MACOSX/._name shadow for every
                # file; only the real PSDs, named for a size, count.
                if info.is_dir() or info.filename.startswith("__MACOSX/") or name.startswith("._"):
                    continue
                if name.lower().rsplit(".", 1)[-1] != "psd" or not SIZE_IN_NAME_RE_LOOSE.search(name):
                    continue
                if only_sizes is not None:
                    found = SIZE_IN_NAME_RE_LOOSE.search(name)
                    if not found or found.group(0).lower() not in only_sizes:
                        continue
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / name
                if dest.exists():
                    TEMPLATE_BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(dest), str(TEMPLATE_BACKUPS_DIR / f"{dest.stem}.{stamp}{dest.suffix}"))
                with zf.open(info) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
                restored.append(name)
    except (OSError, zipfile.BadZipFile) as exc:
        return restored, [], f"Could not restore the templates from {zip_path.name}: {exc}"
    restored_sizes = {m.group(0).lower() for m in (SIZE_IN_NAME_RE_LOOSE.search(n) for n in restored) if m}
    untouched = sorted(
        {
            m.group(0)
            for m in (SIZE_IN_NAME_RE_LOOSE.search(f.name) for f in dest_dir.glob("*.psd"))
            if m and m.group(0).lower() not in restored_sizes
        }
    )
    return restored, untouched, None


def reset_keeps_fields() -> tuple:
    """What a reset keeps: the campaign brief (product, market, audience,
    message), the brand colours and the copy language -- the campaign's
    identity, which is not what anyone means by "start over"."""
    return CAMPAIGN_BRIEF_FIELD_NAMES + BRAND_COLOR_FIELD_NAMES + ("copy_language",)


def forget_remembered_form(keep_from=None) -> bool:
    """Wipe what the next form would otherwise open with -- every
    section's typed values and switches, and the pointer to the last
    run's files (custom hero, logo, product, CTA, references, the
    size-specific PSD rows) -- but keep the campaign brief, brand
    colours and copy language (reset_keeps_fields()), taken from
    `keep_from` (the form that asked for the reset) when given, else
    from what was remembered. The Ideogram key lives in .env and is not
    touched. Returns whether anything was there to forget."""
    prefs = _load_preferences()
    had_anything = bool(prefs)
    products = _product_memories(prefs)
    key = _product_memory_key(keep_from.get("product_name"), keep_from.get("campaign_name")) if keep_from is not None else ""
    source = products.get(key) if key else None
    if source is None:
        source = prefs
    kept = {}
    for name in reset_keeps_fields():
        if keep_from is not None and keep_from.get(name) is not None:
            kept[name] = (keep_from.get(name) or "").strip()
        elif name in source:
            kept[name] = source[name]
    if key:
        # This product's entry becomes just the brief; other products are untouched.
        # The top level follows, since it was the product last used.
        products = {k: v for k, v in products.items() if k != key}
        products[key] = dict(kept)
        new_prefs = dict(kept)
        new_prefs["products"] = products
    else:
        new_prefs = dict(kept)
        if products:
            new_prefs["products"] = products
    _save_preferences(new_prefs)
    return had_anything


def _remove_product_from_briefs(campaign: str, product: str) -> str:
    """Take a product out of its campaign's brief, and the brief with it
    when that was the last one. Returns a note, or "" if nothing changed."""
    path = _brief_file_for_campaign(campaign)
    if path is None:
        return ""
    try:
        if path.suffix.lower() in (".yaml", ".yml"):
            from ruamel.yaml import YAML

            yaml = YAML()
            yaml.preserve_quotes = True
            with open(path, encoding="utf-8") as handle:
                data = yaml.load(handle)
        else:
            yaml = None
            data = json.loads(path.read_text(encoding="utf-8"))
        products = (data.get("campaign") or {}).get("products") or []
        kept = [p for p in products if str(p.get("name") or "").strip().lower() != product.lower()]
        if len(kept) == len(products):
            return ""
        if not kept:
            # Nothing left to describe: the brief goes where the
            # templates go, recoverable rather than erased.
            grave = _grave_dir() / path.name
            shutil.move(str(path), str(grave))
            _brief_campaign_cache.clear()
            return f"briefs/{path.name} held only {product}, so it moved to _to_delete/ too."
        data["campaign"]["products"] = kept
        spare = path.with_suffix(path.suffix + ".tmp")
        if yaml is not None:
            with open(spare, "w", encoding="utf-8") as handle:
                yaml.dump(data, handle)
        else:
            spare.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        spare.replace(path)
        _brief_campaign_cache.clear()
        return f"{product} removed from briefs/{path.name}."
    except Exception as exc:  # noqa: BLE001
        return f"Couldn't update briefs/{path.name}: {type(exc).__name__}: {exc}"


def _grave_dir():
    """_to_delete/, made on demand. Gitignored, and the one place this
    app puts things it is told to get rid of -- nothing is erased."""
    grave = BASE_DIR / "_to_delete"
    grave.mkdir(parents=True, exist_ok=True)
    return grave


@app.route("/delete-campaign", methods=["POST"])
def delete_campaign():
    """Remove one product-in-campaign: its templates, its brief entry and
    its remembered form.

    A card is one product inside one campaign, so this deletes that and
    not every product sharing the campaign. Templates are MOVED to
    _to_delete/, never erased: they are the one thing here a person makes
    by hand, and _template_backups/ only keeps copies of files this app
    overwrote, not ones it removed.

    The remembered form goes too. Leaving it behind is what brings a
    deleted card back on the next page load.
    """
    payload = request.get_json(silent=True) or {}
    product = (payload.get("product_name") or "").strip()
    campaign = (payload.get("campaign_name") or "").strip()
    if not product:
        return jsonify({"error": "no product to remove"}), 400

    notes = []
    folder = DEFAULT_TEMPLATES_DIR.joinpath(*_campaign_folder_parts(product, campaign))
    if folder.is_dir() and folder != DEFAULT_TEMPLATES_DIR:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        label = "-".join(part for part in (campaign, product) if part) or product
        grave = _grave_dir() / f"{_slugify_for_filename(label) or 'campaign'}-{stamp}"
        try:
            shutil.move(str(folder), str(grave))
            notes.append(f"templates moved to _to_delete/{grave.name}")
            # An emptied campaign folder is clutter, not data.
            parent = folder.parent
            if parent != DEFAULT_TEMPLATES_DIR and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError as exc:
            return jsonify({"error": f"couldn't move the templates: {exc}"}), 500

    brief_note = _remove_product_from_briefs(campaign, product)
    if brief_note:
        notes.append(brief_note)

    # Every session slot pointing at this product-in-campaign is stale
    # now. The index only ever gained slots -- generate() files a job
    # under whatever campaign_slot its card posted, and removing a card
    # renumbered the page without touching what was recorded -- so a slot
    # outlived its card and later showed up as a second, identical card
    # on the Edit page. Two slots for one product in one SESSION are
    # legitimate (two cards generated side by side); a slot for a product
    # that no longer exists is not.
    pruned = 0
    sessions = JOBS_DIR / "_sessions"
    if sessions.is_dir():
        for index_path in sessions.glob("*.json"):
            try:
                index = json.loads(index_path.read_text())
                slots = index.get("slots") or {}
                keep = {}
                for slot_key, slot_job in slots.items():
                    state = JOBS_DIR / str(slot_job) / "form_state.json"
                    try:
                        f = json.loads(state.read_text()).get("fields") or {}
                    except (OSError, ValueError):
                        keep[slot_key] = slot_job
                        continue
                    same = (
                        (f.get("product_name") or "").strip().lower() == product.lower()
                        and (f.get("campaign_name") or "").strip().lower() == campaign.lower()
                    )
                    if same:
                        pruned += 1
                    else:
                        keep[slot_key] = slot_job
                if len(keep) != len(slots):
                    index["slots"] = keep
                    index_path.write_text(json.dumps(index))
            except (OSError, ValueError):
                continue
    if pruned:
        notes.append(f"{pruned} session slot{'s' if pruned != 1 else ''} cleared")

    prefs = _load_preferences()
    products = _product_memories(prefs)
    key = _product_memory_key(product, campaign)
    if products.pop(key, None) is not None:
        prefs["products"] = products
        _save_preferences(prefs)
        notes.append("remembered form cleared")

    return jsonify({"ok": True, "notes": notes})


@app.route("/reset-size", methods=["POST"])
def reset_one_size():
    """Put one size's template back to the copy in the backup zip.

    The card's Reset is all or nothing: it restores every size and clears
    the form, which is a lot to lose when a single size's PSD was edited
    into a state you want to undo. This restores just that one, keeps the
    replaced file in _template_backups/ like every other write over a
    saved template, and leaves the form alone."""
    product_name = (request.form.get("product_name") or "").strip()
    campaign_name = (request.form.get("campaign_name") or "").strip()
    size = (request.form.get("reset_size") or "").strip().lower()
    if not SIZE_IN_NAME_RE_LOOSE.fullmatch(size):
        flash("Pick the size to put back first.")
        return redirect(url_for("index"))

    dest_dir = product_templates_dir(product_name, campaign_name=campaign_name)
    # An upload for one 9:16 size is the template for every 9:16 size without its own file, so a
    # restore must put the whole shape back or it looks like it did not take.
    group = sizes_sharing_shape(size)
    restored, _untouched, error = restore_templates_from_backup(dest_dir=dest_dir, only_sizes=set(group))
    if error:
        flash(error)
    elif not restored:
        flash(f"The backup zip has no template for {size}, so nothing was changed.")
    else:
        where = "default_templates/" + (
            f"{dest_dir.relative_to(DEFAULT_TEMPLATES_DIR).as_posix()}/" if dest_dir != DEFAULT_TEMPLATES_DIR else ""
        )
        # A row still set to this size would override the template that
        # was just put back, so the restore would appear to do nothing.
        product_key = _product_memory_key(product_name, campaign_name)
        rows = []
        for label in group:
            for row in forget_psd_row_for_size(product_key, label):
                if row not in rows:
                    rows.append(row)
        # Rows the posted form sets to this size, which is what matters on an Edit page:
        # there the files come from the job being edited, not from the preferences.
        posted_rows = [
            i for i in range(1, MAX_PSD_TEMPLATES + 1)
            if (request.form.get(f"psd_size_{i}") or "").strip().lower() in group
        ]
        drop_row_files_for_job(request.form.get("edit_job_id") or "", posted_rows or rows, product_key)
        for i in posted_rows:
            if i not in rows:
                rows.append(i)
        rows.sort()
        note = ""
        if rows:
            note = (
                f" The size-specific PSD {'row' if len(rows) == 1 else 'rows'} "
                f"{', '.join(str(r) for r in rows)} held an upload for {size}; "
                "that is let go, so the restored template is what renders."
            )
        flash(
            (f"{product_name}: " if product_name else "")
            + f"{', '.join(restored)} restored into {where} from "
            + f"{(template_reset_zip() or Path(TEMPLATE_RESET_ZIP_NAME)).name}. "
            + "The file it replaced is in _template_backups/." + note
            + (
                f" {size} and {', '.join(x for x in group if x != size)} are the same shape, so they went back together."
                if len(group) > 1 else ""
            )
            + " They render exactly like every other size from now on, from the templates just put back.",
            "ok",
        )
    return redirect(url_for("index"))


@app.route("/reset", methods=["POST"])
def reset_form():
    """The "Reset form" button. A blank form -- every remembered field and
    every file kept from the last run forgotten -- and the saved
    templates put back to the ones in default_templates/template-backup.zip:
    the way to undo a run of drops and Photoshop edits and start over
    from the known-good set."""
    # Reset posts the whole card, so the product comes with it and its own folder is
    # what gets restored. No product means the shared set.
    product_name = (request.form.get("product_name") or "").strip()
    campaign_name = (request.form.get("campaign_name") or "").strip()
    dest_dir = product_templates_dir(product_name, campaign_name=campaign_name)
    restored, untouched, error = restore_templates_from_backup(dest_dir=dest_dir)
    forget_remembered_form(keep_from=request.form)
    where = "default_templates/" + (
        f"{dest_dir.relative_to(DEFAULT_TEMPLATES_DIR).as_posix()}/" if dest_dir != DEFAULT_TEMPLATES_DIR else ""
    )
    cleared = (
        "Every other field was cleared and nothing is carried over from your last run; "
        "the campaign brief, brand colours and copy language were kept."
    )
    if error:
        flash(error)
        flash(cleared, "ok")
    else:
        message = (
            (f"{product_name}: " if product_name else "")
            + f"templates restored into {where} from {(template_reset_zip() or Path(TEMPLATE_RESET_ZIP_NAME)).name}: "
            + ", ".join(restored) + "."
        )
        if untouched:
            message += f" Not in the zip, so left as they were: {', '.join(untouched)}."
        message += " The replaced files are in _template_backups/. " + cleared
        flash(message, "ok")
    return redirect(url_for("index"))


@app.route("/settings/ideogram-key", methods=["POST"])
def set_ideogram_key():
    """The box at the top of the form. For someone running the packaged
    app this is the whole key setup: paste, save, done -- no hidden file
    to find and no editor. The key is stored in .env beside the app and
    is never sent back to the browser."""
    key = (request.form.get("ideogram_api_key") or "").strip()
    if not key:
        flash("Paste the Ideogram API key before saving.")
        return redirect(url_for("index"))
    if any(ch.isspace() for ch in key) or len(key) < 20:
        flash("That doesn't look like an Ideogram API key -- check it was copied whole.")
        return redirect(url_for("index"))
    save_env_value("IDEOGRAM_API_KEY", key)
    flash(f"Ideogram key saved (ends in {key[-4:]}). Choose Ideogram as the provider under Layer images to use it.", "ok")
    return redirect(url_for("index"))


@app.route("/settings/deepl-key", methods=["POST"])
def set_deepl_key():
    """Same box, same deal as the Ideogram key: paste, save, done. Stored
    in .env beside the app and never sent back to the browser.

    With a key set, copy is translated by DeepL and the free Google
    endpoint becomes the fallback rather than the plan -- which is what
    stops a day's invisible quota deciding what language a campaign
    ships in."""
    key = (request.form.get("deepl_api_key") or "").strip()
    if not key:
        flash("Paste the DeepL API key before saving.")
        return redirect(url_for("index"))
    if any(ch.isspace() for ch in key):
        flash("That key has a space in it -- check it was copied whole.")
        return redirect(url_for("index"))
    # Ask DeepL whether the key works instead of guessing from its shape.
    # The first version of this box accepted anything over 20 characters,
    # so an Ideogram key pasted by mistake was stored, reported as "set",
    # and quietly produced English copy for hours.
    from src.translation_providers import DeepLProvider

    ok, detail = DeepLProvider(key).verify()
    if not ok:
        flash(detail)
        return redirect(url_for("index"))
    save_env_value("DEEPL_API_KEY", key)
    free = " (free tier)" if key.endswith(":fx") else ""
    flash(f"DeepL key saved (ends in {key[-4:]}){free} -- {detail}.", "ok")
    return redirect(url_for("index"))


# Brand-level settings that outlive a run, kept in a small JSON beside the jobs so a
# test's temp JOBS_DIR gets its own.
BRAND_COLOR_FIELD_NAMES = tuple(
    name for i in (1, 2, 3) for name in (f"brand_color_{i}", f"brand_color_{i}_enabled")
)
# Hero-image and size-specific PSD sections: a fresh form reopens on the last run's
# layout and copy rather than blank.
SECTION_FIELD_PREFIXES = ("layer_", "psd_", "upload_hero", "upload_custom_hero", "upload_ai_enabled")
# Hide boxes are not remembered. Carried over, a box ticked once stayed ticked and
# later runs came back as blank canvases.
REMEMBERED_SECTION_FIELDS = tuple(
    name for name in EDIT_TEXT_FIELD_NAMES + EDIT_CHECKBOX_FIELD_NAMES
    if name.startswith(SECTION_FIELD_PREFIXES) and not name.endswith("_hidden")
)
CAMPAIGN_BRIEF_FIELD_NAMES = ("campaign_name", "product_name", "market", "audience", "campaign_message")
# psd_make_saved persists and starts ticked: ticked, a dropped PSD replaces the saved
# template for its size. The results page restates it every run.
REMEMBERED_FIELD_NAMES = (
    CAMPAIGN_BRIEF_FIELD_NAMES + BRAND_COLOR_FIELD_NAMES + ("copy_language", "psd_make_saved") + REMEMBERED_SECTION_FIELDS
)
REMEMBERED_FILE_PREFIXES = ("upload_hero_image", "layer_", "psd_file_")

# Languages the copy typed on the form can be drawn in. The English stays
# on the form; each run translates it on the way into the templates.
COPY_LANGUAGES = (("en", "English"), ("fr", "Français"), ("es", "Español"))
COPY_LANGUAGE_NAMES = {code: name for code, name in COPY_LANGUAGES}


def market_copy_languages() -> dict:
    """{market as typed, lowercased: copy language code} for the markets
    whose language the form can draw in -- the CLI's region table plus
    the codes and a few spellings people type ("FR", "France", "Québec").
    The page script sets Copy language from the Market field with it;
    a market it doesn't know leaves the language alone."""
    from src.localization import REGION_TO_LANGUAGE

    offered = {code for code, _ in COPY_LANGUAGES}
    table = {region: code for region, code in REGION_TO_LANGUAGE.items() if code in offered}
    table.update({
        "fr": "fr", "fra": "fr", "french": "fr", "belgium": "fr", "quebec": "fr", "québec": "fr",
        "switzerland": "fr", "es": "es", "esp": "es", "spanish": "es", "argentina": "es", "colombia": "es",
        "chile": "es", "peru": "es", "latam": "es", "us": "en", "uk": "en", "gb": "en", "australia": "en",
        "ireland": "en", "new zealand": "en", "en": "en", "english": "en",
    })
    return table
COPY_LANGUAGE_ENGLISH_NAMES = {"en": "English", "fr": "French", "es": "Spanish"}


def _translation_cache_path() -> Path:
    return JOBS_DIR / "translations.json"


def _translate_copy(text, language: str, cache: dict):
    """(translated text, ok). Repeats of a phrase come from the cache
    so an edit-and-rerun doesn't call the translator again for copy
    that hasn't changed; a failed call returns the English with ok=False
    so the run can say so instead of quietly shipping the wrong
    language."""
    if not text or language == "en":
        return text, True
    key = f"{language}\u0000{text}"
    if key in cache:
        return cache[key], True
    # One call from here. Retrying the odd dropped call is worth doing, but it
    # now happens inside localize_message, which also spaces its requests -- two
    # layers of retry meant a rate-limited field waited out the backoff twice.
    translated, ok = localize_message(text, language)
    if ok and translated:
        # Text returned unchanged is not a translation, so don't cache it as one:
        # a cached self-mapping stood in front of the real English.
        if translated.strip() != text.strip():
            cache[key] = translated
        return translated, True
    return text, False


def _english_behind_translations() -> dict:
    """{translated text: the English it was made from}, from the
    translation cache, every language together. A template that is
    itself an export from a French or Spanish run carries that language
    in its type layers; this is how the run finds the English again --
    to draw it when the language is set back to English, and to
    translate from it (not from the Spanish) when another is chosen."""
    try:
        cache = json.loads(_translation_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(cache, dict):
        return {}
    reverse = {}
    for key, translated in cache.items():
        if "\u0000" not in key or not isinstance(translated, str):
            continue
        english = key.split("\u0000", 1)[1]
        # A phrase returned unchanged maps to itself; taken first, that entry hid the
        # real English source.
        if translated.strip() == english.strip():
            continue
        reverse.setdefault(translated.strip(), english.strip())
    return reverse


def _same_words(a, b) -> bool:
    """Whether two pieces of copy say the same thing, ignoring how the
    lines are broken and spaced (a type layer breaks lines with \\r, the
    form with \\n, and either may carry a trailing space)."""
    def norm(text):
        # Curly and straight quotes are the same words: a translator
        # hands back l'hiver, Photoshop typesets l’hiver.
        text = (text or "")
        for curly, plain in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"'), ("\u2013", "-"), ("\u2014", "-"), ("\u00a0", " ")):
            text = text.replace(curly, plain)
        lines = [
            " ".join(line.split())
            for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        ]
        return "\n".join(line for line in lines if line)
    return norm(a) == norm(b)


PLACEHOLDER_COPY_MARKERS = ("lorem ipsum",)


def _is_placeholder_copy(words) -> bool:
    """Whether a type layer's words are filler -- "Lorem ipsum..." -- or
    nothing at all: copy no one wrote for the campaign."""
    text = " ".join((words or "").split()).lower()
    return not text or any(marker in text for marker in PLACEHOLDER_COPY_MARKERS)


def _campaign_copy_from_templates(size_template_paths: dict, english_behind: dict) -> dict:
    """{layer: English copy} -- what the campaign's templates say, taken
    from the sizes that carry real copy. A template whose description
    is Lorem ipsum (a size added to the set without its copy) gets the
    campaign's description from here, the way its header already got
    the campaign's headline. English via the translation cache when the
    template holds a French or Spanish export; the copy most sizes agree
    on wins, ties going to the larger size."""
    votes: dict = {}
    for (width, height), path in sorted(size_template_paths.items(), key=lambda item: -(item[0][0] * item[0][1])):
        try:
            own = get_psd_text_layers(path, visible_only=True) or {}
        except Exception:  # noqa: BLE001
            continue
        for layer_key in ("header", "description", "legal"):
            words = (own.get(layer_key) or "").strip()
            if _is_placeholder_copy(words):
                continue
            english = _english_source_of(words, english_behind) or words
            votes.setdefault(layer_key, []).append(english)
    out = {}
    for layer_key, texts in votes.items():
        counts: dict = {}
        for text in texts:
            counts[text] = counts.get(text, 0) + 1
        out[layer_key] = max(texts, key=lambda t: (counts[t], -texts.index(t)))
    return out


def _english_source_of(words: str, reverse: dict):
    """The English behind `words` (line by line, so a header translated
    line by line reverses the same way), or None if `words` isn't a
    translation this machine made."""
    lines = [line.strip() for line in words.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
    if not lines or not reverse:
        return None

    def back(text):
        # One step back, None when `text` isn't a translation. Followed repeatedly:
        # Spanish translated again is Spanish-of-Spanish, two steps from English.
        source = reverse.get(text)
        return source if source and source != text else None

    def all_the_way(text):
        seen = {text}
        current = text
        for _ in range(10):
            previous = back(current)
            if not previous or previous in seen:
                break
            seen.add(previous)
            current = previous
        return current if current != text else None

    if len(lines) == 1:
        return all_the_way(lines[0])
    english_lines = [all_the_way(line) or line for line in lines]
    if english_lines != lines:
        return "\r".join(english_lines)
    return all_the_way("\r".join(lines))


def _localize_form_copy(language: str, fields: dict, notes: list, warnings: list) -> dict:
    """Translate every non-empty copy field into `language`. Returns the
    same dict with the translations in place, and writes one note per
    translated field (the English beside the translation, so what the
    creative says can be checked without knowing the language) and one
    warning if the translator couldn't be reached."""
    if language == "en" or not any(fields.values()):
        return fields
    try:
        cache = json.loads(_translation_cache_path().read_text(encoding="utf-8"))
        if not isinstance(cache, dict):
            cache = {}
    except (OSError, ValueError):
        cache = {}
    before = dict(cache)
    failed = []
    reason = ""
    out = {}
    for name, text in fields.items():
        if reason == "rate limit" and text:
            # The limit is counted per network, not per phrase. Asking for the
            # next field only waits out the same backoff to be told the same
            # thing -- eight fields turned a failed run into a minute of it.
            out[name] = text
            failed.append(name)
            continue
        translated, ok = _translate_copy(text, language, cache)
        out[name] = translated
        if text and ok and translated != text:
            notes.append(
                f"{COPY_LANGUAGE_ENGLISH_NAMES.get(language, language)} copy -- {name}: \"{text}\" -> \"{translated}\"."
            )
        elif text and not ok:
            failed.append(name)
            reason = take_last_failure() or reason
    if failed:
        language_name = COPY_LANGUAGE_ENGLISH_NAMES.get(language, language)
        # Name the actual failure. "Check the connection" sent someone hunting a
        # network fault for an hour when the endpoint was up and simply counting.
        if reason == "quota":
            why = (
                "the DeepL key has used up this month's characters, and the free "
                "endpoint behind it couldn't answer either. Check the usage on your "
                "DeepL account; the allowance resets monthly."
            )
        elif reason == "no key":
            why = (
                "DeepL refused the key. A free key ends in \":fx\" and must be used "
                "against api-free.deepl.com -- the app works that out from the key "
                "itself, so this usually means the key was copied short or has been "
                "revoked. Paste it again at the top of the form."
            )
        elif reason == "unsupported language":
            why = (
                "neither translator offers this language. DeepL doesn't have it, and "
                "the free endpoint couldn't answer."
            )
        elif reason == "rate limit":
            why = (
                "the translator rate-limited this run. Google's free endpoint allows "
                "about five calls a second and 200,000 a day, counted across everyone "
                "sharing your connection -- waiting a few minutes usually clears it, "
                "and copy translated earlier is cached so a rerun asks for less. A "
                "DeepL key avoids this entirely."
            )
        elif reason == "error page":
            why = (
                "the translator answered with an error page instead of a translation, "
                "which is the endpoint having a bad minute rather than anything wrong "
                "with the copy. Run again in a moment."
            )
        else:
            why = (
                "no translator answered. Check the connection and run again."
            )
        warnings.append(
            f"Couldn't translate the {', '.join(failed)} into {language_name} -- {why} "
            "Those are drawn in English."
        )
    if cache != before:
        try:
            _translation_cache_path().write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    return out


def _preferences_path() -> Path:
    return JOBS_DIR / "preferences.json"


def forget_psd_row_for_size(product_key: str, size: str) -> list:
    """Let go of any remembered PSD row set to `size`, and return the row
    numbers dropped.

    Putting a size back to the backup zip only changes the saved template.
    A row still holding an uploaded PSD for that same size would win over
    it on the next run, so the restore would look like it did nothing --
    which is exactly what it looked like. The row's size is cleared and its
    kept file is marked dropped; the file itself stays in the job folder,
    untouched, like every other file from a past run.
    """
    prefs = _load_preferences()
    products = _product_memories(prefs)
    dropped_rows = []

    def forget_in(memory: dict) -> bool:
        job_id = (memory.get("last_job_id") or "").strip()
        changed = False
        for i in range(1, MAX_PSD_TEMPLATES + 1):
            if (memory.get(f"psd_size_{i}") or "").strip().lower() != size:
                continue
            memory[f"psd_size_{i}"] = ""
            changed = True
            if i not in dropped_rows:
                dropped_rows.append(i)
            if job_id:
                marker = f"{job_id}:psd_file_{i}"
                dropped = [d for d in (memory.get("dropped_files") or []) if isinstance(d, str)]
                if marker not in dropped:
                    dropped.append(marker)
                memory["dropped_files"] = dropped
        return changed

    touched = False
    if product_key and product_key in products:
        touched = forget_in(products[product_key]) or touched
    # The top level mirrors the product last used, so it gets the same
    # treatment -- otherwise the row would come back on the next page load.
    touched = forget_in(prefs) or touched
    if products:
        prefs["products"] = products
    _save_preferences(prefs)
    return dropped_rows


def _load_preferences() -> dict:
    try:
        data = json.loads(_preferences_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_preferences(prefs: dict) -> None:
    try:
        _preferences_path().parent.mkdir(parents=True, exist_ok=True)
        _preferences_path().write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    except OSError:
        pass


def _product_memory_key(product_name, campaign_name=None) -> str:
    """How a campaign's product is filed in the remembered form: its
    folder path ("Winter Glow 2026/HydroBoost Sports Drink", or just the
    product with no campaign), so the same product in two campaigns is
    two entries and different spacing is not. Empty for no product."""
    return "/".join(_campaign_folder_parts(product_name, campaign_name))


def _product_memories(prefs: dict) -> dict:
    """{product key: {field: value, ..., "last_job_id": ...}} -- each
    product's own remembered form. Products are isolated: what was
    typed, ticked or uploaded for one never shows up on another."""
    products = prefs.get("products")
    return products if isinstance(products, dict) else {}


def _remember_form_fields(form) -> None:
    """Save the remembered fields' values from this submission, so the
    next fresh form opens with them: the brand colours, and which of
    them are ticked -- re-picking three swatches every run was the
    complaint. An unticked box is saved as unticked (an absent checkbox
    is what "unticked" looks like in a form), not left as it was.

    Saved twice: at the top level (the last product used -- what a form
    with no products yet opens with) and under the product's own entry,
    which is what that product's card opens with."""
    prefs = _load_preferences()
    values = {name: (form.get(name) or "").strip() for name in REMEMBERED_FIELD_NAMES}
    prefs.update(values)
    key = _product_memory_key(form.get("product_name"), form.get("campaign_name"))
    if key:
        products = _product_memories(prefs)
        entry = dict(products.get(key) or {})
        entry.update(values)
        products[key] = entry
        # Most recently used last, so the page shows products in the
        # order they were worked on.
        products = {k: v for k, v in products.items() if k != key}
        products[key] = entry
        prefs["products"] = products
    _save_preferences(prefs)


def _remembered_prefill(product_key: str | None = None) -> dict:
    prefs = _load_preferences()
    source = _product_memories(prefs).get(product_key) if product_key else None
    if source is None:
        source = prefs
    # Empty values come through too: an unticked box that defaults to
    # ticked (psd_make_saved) has to be remembered as unticked.
    return {name: source[name] for name in REMEMBERED_FIELD_NAMES if name in source}


def _remembered_files(product_key: str | None = None):
    """(job id, {field: filename}) for the section files of the last
    run -- the product's own last run when a product is given -- when
    that run's folder is still there; (None, {}) otherwise."""
    prefs = _load_preferences()
    source = _product_memories(prefs).get(product_key) if product_key else None
    if source is None:
        source = prefs
    job_id = (source.get("last_job_id") or "").strip()
    if not job_id or not re.fullmatch(r"[0-9a-f]{12,32}", job_id):
        return None, {}
    state_path = JOBS_DIR / job_id / "form_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, {}
    # Keyed "<job id>:<field>" so the marker dies with its run. The next run writes a
    # new job id, so a stale marker can't hide a fresh upload.
    dropped = {d for d in (source.get("dropped_files") or []) if isinstance(d, str)}
    files = {
        name: filename
        for name, filename in (state.get("files") or {}).items()
        if name.startswith(REMEMBERED_FILE_PREFIXES) and filename
        and f"{job_id}:{name}" not in dropped
        and (JOBS_DIR / job_id / "uploads" / filename).is_file()
    }
    return (job_id if files else None), files


@app.route("/", methods=["GET"])
def index():
    try:
        prune_job_folders()
    except Exception:  # noqa: BLE001
        pass
    return render_template(
        "index.html",
        size_presets=SIZE_PRESET_CHOICES,
        video_extensions=VIDEO_EXTENSIONS,
        build_stamp=BUILD_STAMP,
        campaigns=_remembered_campaign_cards(),
        session_id=uuid.uuid4().hex,
        hideable_layers=HIDEABLE_LAYER_NAMES,
        **_page_layer_sets(),
    )


def _brief_file_for_campaign(campaign_name):
    """The brief file that already holds this campaign, or None."""
    wanted = (campaign_name or "").strip().lower()
    if not wanted:
        return None
    try:
        files = sorted(p for p in BRIEFS_DIR.iterdir() if p.suffix.lower() in (".json", ".yaml", ".yml"))
    except OSError:
        return None
    # Read the name, do not validate the brief. load_brief() enforces the
    # whole schema, so a file it rejects becomes invisible to the code
    # that maintains it -- and removing the second-to-last product from a
    # campaign made its brief unloadable, which meant the last product
    # could never be removed and the brief was orphaned.
    for path in files:
        try:
            if path.suffix.lower() in (".yaml", ".yml"):
                import yaml as _yaml

                data = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
            name = ((data.get("campaign") or data).get("name") or "").strip().lower()
            if name == wanted:
                return path
        except Exception:  # noqa: BLE001
            continue
    return None


def _same_text(a, b) -> bool:
    """Whether two brief values say the same thing.

    A prompt written in the form arrives as one line; the same prompt in
    the file is a folded block scalar wrapped over three. Comparing them
    literally made every run "a change", so ruamel rewrote the whole YAML
    brief in its own style -- reindenting the product list and reflowing
    the comments somebody wrote by hand -- to save a difference nobody
    made.
    """
    return " ".join(str(a or "").split()) == " ".join(str(b or "").split())


def record_run_in_briefs(fields: dict) -> str:
    """Write this run's campaign and product back into briefs/.

    A campaign typed into the form is otherwise invisible to the brief
    picker, so the next page load offers only what was in the files. This
    puts it there: the campaign's own brief gains the product, and a
    campaign no file knows yet gets one of its own.

    Returns a short note for the results page, or "" when nothing needed
    writing. Never raises -- a run must not fail because its brief could
    not be updated.
    """
    campaign = (fields.get("campaign_name") or "").strip()
    product = (fields.get("product_name") or "").strip()
    if not campaign or not product:
        return ""

    # The slug is not a display detail: it names the product's asset file,
    # its output folder and its template folder. A brief that already has
    # one is the authority on it. Regenerating "hydroboost" as
    # "HydroBoost_Sports_Drink" on an ordinary run moved every rendered
    # path at once and pointed the pipeline at folders that don't exist,
    # from a brief nobody had edited. Only a product being written for the
    # first time gets one made up for it.
    entry = {"name": product}
    new_slug = _slugify_for_filename(product) or product.lower()
    if (fields.get("upload_ai_prompt") or "").strip():
        entry["prompt_hint"] = fields["upload_ai_prompt"].strip()
    if (fields.get("upload_ai_headline") or "").strip():
        entry["headline"] = fields["upload_ai_headline"].strip()

    colors = [
        (fields.get(f"brand_color_{i}") or "").strip().upper()
        for i in (1, 2, 3)
        if fields.get(f"brand_color_{i}_enabled") and (fields.get(f"brand_color_{i}") or "").strip()
    ]
    path = _brief_file_for_campaign(campaign)
    if path is not None and path.suffix.lower() in (".yaml", ".yml"):
        return _record_in_yaml_brief(path, campaign, product, entry, new_slug)
    try:
        if path is None:
            path = BRIEFS_DIR / f"{_slugify_for_filename(campaign) or 'campaign'}.json"
            if path.exists():
                return ""
            data = {"campaign": {
                "name": campaign,
                "target_region": (fields.get("market") or "").strip(),
                "target_audience": (fields.get("audience") or "").strip(),
                "message": (fields.get("campaign_message") or "").strip(),
                "brand": {"logo": "assets/brand/logo.png", "colors": colors},
                "products": [dict(entry, slug=new_slug)],
            }}
            written = f"Campaign written to briefs/{path.name} -- it is in the brief list now."
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            products = data.setdefault("campaign", {}).setdefault("products", [])
            for i, existing in enumerate(products):
                if (existing.get("name") or "").strip().lower() == product.lower():
                    merged = dict(existing)
                    for field, value in entry.items():
                        if not _same_text(existing.get(field), value):
                            merged[field] = value
                    merged.setdefault("slug", new_slug)   # only if it never had one
                    if merged == existing:
                        return ""       # nothing changed; leave the file alone
                    products[i] = merged
                    written = f"{product} updated in briefs/{path.name}."
                    break
            else:
                products.append(dict(entry, slug=new_slug))
                written = f"{product} added to \"{campaign}\" in briefs/{path.name}."
        # Written beside the target and moved into place, so a failure
        # halfway through cannot leave a half-written brief behind.
        spare = path.with_suffix(path.suffix + ".tmp")
        spare.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        spare.replace(path)
        _brief_campaign_cache.clear()
        return written
    except Exception as exc:  # noqa: BLE001
        return f"Couldn't record this campaign in briefs/: {type(exc).__name__}: {exc}"


def _record_in_yaml_brief(path, campaign: str, product: str, entry: dict, new_slug: str) -> str:
    """Add a product to a YAML brief without losing the file's comments.

    PyYAML parses a brief fine but dumping it back writes a fresh
    document: comments gone, key order gone, block scalars reflowed.
    sample_campaign.yaml is mostly comments documenting the optional
    schema, so that trade is not worth a convenience. ruamel's
    round-trip mode keeps all of it.
    """
    try:
        from ruamel.yaml import YAML
    except ImportError:
        return (
            f'"{campaign}" is described in {path.name}, and ruamel.yaml is not installed, '
            f"so it was left alone. Add {product} to it by hand, or pip install ruamel.yaml."
        )
    try:
        yaml = YAML()
        yaml.preserve_quotes = True
        with open(path, encoding="utf-8") as handle:
            data = yaml.load(handle)
        products = data.setdefault("campaign", {}).setdefault("products", [])
        for i, existing in enumerate(products):
            if str(existing.get("name") or "").strip().lower() == product.lower():
                changed = False
                for field, value in entry.items():
                    if not _same_text(existing.get(field), value):
                        existing[field] = value
                        changed = True
                if not existing.get("slug"):
                    existing["slug"] = new_slug   # only if it never had one
                    changed = True
                if not changed:
                    return ""
                written = f"{product} updated in briefs/{path.name}."
                break
        else:
            products.append(dict(entry, slug=new_slug))
            written = f'{product} added to "{campaign}" in briefs/{path.name}.'
        spare = path.with_suffix(path.suffix + ".tmp")
        with open(spare, "w", encoding="utf-8") as handle:
            yaml.dump(data, handle)
        spare.replace(path)
        _brief_campaign_cache.clear()
        return written
    except Exception as exc:  # noqa: BLE001
        return f"Couldn't update briefs/{path.name}: {type(exc).__name__}: {exc}"


def _brief_prefill(choice: dict) -> dict:
    """A card's fields from one brief entry: the campaign brief, the
    brand colours (ticked), the copy language and the AI prompt hint."""
    prefill = {
        "campaign_name": choice.get("campaign") or "",
        "product_name": choice.get("product_name") or "",
        "market": choice.get("market") or "",
        "audience": choice.get("audience") or "",
        "campaign_message": choice.get("campaign_message") or "",
    }
    if choice.get("language"):
        prefill["copy_language"] = choice["language"]
    if choice.get("prompt_hint"):
        prefill["upload_ai_prompt"] = choice["prompt_hint"]
    if choice.get("headline"):
        prefill["upload_ai_headline"] = choice["headline"]
    for i, hex_colour in enumerate((choice.get("colors") or [])[:3], start=1):
        prefill[f"brand_color_{i}"] = hex_colour.lower()
        prefill[f"brand_color_{i}_enabled"] = "1"
    return prefill


def _remembered_campaign_cards() -> list:
    """The cards a fresh form opens with: one per remembered product,
    each with that product's own fields and kept files -- isolated from
    the others. With no products remembered yet, one card from the
    top-level memory (the form as it always was)."""
    products = _product_memories(_load_preferences())
    cards = []
    seen = set()
    for choice in _brief_choices():
        key = _product_memory_key(choice.get("product_name"), choice.get("campaign"))
        if not key or key in seen:
            continue
        seen.add(key)
        prefill = _brief_prefill(choice)
        if key in products:
            remembered = _remembered_prefill(key)
            # Runs predating the required Campaign field remember it as "", which
            # would wipe the value the brief supplied. Identity only; cleared fields
            # stay clear.
            for identity in ("campaign_name", "product_name"):
                if not (remembered.get(identity) or "").strip() and prefill.get(identity):
                    remembered.pop(identity, None)
            prefill.update(remembered)
        job_id, files = _remembered_files(key) if key in products else (None, {})
        cards.append({
            "prefill": prefill,
            "prefill_files": files,
            "edit_job_id": None,
            "carry_job_id": job_id,
            "layers": card_layer_sets(prefill.get("product_name"), prefill.get("campaign_name")),
        })
    for key in products:
        if key in seen:
            continue
        seen.add(key)
        job_id, files = _remembered_files(key)
        prefill = _remembered_prefill(key)
        # Remembered from before the field was required: fill it from the
        # briefs so the card opens naming the campaign it actually uses.
        if not (prefill.get("campaign_name") or "").strip():
            from_brief = campaign_for_product(
                prefill.get("product_name"), prefill.get("market")
            )
            if from_brief:
                prefill["campaign_name"] = from_brief
        # Filling the campaign in can turn this key into one already on
        # the page. Memories written while the campaign was never saved
        # are filed under the bare product name ("HydroBoost Sports
        # Drink"), and resolving that to Winter Glow 2026 makes it the
        # same card as "Winter Glow 2026/HydroBoost Sports Drink" -- two
        # rows a person cannot tell apart. It is one product in one
        # campaign, so it is one card.
        resolved = _product_memory_key(
            prefill.get("product_name"), prefill.get("campaign_name")
        )
        if resolved and resolved != key and resolved in seen:
            # Merge, not discard. This memory is the same product in the
            # same campaign as a card already built -- it was just filed
            # under the bare product name, from before the campaign was
            # saved. Its remembered files and fields are the only copy,
            # so dropping it loses the hero and copy of that run.
            for built in cards:
                built_fields = built.get("prefill") or {}
                if _product_memory_key(
                    built_fields.get("product_name"), built_fields.get("campaign_name")
                ) != resolved:
                    continue
                for field, value in prefill.items():
                    if not str(built_fields.get(field) or "").strip():
                        built_fields[field] = value
                built["prefill"] = built_fields
                if not built.get("prefill_files"):
                    built["prefill_files"] = files
                if not built.get("carry_job_id"):
                    built["carry_job_id"] = job_id
                break
            continue
        seen.add(resolved)
        cards.append({
            "prefill": prefill,
            "prefill_files": files,
            "edit_job_id": None,
            "carry_job_id": job_id,
            "layers": card_layer_sets(prefill.get("product_name"), prefill.get("campaign_name")),
        })
    # Fall back to the top-level memory when the last run matches none of the cards
    # above, or when there is nothing else to show.
    prefs = _load_preferences()
    top_key = _product_memory_key(prefs.get("product_name"), prefs.get("campaign_name"))
    top_prefill = _remembered_prefill()
    # The top-level memory earns a card only when it names a product.
    #
    # `top_key` is its folder path, and that sanitises to "" for a name
    # made only of punctuation as well as for no name at all -- and "" is
    # never in `seen`, so both produced an extra card. They are not the
    # same thing. A run that named something gets its settings back; a
    # memory naming nothing has nothing to show, and appearing beside the
    # brief cards it just reads as a campaign that came from somewhere.
    # "Add campaign" is how you start a new one.
    top_named = bool((prefs.get("product_name") or "").strip())
    if not cards or (top_named and top_key not in seen and (top_prefill or prefs.get("last_job_id"))):
        job_id, files = _remembered_files()
        cards.append({
            "prefill": top_prefill,
            "prefill_files": files,
            "edit_job_id": None,
            "carry_job_id": job_id,
            "layers": card_layer_sets(top_prefill.get("product_name"), top_prefill.get("campaign_name")),
        })
    return cards


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
        # This page and the form are the same template, down to the
        # heading and the brief picker, and the only thing telling them
        # apart was a line inside a card -- invisible when the card is
        # folded. So an Edit showing one card reads as the form having
        # lost the other four.
        editing_job_id=job_id,
        editing_count=len(campaigns),
        session_id=session_id,
        hideable_layers=HIDEABLE_LAYER_NAMES,
        **_page_layer_sets(),
    )


PROGRESS_TOKEN_RE = re.compile(r"[0-9a-f]{8,64}")


def _progress_path(token: str) -> Path:
    return JOBS_DIR / "progress" / f"{token}.json"


def _progress_token_from(form) -> str | None:
    """The form's own run token (set by the page when Generate is
    pressed), so the overlay polls for THIS run and never reads the
    file an earlier run of the same card left behind."""
    token = (form.get("progress_token") or "").strip().lower()
    return token if PROGRESS_TOKEN_RE.fullmatch(token) else None


def _report_progress(token, percent, stage: str) -> None:
    """How far this run has got, for the busy overlay's percentage.
    Best-effort and monotonic: a milestone never moves the number
    back, and a failure to write never touches the run itself."""
    if not token:
        return
    try:
        path = _progress_path(token)
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = 0
        try:
            previous = int(json.loads(path.read_text(encoding="utf-8")).get("percent") or 0)
        except (OSError, ValueError, TypeError):
            previous = 0
        percent = max(previous, min(100, int(percent)))
        path.write_text(json.dumps({"percent": percent, "stage": stage, "at": time.time()}), encoding="utf-8")
        if percent >= 100:
            # A finished run's file is only read for the second or two
            # until the page moves on; older ones are litter.
            for old in path.parent.glob("*.json"):
                try:
                    if old != path and time.time() - old.stat().st_mtime > 3600:
                        old.unlink()
                except OSError:
                    continue
    except OSError:
        pass


@app.route("/progress/<token>")
def progress(token):
    """Polled by the busy overlay while a Generate is running:
    {"percent": 0-100, "stage": "..."} for the run with this token, or
    percent null before its first milestone."""
    token = (token or "").strip().lower()
    if not PROGRESS_TOKEN_RE.fullmatch(token):
        return jsonify({"percent": None, "stage": ""}), 404
    try:
        data = json.loads(_progress_path(token).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return jsonify({"percent": None, "stage": ""})
    return jsonify({"percent": data.get("percent"), "stage": data.get("stage") or ""})


@app.route("/generate", methods=["GET"])
def generate_reload():
    """A GET on /generate means the results page was reloaded.

    The results are rendered straight from the POST, so the address bar
    keeps saying /generate -- and reloading that (or reaching it from
    history, where browsers drop the body) arrives as a GET, which a
    POST-only route answers with a bare "405 Method Not Allowed" error
    page. Sending them back to the form with an explanation beats that.

    Nothing is lost by landing here: the batch is already on disk. Its
    zip is in downloads/, and /edit/<job_id> reopens it pre-filled.
    """
    flash(
        "That results page can't be reloaded directly -- its address is the "
        "generate step itself. Your creatives are safe: the zip is in the "
        "downloads/ folder, and \"Edit\" on the results page reopens the batch. "
        "Fill the form in again to make a new one."
    )
    return redirect(url_for("index"))


def _keep_submission(message: str):
    """A failed /generate that keeps what was typed.

    Every exit that used to flash and send the person back to a blank
    form now saves the submission as a draft -- the text fields as they
    were, every uploaded file copied aside -- and reopens the form from
    it, exactly the way Edit reopens a finished run. So an Ideogram
    outage, a missing key, a wrong file type or a size that won't parse
    costs one click, not a retyped campaign brief. Drafts live under
    outputs/web/draft_*, hold no creatives, and are never listed as runs.
    """
    # The overlay's percentage: this run is over, however it ended.
    _report_progress(_progress_token_from(request.form), 100, "Stopped")
    draft_id = f"draft_{uuid.uuid4().hex[:12]}"
    draft_dir = JOBS_DIR / draft_id
    uploads_dir = draft_dir / "uploads"
    try:
        uploads_dir.mkdir(parents=True, exist_ok=True)
        fields = {}
        for key in request.form:
            if key in ("upload_ai_reference_data", "upload_ai_reference_data_name"):
                continue  # kept as files below, not as a megabyte of base64
            values = request.form.getlist(key)
            fields[key] = values[0] if len(values) == 1 else values
        for name in EDIT_CHECKBOX_FIELD_NAMES:
            fields[name] = bool(request.form.get(name))
        files = {}
        free_slots = list(REFERENCE_SLOTS)
        names = request.form.getlist("upload_ai_reference_data_name")
        for i, data_url in enumerate(request.form.getlist("upload_ai_reference_data")):
            if not data_url.strip() or not free_slots:
                continue
            try:
                saved = _save_data_url_image(data_url, names[i] if i < len(names) else "", uploads_dir)
            except ValueError:
                continue
            files[free_slots.pop(0)] = saved.relative_to(uploads_dir).as_posix()
        for key in request.files:
            uploads = [f for f in request.files.getlist(key) if f is not None and f.filename]
            if not uploads:
                continue
            for upload in uploads:
                # Most of these were already saved once, which leaves the stream at
                # its end; a second save from there writes an empty file.
                try:
                    upload.stream.seek(0)
                except Exception:  # noqa: BLE001
                    pass
            if key == "upload_ai_reference":
                # The mood board's pictures go to their numbered slots,
                # which is where Edit looks for them.
                for slot, upload in zip(list(free_slots), uploads):
                    saved = _save_upload(upload, uploads_dir)
                    files[slot] = saved.relative_to(uploads_dir).as_posix()
                    free_slots.remove(slot)
            else:
                saved = _save_upload(uploads[0], uploads_dir)
                files[key] = saved.relative_to(uploads_dir).as_posix()
        try:
            campaign_slot = int((request.form.get("campaign_slot") or "1").strip())
        except ValueError:
            campaign_slot = 1
        _refuse_the_real_jobs_dir_under_test()
        (draft_dir / "form_state.json").write_text(
            json.dumps(
                {
                    "fields": fields,
                    "files": files,
                    "session_id": (request.form.get("session_id") or "").strip() or uuid.uuid4().hex,
                    "campaign_slot": campaign_slot,
                    "draft": True,
                },
                indent=2,
            )
        )
    except Exception:  # noqa: BLE001 -- keeping the form is a courtesy, never a second failure
        flash(message)
        return redirect(url_for("index"))
    flash(message)
    return redirect(url_for("edit", job_id=draft_id))


@app.route("/generate", methods=["POST"])
def generate():
    # Old runs go before this one writes anything: a full disk was a
    # 500 on the very first file copy.
    try:
        prune_job_folders()
    except Exception:  # noqa: BLE001
        pass
    progress_token = _progress_token_from(request.form)
    _report_progress(progress_token, 3, "Reading the form")
    # Editing a prior job posts edit_job_id; its saved form_state.json supplies file
    # fields not re-uploaded this time (see _carry_forward_upload()).
    edit_job_id = (request.form.get("edit_job_id") or "").strip() or None
    # A fresh form carries the last run's section files the way Edit does, but nothing
    # else from that run, its approvals in particular.
    carry_job_id = edit_job_id or (request.form.get("carry_files_job_id") or "").strip() or None
    prior_job_dir = None
    prior_form_state: dict = {}
    if carry_job_id and re.fullmatch(r"(draft_)?[0-9a-f]{12,32}", carry_job_id):
        candidate_dir = JOBS_DIR / carry_job_id
        state_path = candidate_dir / "form_state.json"
        if state_path.is_file():
            try:
                prior_form_state = json.loads(state_path.read_text())
            except (OSError, ValueError):
                prior_form_state = {}
            else:
                prior_job_dir = candidate_dir
    # Approved sizes are carried over untouched rather than regenerated, and in normal
    # mode their backdrop is pinned for the other sizes.
    approved_prior_sizes = set()
    if edit_job_id and prior_job_dir is not None and not prior_job_dir.name.startswith("draft_"):
        for label, entry in load_approvals(prior_job_dir.name).items():
            if not (isinstance(entry, dict) and entry.get("approved")):
                continue
            match = re.fullmatch(r"(\d+)x(\d+)", label)
            if match and any(prior_job_dir.glob(f"*_{label}.png")):
                approved_prior_sizes.add((int(match.group(1)), int(match.group(2))))

    hero_file = request.files.get("hero_image")
    hero_fresh = hero_file is not None and bool(hero_file.filename)
    if hero_fresh and not _allowed(hero_file.filename, SUPPORTED_EXTENSIONS):
        return _keep_submission(
            f"'{hero_file.filename}' isn't a supported file type. Accepted: "
            + ", ".join(SUPPORTED_EXTENSIONS)
        )

    # Opt-in AI hero for sizes with no uploaded hero and no matching PSD, using the
    # same src/providers/ abstraction as the CLI pipeline. Upload Creative's own
    # generator instead supplies artwork to the saved templates.
    upload_ai_enabled = bool(request.form.get("upload_ai_enabled"))
    # Reuse the last run's artwork instead of generating again: every call returns a
    # different picture, so tweaking a font would lose the backdrop.
    upload_ai_keep = bool(request.form.get("upload_ai_keep"))
    approval_pinned_backdrop = False
    if (
        approved_prior_sizes
        and not upload_ai_keep
        and (prior_form_state.get("files") or {}).get("upload_ai_generated")
        and not request.form.get("upload_ai_full_ad")
    ):
        # A size was approved last run, so the others update against that same
        # backdrop rather than a fresh generation that would break up the set.
        upload_ai_keep = True
        approval_pinned_backdrop = True
    # Off by default: lettering competes with the template's own header and CTA.
    # Ticked, every no-text defence downstream has to stand down.
    upload_ai_allow_text = bool(request.form.get("upload_ai_allow_text"))
    # Full ad: the brief's copy goes into the prompt and the result is the creative,
    # with no template layers drawn over it. Implies allow_text. Ticking the custom
    # route alone means a templated batch, file or not.
    upload_custom_hero_enabled = bool(request.form.get("upload_custom_hero_enabled"))
    # Off by default: folding one campaign's words into a reused template means every
    # later campaign starts from them. Ticked, a retype sticks for good.
    update_saved_templates = bool(request.form.get("update_saved_templates"))
    upload_ai_full_ad = bool(request.form.get("upload_ai_full_ad"))
    # Choosing your own hero settles it here, whatever the form posted.
    #
    # The page pairs these boxes, but only some of them and only on click,
    # and the settings are remembered per product -- so a campaign where
    # someone once used Ideogram kept generating for the one they moved on
    # to, over the artwork they had just supplied, at nine paid images a
    # run. The browser is not the place to enforce "my file, not yours":
    # every path through the form has to come out the same, so the answer
    # is decided here instead.
    supplied_own_hero_instead = upload_custom_hero_enabled and (
        upload_ai_enabled or upload_ai_full_ad
    )
    if supplied_own_hero_instead:
        upload_ai_enabled = False
        upload_ai_full_ad = False
    # Separate from layer_header_text, which greys out when its layer is off in
    # Photoshop; here the words go to the prompt, not a layer. Falls back to that
    # field, then to the campaign message.
    upload_ai_headline = (request.form.get("upload_ai_headline") or "").strip() or None
    if upload_ai_full_ad:
        upload_ai_allow_text = True
    upload_ai_prompt = (request.form.get("upload_ai_prompt") or "").strip() or None
    # Exclusions typed into the prompt move to the negative channel, where they work.
    # A typed "no text" beats allow-text or full-ad: the run becomes a text-free
    # backdrop with the OCR check and retry back on.
    upload_ai_prompt, upload_ai_prompt_negations, upload_ai_prompt_no_text = (
        split_prompt_negations(upload_ai_prompt)
    )
    upload_ai_prompt_notes = []
    if upload_ai_prompt_negations:
        upload_ai_prompt_notes.append(
            "The exclusion you typed into the Image prompt ("
            + "; ".join(f'"{n}"' for n in upload_ai_prompt_negations)
            + ") was taken out of it and sent as a negative prompt instead -- "
            "in the prompt itself, those words tell the model what to paint."
        )
        if upload_ai_prompt_no_text and (upload_ai_allow_text or upload_ai_full_ad):
            upload_ai_prompt_notes.append(
                "Because it asks for no text, the "
                + ("whole-ad" if upload_ai_full_ad else "allow-text")
                + " box was ignored for this run: the picture was generated as a "
                "text-free backdrop and checked for lettering."
            )
            upload_ai_allow_text = False
            upload_ai_full_ad = False
        if upload_ai_prompt is None:
            upload_ai_prompt_notes.append(
                "Nothing else was in the prompt, so the scene came from the campaign brief."
            )
    upload_ai_provider = request.form.get("upload_ai_provider", "pollinations")
    # Only set where a generation actually happens; the reuse path and provider
    # failures both skip it.
    upload_ai_prompt_used = None
    upload_ai_speed = (request.form.get("upload_ai_speed") or DEFAULT_IDEOGRAM_SPEED).upper()
    if upload_ai_speed not in dict(IDEOGRAM_SPEED_CHOICES):
        upload_ai_speed = DEFAULT_IDEOGRAM_SPEED
    # Everything this run generates is billed through one meter.
    spend = {"images": 0, "cost": 0.0}

    def _provider(name):
        try:
            raw = get_provider(name, rendering_speed=upload_ai_speed)
        except TypeError:
            # A provider factory with the old one-argument signature
            # (the test stubs, mostly): no speed knob to turn.
            raw = get_provider(name)
        return _MeteredProvider(raw, spend)
    # ALL_PROVIDER_NAMES, not PROVIDER_NAMES: the offline placeholder is kept out of
    # the dropdowns but accepted by name, which is how tests render.
    if upload_ai_provider not in ALL_PROVIDER_NAMES:
        upload_ai_provider = "pollinations"
    # Defaults ON (see BACKGROUND_PROMPT_GUIDANCE). Forms predating the checkbox
    # submit nothing, so the default is carried by a marker field.
    upload_ai_background_style = bool(
        request.form.get("upload_ai_background_style")
        or not request.form.get("upload_ai_background_style_seen")
    )

    ai_hero_enabled = bool(request.form.get("ai_hero_enabled"))
    ai_hero_prompt = (request.form.get("ai_hero_prompt") or "").strip() or None
    ai_hero_provider = request.form.get("ai_hero_provider", "pollinations")
    if ai_hero_provider not in ALL_PROVIDER_NAMES:
        ai_hero_provider = "pollinations"

    # The two provider selects are one choice; disagreement made the results page
    # report a failure from a provider the form didn't show. The non-default value
    # wins, since "pollinations" is what an unset field yields.
    if upload_ai_provider != ai_hero_provider:
        chosen = next(
            (
                name
                for name in (upload_ai_provider, ai_hero_provider)
                if name != DEFAULT_PROVIDER_NAME
            ),
            DEFAULT_PROVIDER_NAME,
        )
        upload_ai_provider = ai_hero_provider = chosen

    # Campaign brief is context only: nothing composites it into the creatives. It
    # travels to the results page and through save/carry-forward.
    product_name = (request.form.get("product_name") or "").strip() or None
    # This product's own templates, default_templates/<product>/, seeded from the
    # shared set on first run. Every scan, drop and style save uses it.
    from flask import g as _g
    campaign_name = (request.form.get("campaign_name") or "").strip() or None
    _g.templates_dir = product_templates_dir(product_name, create=True, campaign_name=campaign_name)
    product_templates_created = (
        _g.templates_dir != DEFAULT_TEMPLATES_DIR
        and any(_g.templates_dir.glob("*.psd"))
        and (time.time() - _g.templates_dir.stat().st_mtime) < 5
    )
    # Names downloaded PNG/PSD/zip files after the product. Falls back to the generic
    # "creative" prefix when there is no usable product name.
    product_name_slug = _slugify_for_filename(product_name) if product_name else ""
    # Which campaign card posted this. Parsed here because it names files: two cards
    # produce the same sizes, and without it their downloads collide.
    try:
        campaign_slot = int((request.form.get("campaign_slot") or "1").strip())
    except ValueError:
        campaign_slot = 1
    # Named for the campaign when the card has one, otherwise the card's slot number,
    # which is still unique per card.
    campaign_label = _slugify_for_filename(campaign_name) if campaign_name else ""
    campaign_label = campaign_label or f"campaign{campaign_slot}"
    file_name_prefix = f"{product_name_slug}_{campaign_label}" if product_name_slug else f"creative_{campaign_label}"
    market = (request.form.get("market") or "").strip() or None
    audience = (request.form.get("audience") or "").strip() or None
    campaign_message = (request.form.get("campaign_message") or "").strip() or None

    # All four brief fields are required: they name files and travel with the batch to
    # the results page and any later edit.
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
        return _keep_submission(
            "Campaign brief is required -- please fill in: "
            + ", ".join(missing_brief_fields)
            + "."
        )

    # Blocks generation outright, like an incomplete brief. Header, description, CTA
    # and layer description are read raw here, before they're parsed below.
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
        return _keep_submission(
            "That contains language we can't allow through -- please edit: "
            + ", ".join(flagged_fields)
            + "."
        )

    # Each of the three is independently opt-in: <input type="color"> always carries a
    # value, so there is no blank to detect.
    brand_colors = []
    for i in (1, 2, 3):
        if request.form.get(f"brand_color_{i}_enabled"):
            brand_colors.append(_parse_hex_color(request.form.get(f"brand_color_{i}"), default=(0, 0, 0)))
    _remember_form_fields(request.form)

    headline = (request.form.get("header") or "").strip() or None
    message = (request.form.get("description") or "").strip() or None
    fit_mode = request.form.get("fit_mode", "crop")
    if fit_mode not in ("crop", "contain"):
        fit_mode = "crop"
    # Fill crops the hero to the layer's shape (right for a photo); fit pads the edges
    # with the picture's own edge colour so nothing is cut off.
    upload_hero_fit = request.form.get("upload_hero_fit", "crop")
    if upload_hero_fit not in ("crop", "contain"):
        upload_hero_fit = "crop"

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
        return _keep_submission(f"Couldn't parse the sizes you entered: {exc}")

    job_id = uuid.uuid4().hex
    _refuse_the_real_jobs_dir_under_test()
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir = job_dir / "uploads"
    uploads_dir.mkdir(exist_ok=True)

    if hero_fresh:
        hero_path = _save_upload(hero_file, uploads_dir)
    else:
        prior_hero_rel = (prior_form_state.get("files") or {}).get("hero_image")
        if ai_hero_enabled and prior_hero_rel == AI_GENERATED_HERO_FILENAME:
            # The carried hero is itself a previous AI generation and the AI box is
            # still ticked. Carrying it would lock that image in and make the prompt
            # and provider fields silently do nothing, so leave hero_path unset.
            hero_path = None
        else:
            hero_path = _carry_forward_upload("hero_image", uploads_dir, prior_job_dir, prior_form_state)
    hero_provided = hero_path is not None

    # Up to MAX_PSD_TEMPLATES rows of (psd_size_N, psd_file_N). A row with a file
    # backgrounds that exact size and forces it into the batch.
    psd_templates: dict = {}
    psd_template_paths: dict = {}
    psd_file_paths: dict = {}  # {row index: Path} -- for form_state.json, see below
    # Use the uploaded PSD untouched: no hero, no typed copy, no hide boxes. For a
    # finished one-size creative, repainting it read as the upload not taking.
    psd_as_is_sizes: set = set()
    # (label, Path) for PSDs uploaded this request only; profanity-scanned below.
    fresh_psd_uploads = []
    # A file uploaded now carries onto same-shape sizes whose own file is only a
    # carry-over from an earlier run.
    fresh_psd_sizes: set = set()
    psd_row_by_size: dict = {}
    # Rows whose file became the saved template for its size, recorded so a later run
    # knows the saved template supersedes the file in the row.
    prior_promoted_rows = {int(r) for r in (prior_form_state.get("promoted_psd_rows") or []) if str(r).isdigit()}
    promoted_psd_rows: set = set()
    psd_size_snaps: list = []  # (row, as typed, used) -- noted once background_notes exists
    fresh_psd_drops: list = []  # filenames dropped THIS run, not carried
    for i in range(1, MAX_PSD_TEMPLATES + 1):
        psd_size_raw = (request.form.get(f"psd_size_{i}") or "").strip()
        psd_file = request.files.get(f"psd_file_{i}")
        psd_file_fresh = psd_file is not None and bool(psd_file.filename)
        if psd_file_fresh:
            fresh_psd_drops.append(psd_file.filename)
        # The row's "x" sets this hidden field, the only way to cancel a carried
        # template and go back to the hero image for that size.
        psd_cleared = bool(request.form.get(f"psd_size_{i}_clear"))
        if psd_file_fresh:
            if not _allowed(psd_file.filename, ALLOWED_PSD_TEMPLATE_EXTENSIONS):
                return _keep_submission(
                    f"PSD template row {i}: '{psd_file.filename}' isn't a supported file type. Accepted: "
                    + ", ".join(ALLOWED_PSD_TEMPLATE_EXTENSIONS)
                )
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
            return _keep_submission(f"PSD template row {i}: choose a target size (e.g. {CONTENT_PSD_LABEL}) for the uploaded PSD file.")
        if psd_size_raw and not psd_file_provided:
            return _keep_submission(f"PSD template row {i}: you entered a size ({psd_size_raw}) but didn't attach a .psd file.")
        try:
            psd_width, psd_height = parse_size(psd_size_raw)
        except ValueError as exc:
            return _keep_submission(f"PSD template row {i}: {exc}")
        # Snap a near-miss to a known size (728x480 -> 720x480). Without it the row
        # opened a size of its own and the real slot kept rendering the template.
        snapped = _snap_row_size(
            (psd_width, psd_height), set(_default_template_paths()) | set(SIZE_NAMES) | set(DEFAULT_SIZES)
        )
        if snapped != (psd_width, psd_height):
            psd_size_snaps.append((i, (psd_width, psd_height), snapped))
            psd_width, psd_height = snapped
        # A remembered row whose file was promoted renders from the saved template,
        # not the row's older copy; the row is only the chip showing what was dropped.
        # A never-promoted row renders its own file as uploaded.
        if (
            not psd_file_fresh
            and i in prior_promoted_rows
            and (psd_width, psd_height) in _default_template_paths()
        ):
            promoted_psd_rows.add(i)
            if request.form.get("psd_as_is") and not upload_ai_enabled:
                psd_as_is_sizes.add((psd_width, psd_height))
            continue
        try:
            psd_templates[(psd_width, psd_height)] = open_as_rgb(psd_path)
        except ValueError as exc:
            return _keep_submission(f"PSD template row {i}: {exc}")
        psd_template_paths[(psd_width, psd_height)] = psd_path
        psd_row_by_size[(psd_width, psd_height)] = i
        if psd_file_fresh:
            fresh_psd_sizes.add((psd_width, psd_height))
        if request.form.get("psd_as_is") and not upload_ai_enabled:
            psd_as_is_sizes.add((psd_width, psd_height))

    # Quick campaign: upload one flagship 728x480 PSD and the Output/Custom size
    # choices are ignored; the batch is that size plus default_templates/.
    content_psd_file = request.files.get("content_psd")
    content_psd_fresh = content_psd_file is not None and bool(content_psd_file.filename)
    if content_psd_fresh:
        fresh_psd_drops.append(content_psd_file.filename)
        if not _allowed(content_psd_file.filename, ALLOWED_PSD_TEMPLATE_EXTENSIONS):
            return _keep_submission(
                f"{CONTENT_PSD_LABEL} content PSD: '{content_psd_file.filename}' isn't a supported file type. "
                "Accepted: " + ", ".join(ALLOWED_PSD_TEMPLATE_EXTENSIONS)
            )
        content_psd_path = _save_upload(content_psd_file, uploads_dir)
        fresh_psd_uploads.append((f"{CONTENT_PSD_LABEL} content PSD", content_psd_path))
    else:
        content_psd_path = _carry_forward_upload("content_psd", uploads_dir, prior_job_dir, prior_form_state)
    content_psd_provided = content_psd_path is not None
    content_psd_image = None
    if content_psd_provided:
        try:
            content_psd_image = open_as_rgb(content_psd_path)
        except ValueError as exc:
            return _keep_submission(f"{CONTENT_PSD_LABEL} content PSD: {exc}")
        # The upload renders as its own size and pulls in the saved templates; the
        # size selections and general hero image stay ignored in this mode.
        sizes = []

    # The ordinary path: one picture becomes the backdrop of the saved 728x480
    # template and carries onto every saved size, as the AI generator's does.
    upload_hero_file = request.files.get("upload_hero_image")
    upload_hero_fresh = upload_hero_file is not None and bool(upload_hero_file.filename)
    background_notes_pending_hero = None
    if upload_hero_fresh:
        if not _allowed(upload_hero_file.filename, PICTURE_UPLOAD_EXTENSIONS):
            return _keep_submission(
                f"Campaign hero image: '{upload_hero_file.filename}' isn't a supported file type. "
                "Accepted: " + ", ".join(PICTURE_UPLOAD_EXTENSIONS)
            )
        try:
            upload_hero_path = _flatten_psd_upload(_save_upload(upload_hero_file, uploads_dir))
        except ValueError as exc:
            return _keep_submission(f"Campaign hero image: {exc}")
    elif request.form.get("upload_hero_from_template"):
        # The template's own backdrop as the hero: nothing to upload.
        try:
            upload_hero_path, hero_note = template_background_as_hero(uploads_dir)
        except ValueError as exc:
            return _keep_submission(f"Campaign hero image from the template: {exc}")
        background_notes_pending_hero = hero_note
    elif request.form.get("upload_hero_image_clear"):
        upload_hero_path = None
    else:
        upload_hero_path = _carry_forward_upload(
            "upload_hero_image", uploads_dir, prior_job_dir, prior_form_state
        )
    upload_hero_image = None
    if upload_hero_path is not None:
        try:
            upload_hero_image = Image.open(upload_hero_path).convert("RGBA")
        except Exception as exc:  # noqa: BLE001
            return _keep_submission(f"Couldn't read the campaign hero image: {exc}")
        # Like a content PSD upload, this drives a templated batch: the
        # sizes come from default_templates/, not from the size pickers.
        sizes = []

    # Up to REFERENCE_LIMIT reference pictures, sent to Ideogram as style references
    # and described in words for every provider. Numbered slots so Edit carries each
    # forward and (x) drops one.
    upload_ai_reference_paths = []
    fresh_files = [
        f for f in request.files.getlist("upload_ai_reference") if f is not None and f.filename
    ]
    for upload in fresh_files:
        if not _allowed(upload.filename, REFERENCE_EXTENSIONS):
            return _keep_submission(
                f"Reference image: '{upload.filename}' isn't a supported file type. "
                "Accepted: pictures (" + ", ".join(ALLOWED_LAYER_IMAGE_EXTENSIONS + (".heic", ".tif", ".gif", ".psd"))
                + ") and videos (" + ", ".join(VIDEO_EXTENSIONS) + " -- the middle frame is used)."
            )
        try:
            upload_ai_reference_paths.append(_normalise_reference(_save_upload(upload, uploads_dir)))
        except ValueError as exc:
            return _keep_submission(str(exc))
    dropped_names = request.form.getlist("upload_ai_reference_data_name")
    for i, data_url in enumerate(request.form.getlist("upload_ai_reference_data")):
        if not data_url.strip():
            continue
        try:
            upload_ai_reference_paths.append(
                _save_data_url_image(data_url, dropped_names[i] if i < len(dropped_names) else "", uploads_dir)
            )
        except ValueError as exc:
            return _keep_submission(str(exc))
    for url in (u.strip() for u in request.form.getlist("upload_ai_reference_url")):
        if not url:
            continue
        # Dragged or pasted addresses are fetched once and kept as files, so Edit
        # carries them forward and the page never refetches.
        try:
            upload_ai_reference_paths.append(_fetch_web_image(url, uploads_dir))
        except ValueError as exc:
            return _keep_submission(str(exc))
    for slot in REFERENCE_SLOTS:
        if request.form.get(f"{slot}_clear"):
            continue
        carried = _carry_forward_upload(slot, uploads_dir, prior_job_dir, prior_form_state)
        if carried is not None and carried not in upload_ai_reference_paths:
            upload_ai_reference_paths.append(carried)
    reference_overflow_warning = None
    if len(upload_ai_reference_paths) > REFERENCE_LIMIT:
        reference_overflow_warning = (
            f"A mood board can hold {REFERENCE_LIMIT} pictures (Ideogram's limit for a style); "
            f"{len(upload_ai_reference_paths)} were given -- the first {REFERENCE_LIMIT} were used."
        )
        upload_ai_reference_paths = upload_ai_reference_paths[:REFERENCE_LIMIT]
    upload_ai_reference_path = upload_ai_reference_paths[0] if upload_ai_reference_paths else None
    upload_ai_reference_images = []
    upload_ai_reference_bytes_list = []
    upload_ai_reference_notes = []
    # A style reference is copied whole, layout and typography included: a board of
    # finished ads came back as a poster with invented brand names, and no amount of
    # "no text" stopped it. By default only the look is described.
    upload_ai_send_references = bool(request.form.get("upload_ai_send_references"))
    for path in upload_ai_reference_paths:
        try:
            reference = Image.open(path).convert("RGB")
            reference_bytes = path.read_bytes() if upload_ai_send_references else None
            if not upload_ai_allow_text:
                # See _textless_reference(): a reference with words on it
                # is a request for words, whatever the negative prompt says.
                textless, note = _textless_reference(reference, path.name)
                if note:
                    upload_ai_reference_notes.append(note)
                if textless is None:
                    continue
                if textless is not reference:
                    # A reference with words comes back from Ideogram as an ad with
                    # the words painted in. Its cleaned copy still feeds the mood
                    # board description.
                    reference = textless
                    buffer = io.BytesIO()
                    reference.save(buffer, format="PNG")
                    (path.parent / f"{path.stem}_textless.png").write_bytes(buffer.getvalue())
                    if upload_ai_send_references:
                        upload_ai_reference_notes.append(
                            f"{path.name} was described in the prompt only, not sent as a style reference."
                        )
                    reference_bytes = None
            upload_ai_reference_images.append(reference)
            if reference_bytes is not None:
                upload_ai_reference_bytes_list.append(reference_bytes)
        except Exception as exc:  # noqa: BLE001
            return _keep_submission(f"Couldn't read the reference image {path.name}: {exc}")
    upload_ai_reference_image = _mood_board(upload_ai_reference_images) if upload_ai_reference_images else None
    # One picture goes as bytes, a board as a list -- the provider takes either.
    upload_ai_reference_bytes = (
        upload_ai_reference_bytes_list[0] if len(upload_ai_reference_bytes_list) == 1
        else upload_ai_reference_bytes_list or None
    )

    # The campaign backdrop, generated at the content PSD's size and fed in as a
    # background-layer override, which carries it onto every template. Runs with or
    # without a PSD; with one it replaces only that file's background.
    upload_ai_image = None
    upload_ai_path = None
    # What the provider returned before the shortfall was made up, so the render loop
    # can warn only the sizes the enlargement actually softens.
    upload_ai_source_size = None
    # Both are raised before background_notes/background_warnings exist,
    # so they wait here and are flushed onto those lists below.
    background_notes_pending = None
    background_warnings_pending = []
    kept_ai_path = None
    # Not gated on upload_ai_enabled: ticking "Keep this image" unticks the generate
    # box, so gating here would drop the artwork keep was preserving.
    if upload_ai_keep:
        # Carried under its own key so it can only return when asked for. A silent
        # carry-forward outranks the fresh generation and overwrites its file, making
        # a new prompt return a byte-identical result.
        kept_ai_path = _carry_forward_upload(
            "upload_ai_generated", uploads_dir, prior_job_dir, prior_form_state
        )
    if kept_ai_path is not None:
        try:
            upload_ai_image = Image.open(kept_ai_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            # Nothing worth failing a run over -- fall through and
            # generate a new one, saying why.
            app.logger.warning("Couldn't reuse the kept AI image: %s", exc)
            upload_ai_image = None
        else:
            upload_ai_path = uploads_dir / AI_GENERATED_CAMPAIGN_FILENAME
            upload_ai_image.save(upload_ai_path)
            background_notes_pending = (
                "Campaign artwork: reused the image from the previous run -- an approved size "
                "pins its backdrop, so the other sizes were updated against the same picture and "
                "no new one was generated. Unapprove every size to generate a fresh one."
                if approval_pinned_backdrop else
                "Campaign artwork: reused the image from the previous run -- \"Keep this image\" "
                "is ticked, so no new one was generated. Untick it to generate a fresh one."
            )
    elif upload_ai_keep:
        # Ticked with nothing to carry: a first run, or the previous job folder is
        # gone.
        background_warnings_pending.append(
            "\"Keep this image\" is ticked but there's no previous image to keep -- nothing was "
            "generated and the templates kept their own backdrops. Untick it and tick "
            "\"Generate the hero image with AI\" to make one."
        )
    # Full ad mode replaces every template with its own generation, so a campaign
    # backdrop here is an extra call and up to 40s thrown away.
    if upload_ai_enabled and upload_ai_image is None and not upload_ai_full_ad:
        # A backdrop, not a product shot. Asking for a "branded" one gets an invented
        # logo, and nothing downstream undoes it: Ideogram's prompt beats its negative
        # prompt. The product name stays for mood; "unbranded" defuses it.
        if upload_ai_allow_text:
            # Quote the product name so the model sets those characters; named only as
            # a subject it letters whatever it likes. Ask for a scene first and type
            # second, or Ideogram returns a wordmark on a flat field every run.
            upload_ai_prompt_text = upload_ai_prompt or (
                f"advertising photograph: "
                f"{_backdrop_scene(product_name, campaign_message, audience)}, "
                + (
                    f'with the words "{product_name}" set as a headline over the photograph, '
                    if product_name else ""
                )
                + "bold headline typography, clean layout"
            )
        else:
            upload_ai_prompt_text = upload_ai_prompt or (
                f"{_backdrop_scene(product_name, campaign_message, audience, textless=True)}, "
                "unbranded, open uncluttered space, soft lighting"
            )
        # Ticked brand colours go into every prompt, not just the full-ad one. Left
        # out, the results-page colour check flagged every size as missing them.
        palette = _brand_palette_phrase(
            brand_colors, _prompt_names_its_own_colour(upload_ai_prompt)
        )
        if palette:
            upload_ai_prompt_text = f"{upload_ai_prompt_text}, {palette}"
        if upload_ai_reference_image is not None:
            upload_ai_prompt_text = (
                f"{upload_ai_prompt_text}, {reference_look_phrase(upload_ai_reference_image)}"
            )
        if upload_ai_background_style:
            if upload_ai_prompt and not upload_ai_allow_text:
                # The brief's prompt is followed; only the brand name is stripped,
                # since a named brand gets lettered into the picture.
                trimmed, left_out = _keep_the_product_out_of_a_backdrop_prompt(
                    upload_ai_prompt_text, product_name
                )
                if left_out:
                    # A prompt that was ONLY the brand name has nothing
                    # left once it goes: the automatic scene stands in.
                    if len(trimmed.split()) < 3:
                        trimmed = (
                            f"{_backdrop_scene(product_name, campaign_message, audience, textless=True)}, "
                            "unbranded, open uncluttered space, soft lighting"
                        )
                    upload_ai_prompt_text = trimmed
                    upload_ai_prompt_notes.append(
                        "Backdrop mode: "
                        + ", ".join(f'"{w}"' for w in dict.fromkeys(left_out))
                        + " left out of your prompt -- a model asked for the brand by name writes "
                        "the name on the picture, and lettering is what a backdrop can't have. "
                        "The rest of your prompt was sent as written."
                    )
            upload_ai_prompt_text = (
                f"{upload_ai_prompt_text}, "
                f"{BACKGROUND_PROMPT_GUIDANCE_WITH_TEXT if upload_ai_allow_text else BACKGROUND_PROMPT_GUIDANCE}"
            )
        try:
            upload_ai_width, upload_ai_height = _generation_size(
                _default_template_sizes(), CONTENT_PSD_SIZE
            )
            _report_progress(progress_token, 8, f"Generating the artwork with {upload_ai_provider}")
            (
                upload_ai_image,
                upload_ai_prompt_used,
                upload_ai_attempts,
                upload_ai_text,
            ) = _generate_text_free(
                _provider(upload_ai_provider),
                upload_ai_prompt_text,
                upload_ai_width,
                upload_ai_height,
                allow_text=upload_ai_allow_text,
                negative_extra=", ".join(
                    clause for clause in (
                        PALETTE_NEGATIVE_CLAUSE if brand_colors else None,
                        LOGO_NEGATIVE_CLAUSE if upload_ai_allow_text else None,
                        # Backdrop exclusions used to ride in the positive prompt,
                        # feeding the model the very nouns it was meant to avoid.
                        BACKDROP_NEGATIVE_CLAUSE
                        if upload_ai_background_style and not upload_ai_allow_text else None,
                        ", ".join(upload_ai_prompt_negations) or None,
                    ) if clause
                ) or None,
                style_reference=upload_ai_reference_bytes,
            )
            _report_progress(progress_token, 30, "Artwork ready")
            background_notes_pending = (
                f"Campaign artwork generated with AI ({upload_ai_provider}) at "
                f"{upload_ai_image.width}x{upload_ai_image.height} -- prompt: "
                f"\"{upload_ai_prompt_used}\"."
            )
            if upload_ai_prompt_notes:
                background_notes_pending += " " + " ".join(upload_ai_prompt_notes)
            if upload_ai_reference_notes:
                background_notes_pending += " " + " ".join(upload_ai_reference_notes)
            if upload_ai_reference_path is not None:
                sent_as_file = (
                    getattr(_provider(upload_ai_provider), "supports_style_reference", False)
                    and bool(upload_ai_reference_bytes)
                )
                names = ", ".join(p.name for p in upload_ai_reference_paths)
                background_notes_pending += (
                    f" Styled after the reference image{'s' if len(upload_ai_reference_paths) > 1 else ''} {names}"
                    + (
                        f" (sent to Ideogram as {'style references' if len(upload_ai_reference_paths) > 1 else 'a style reference'}, and described in the prompt)."
                        if sent_as_file
                        else " (their palette, lighting and mood described in the prompt; the pictures themselves were not sent -- "
                        "tick \"Send the mood board to Ideogram as style references\" to send them, which copies their layout and lettering too)."
                    )
                )
            if upload_ai_allow_text:
                background_notes_pending += (
                    " Text was allowed in this image, so no no-text instruction was sent, "
                    "nothing was checked for lettering, and nothing was painted out."
                )
            if upload_ai_attempts > 1:
                background_notes_pending += (
                    f" Regenerated {upload_ai_attempts - 1} more time"
                    f"{'s' if upload_ai_attempts > 2 else ''} because the first result had "
                    "text in it."
                )
            if upload_ai_text.found_text:
                upload_ai_image, cleaned_note, cleaned_warning = _clean_text_out(
                    upload_ai_image, upload_ai_text, "backdrop", upload_ai_attempts
                )
                if cleaned_note:
                    background_notes_pending += " " + cleaned_note
                if cleaned_warning:
                    background_warnings_pending.append(cleaned_warning)
            elif not upload_ai_text.available and not upload_ai_allow_text:
                background_warnings_pending.append(
                    "Couldn't check the generated backdrop for text -- no text detector is "
                    f"available ({_text_detector_problem()}). The image may have lettering "
                    "baked into it; give it a look before shipping."
                )
            if not upload_ai_allow_text and upload_ai_text.available and "Tesseract only" in detector_description():
                background_warnings_pending.append(
                    "The text check is running on Tesseract alone, which misses stylised headlines and "
                    f"lettering in pictures -- the scene-text detector isn't running ({_text_detector_problem()}). "
                    "Until it is, treat the backdrop's text check as a rough one and look the image over."
                )
            if (
                upload_ai_image.width < upload_ai_width
                or upload_ai_image.height < upload_ai_height
            ):
                # Make the shortfall up once here instead of every size enlarging
                # separately. Measure before upscaling, or the size matches the
                # request exactly and the warning reads as being about nothing.
                returned_width, returned_height = upload_ai_image.width, upload_ai_image.height
                upload_ai_source_size = (returned_width, returned_height)
                upload_ai_image = upscale_to_cover(
                    upload_ai_image, (upload_ai_width, upload_ai_height)
                )
                # Recorded as a note so the report says what the provider gave; the
                # warning is raised per size below, since run-wide it read as the
                # whole batch soft.
                background_notes_pending = (background_notes_pending or "") + (
                    f" The '{upload_ai_provider}' provider capped this at "
                    f"{returned_width}x{returned_height} against a requested "
                    f"{upload_ai_width}x{upload_ai_height}."
                )
        except ImageProviderError as exc:
            # A flaky free API degrades to the offline placeholder instead of failing
            # the run, and says so.
            app.logger.warning(
                "AI provider %r failed for campaign artwork: %s", upload_ai_provider, exc
            )
            upload_ai_image = MockImageProvider().generate(upload_ai_prompt_text)
            # A warning, not a note: filed under the collapsed "Details", the fact
            # that the artwork is a labelled placeholder stayed folded away.
            background_warnings_pending.append(
                f"Campaign artwork: the '{upload_ai_provider}' AI provider failed ({exc}) -- used the "
                f"offline placeholder generator instead. Prompt: \"{upload_ai_prompt_text}\"."
            )
        upload_ai_path = uploads_dir / AI_GENERATED_CAMPAIGN_FILENAME
        upload_ai_image.save(upload_ai_path)
    elif upload_ai_image is None:
        # Generator off entirely. Not the reuse path, which has its own note to report
        # and must not be cleared here.
        background_notes_pending = None

    # Same hard gate as the typed fields, over text layers in PSDs uploaded this
    # request only: carried-forward and saved templates were checked already.
    for psd_label, psd_path_to_scan in fresh_psd_uploads:
        for layer_name, layer_text in get_psd_text_layers(psd_path_to_scan).items():
            if check_profanity(layer_text):
                return _keep_submission(
                    f"{psd_label}: the '{layer_name}' text layer contains language we can't allow "
                    "through -- please edit it in the PSD and re-upload."
                )

    background_notes = []  # shown on the results page -- flash() only survives a redirect, and this path doesn't redirect
    if product_templates_created:
        background_notes.append(
            f"{product_name}: made its own templates folder, default_templates/{_g.templates_dir.relative_to(DEFAULT_TEMPLATES_DIR).as_posix()}/, "
            "from the backup zip (the shared PSDs when there is no zip). This product's runs use and save "
            "into that folder from now on; its Reset puts the zip's copies back."
        )
    elif _g.templates_dir != DEFAULT_TEMPLATES_DIR:
        background_notes.append(
            f"{product_name}: templates from default_templates/{_g.templates_dir.relative_to(DEFAULT_TEMPLATES_DIR).as_posix()}/ (this product's own)."
        )
    background_warnings = []  # same idea, but rendered in red -- for things worth flagging (e.g. a missing brand color), not just FYI context
    if supplied_own_hero_instead:
        # Said out loud: an image generator switching itself off silently
        # is the same class of surprise as one switching itself on.
        background_notes.append(
            "Custom hero image was ticked, so the AI generator was left off for this run "
            "(both it and \"generate the whole ad\" are remembered per product, and one of "
            "them was still on from an earlier campaign). Untick Custom hero image to "
            "generate a backdrop instead."
        )
    missing_fonts_reported: set = set()  # each uninstalled template font is reported once, not once per size
    if background_notes_pending_hero:
        background_notes.append(background_notes_pending_hero)
    if reference_overflow_warning:
        background_warnings.append(reference_overflow_warning)
    if background_notes_pending:
        background_notes.append(background_notes_pending)
    background_warnings.extend(background_warnings_pending)

    # default_templates/ defines the batch only for a templated campaign; scanning it
    # otherwise handed every plain hero-image campaign the same saved set. Full ad
    # mode and any layer-image override count as templated as well.
    layer_image_supplied = any(
        (request.files.get(field) is not None and bool(request.files[field].filename))
        or (prior_form_state.get("files") or {}).get(field)
        for field in (
            "layer_logo_image",
            "layer_product_image",
        )
    )

    if (
        content_psd_provided
        or upload_ai_image is not None
        or upload_hero_image is not None
        or upload_ai_full_ad
        or upload_custom_hero_enabled
        or layer_image_supplied
        # Once uploads are promoted into them the saved templates are the live design,
        # so a run with this on always starts from them.
        or bool(request.form.get("psd_make_saved"))
    ):
        default_templates, default_template_paths = _default_size_templates()
    else:
        default_templates, default_template_paths = {}, {}
    if content_psd_provided and not default_templates:
        background_notes.append(
            "default_templates/ doesn't have any saved templates yet -- only the "
            f"uploaded {CONTENT_PSD_LABEL} content PSD's own size was exported."
        )
    size_templates = dict(default_templates)
    size_template_paths = dict(default_template_paths)
    content_psd_size = None
    if content_psd_provided:
        # Snapped rather than added: an upload a few pixels off a saved size is a new
        # version of that creative, so the preview count matches the template count.
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
    # "Exactly as uploaded" covers the saved templates too: once promoted they are the
    # current designs, and a hero or typed copy over them is the same repaint.
    if request.form.get("psd_as_is") and not upload_ai_enabled:
        psd_as_is_sizes.update(default_templates.keys())
        if content_psd_size is not None:
            psd_as_is_sizes.add(content_psd_size)
    elif request.form.get("psd_as_is") and upload_ai_enabled:
        # The generator's backdrop goes into every template; "exactly as uploaded"
        # belongs to custom-hero mode and returns when that mode is picked.
        background_notes.append(
            "AI hero image on: the generated backdrop goes into every template. \"Use these files exactly as "
            "uploaded\" applies in the custom hero mode, not here."
        )

    # A size-specific upload carries onto every same-ratio size with no file of its
    # own, scaled to fit; otherwise one design change needed two uploads. A near-miss
    # typed size (3480x2160) is flagged but used as typed.
    known_sizes = (set(size_templates.keys()) | set(SIZE_NAMES) | set(DEFAULT_SIZES)) - set(psd_template_paths)
    for typed in sorted(psd_template_paths):
        if typed in known_sizes:
            continue
        near = _near_miss_size(typed, known_sizes)
        if near is not None:
            background_warnings.append(
                f"{size_label(*typed)} on a PSD row isn't a size this batch or the app knows, but it is close to "
                f"{size_label(*near)} -- if that's the size you meant, fix the Size field on that row: as typed it "
                f"exports as a new {_ratio_label(*typed)} size instead of updating {size_label(*near)}."
            )
    # {target size: source size} for the notes and the as-is flag.
    ratio_matched_templates: dict = {}
    uploaded_sources = dict(psd_templates)
    if content_psd_size is not None:
        uploaded_sources[content_psd_size] = content_psd_image
    batch_sizes = set(size_templates.keys()) | set(sizes)
    # A file chosen this run beats a same-shape row still carrying the last run's
    # file, so dropping one new file updates both 9:16 rows.
    fresh_sources = {size: image for size, image in uploaded_sources.items() if size in fresh_psd_sizes}
    superseded_rows: dict = {}
    for target in sorted(batch_sizes):
        if target in uploaded_sources:
            if target in fresh_psd_sizes or not fresh_sources:
                continue
            source = _same_ratio_source(target, fresh_sources)
            if source is None:
                continue
            size_templates[target] = uploaded_sources[source]
            size_template_paths[target] = psd_template_paths[source]
            ratio_matched_templates[target] = source
            superseded_rows[target] = source
            # The row keeps the new file too, so the next Edit carries
            # it forward instead of the one it just replaced.
            row = psd_row_by_size.get(target)
            if row is not None:
                psd_file_paths[row] = psd_template_paths[source]
            continue
        source = _same_ratio_source(target, uploaded_sources)
        if source is None:
            continue
        size_templates[target] = uploaded_sources[source]
        size_template_paths[target] = (
            psd_template_paths.get(source) if source in psd_template_paths else content_psd_path
        )
        ratio_matched_templates[target] = source
        if source in psd_as_is_sizes:
            psd_as_is_sizes.add(target)

    # A file uploaded this run becomes default_templates/'s template for its size and
    # every same-shape size it carried onto; the replaced file is kept in
    # _template_backups/ and the row is let go so it can't shadow the template.
    form_field_overrides: dict = {}
    for row, typed, used in psd_size_snaps:
        background_notes.append(
            f"PSD row {row}: {size_label(*typed)} is a few pixels off {size_label(*used)}, so the file went to the "
            f"{size_label(*used)} slot (the Size field is set to that)."
        )
        form_field_overrides[f"psd_size_{row}"] = size_label(*used)
    if fresh_psd_sizes and not request.form.get("psd_make_saved"):
        background_warnings.append(
            "The PSD(s) you dropped were used for this run only -- \"The most recent upload is the "
            "template\" is unticked, so default_templates/ was not changed. Tick it (above the rows) "
            "and run again to make them the saved templates."
        )
    if request.form.get("psd_make_saved") and fresh_psd_sizes:
        saved_paths = _default_template_paths()
        stamp = _datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        promoted = []
        targets: dict = {}
        for size in fresh_psd_sizes:
            targets[size] = size
        for target, source in ratio_matched_templates.items():
            if source in fresh_psd_sizes and target in saved_paths and target not in targets:
                targets[target] = source
        for target, source in sorted(targets.items()):
            source_path = psd_template_paths.get(source)
            if source_path is None:
                continue
            # One PSD per size, named for the size (tester-720x480.psd). Anything else
            # claiming that size is moved to backups, so the name always says which.
            dest = templates_dir() / f"{SAVED_TEMPLATE_PREFIX}{size_label(*target)}.psd"
            try:
                templates_dir().mkdir(parents=True, exist_ok=True)
                TEMPLATE_BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
                for other in _files_claiming_size(target):
                    if other == dest:
                        continue
                    shutil.move(str(other), str(TEMPLATE_BACKUPS_DIR / f"{other.stem}.{stamp}{other.suffix}"))
                if dest.exists():
                    shutil.copy(dest, TEMPLATE_BACKUPS_DIR / f"{dest.stem}.{stamp}{dest.suffix}")
                shutil.copy(source_path, dest)
                promoted.append((target, source, dest))
            except OSError as exc:
                background_warnings.append(
                    f"Couldn't save the {size_label(*target)} template into default_templates/: {exc}"
                )
        for target, source, dest in promoted:
            background_notes.append(
                f"{size_label(*target)}: the file you uploaded"
                + (f" for {size_label(*source)}" if source != target else "")
                + f" is now the saved template ({dest.name}); the previous one is in _template_backups/. "
                "Every run starts from it from here on; the row keeps the file as a reminder of what was dropped."
            )
            row = psd_row_by_size.get(target)
            if row is not None and target == source:
                promoted_psd_rows.add(row)
    # Every template in play needs "logo", "description" and "product" layers.
    # A missing one skips its override silently, so fail loudly here instead.
    for (template_width, template_height), template_path in size_template_paths.items():
        template_layer_boxes = get_psd_layer_boxes(template_path)
        missing_layers = [
            name for name in REQUIRED_PSD_LAYERS if name not in template_layer_boxes
        ]
        if missing_layers:
            # A layer that exists but sits entirely off the canvas counts as absent above.
            # Name it separately: "missing" sends people hunting for a layer they can see.
            off_canvas = _off_canvas_layers(template_path, missing_layers)
            truly_missing = [n for n in missing_layers if n not in off_canvas]
            parts = []
            if truly_missing:
                parts.append("is missing required layer(s): " + ", ".join(truly_missing))
            for name, where in off_canvas.items():
                parts.append(
                    f"has a '{name}' layer that is entirely outside the {template_width}x{template_height} "
                    f"canvas ({where}) -- move it onto the artboard in Photoshop and save"
                )
            return _keep_submission(
                f"{size_label(template_width, template_height)} template "
                f"({Path(template_path).name}) " + "; ".join(parts)
                + ". Every PSD template needs 'logo', 'description', and 'product' layers "
                "(named exactly that, case-insensitive), with pixels inside the canvas."
            )

    # Layer overrides apply per size, each at that size's own layer bbox.
    # A PSD dropped this run is the design: the form's styling for its layers is
    # off for the run, else a glow recoloured in Photoshop came back overdrawn.
    if fresh_psd_drops:
        switched_off = _switch_off_form_layer_styling()
        if switched_off:
            background_notes.append(
                "A PSD was dropped this run (" + ", ".join(fresh_psd_drops) + "), so the form's own layer "
                "styling -- " + ", ".join(switched_off) + " -- was switched off and the file's styling is used. "
                "Tick a control again to override the file."
            )
    layer_header_text = (request.form.get("layer_header_text") or "").strip() or None
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
    layer_header_align = _parse_layer_align(request.form.get("layer_header_align"))
    layer_header_background = bool(request.form.get("layer_header_background"))
    layer_header_background_color = _parse_hex_color(
        request.form.get("layer_header_background_color"), default=(0, 0, 0)
    )
    layer_header_background_opacity = _parse_glow_opacity(
        request.form.get("layer_header_background_opacity")
    )
    layer_header_background_blur = _parse_band_blur(request.form.get("layer_header_background_blur"))
    layer_header_stroke_size = _parse_stroke_size(request.form.get("layer_header_stroke_size"))

    def _parse_form_shadow(key):
        """The form's drop shadow for a text layer, as the dict the
        renderer draws from (see _draw_text_shadow) -- or None when
        the box is off. Photoshop's own dials: opacity %, distance px,
        spread % (the solid part of the size), size px, angle deg."""
        if not request.form.get(f"layer_{key}_shadow"):
            return None

        def number(name, default, low, high):
            raw = (request.form.get(f"layer_{key}_shadow_{name}") or "").strip()
            try:
                value = float(raw) if raw else default
            except ValueError:
                value = default
            return max(low, min(high, value))

        return {
            "color": _parse_hex_color(request.form.get(f"layer_{key}_shadow_color"), default=(0, 0, 0)),
            "opacity": number("opacity", 75, 0, 100),
            "distance": number("distance", 5, 0, 500),
            "spread": number("spread", 0, 0, 100),
            "size": number("size", 5, 0, 250),
            "angle": number("angle", 120, -360, 360),
        }

    layer_header_shadow = _parse_form_shadow("header")
    layer_description_shadow = _parse_form_shadow("description")

    def _parse_picture_glow(key):
        """An outer glow for a picture layer (logo, product): colour,
        size in the PSD's pixels, opacity %. None when the box is off."""
        if not request.form.get(f"layer_{key}_glow"):
            return None
        raw_size = (request.form.get(f"layer_{key}_glow_size") or "").strip()
        raw_opacity = (request.form.get(f"layer_{key}_glow_opacity") or "").strip()
        try:
            size = float(raw_size) if raw_size else 12.0
        except ValueError:
            size = 12.0
        try:
            opacity = float(raw_opacity) if raw_opacity else 75.0
        except ValueError:
            opacity = 75.0
        return {
            "color": _parse_hex_color(request.form.get(f"layer_{key}_glow_color"), default=(255, 255, 255)),
            "size": max(0.0, min(250.0, size)),
            "opacity": max(0.0, min(100.0, opacity)),
        }

    def _parse_picture_stroke(key):
        """An outline round a picture layer (logo, product): colour and
        width in the PSD's pixels. None when the box is off."""
        if not request.form.get(f"layer_{key}_stroke"):
            return None
        raw_size = (request.form.get(f"layer_{key}_stroke_size") or "").strip()
        try:
            size = float(raw_size) if raw_size else 3.0
        except ValueError:
            size = 3.0
        return {
            "color": _parse_hex_color(request.form.get(f"layer_{key}_stroke_color"), default=(0, 0, 0)),
            "size": max(0.0, min(100.0, size)),
        }

    # Effects on the picture layers, keyed as _layer_effect_images() reads them.
    picture_layer_fx = {}
    for key in ("logo", "product"):
        spec = {}
        glow_spec = _parse_picture_glow(key)
        shadow_spec = _parse_form_shadow(key)
        stroke_spec = _parse_picture_stroke(key)
        if glow_spec and glow_spec["size"] > 0 and glow_spec["opacity"] > 0:
            spec["glow"] = glow_spec
        if shadow_spec:
            spec["shadow"] = shadow_spec
        if stroke_spec and stroke_spec["size"] > 0:
            spec["stroke"] = stroke_spec
        if spec:
            picture_layer_fx[key] = spec
    layer_header_stroke_color = _parse_hex_color(
        request.form.get("layer_header_stroke_color"), default=(0, 0, 0)
    )

    layer_description_text = (request.form.get("layer_description_text") or "").strip() or None
    # Blank family or size means "match the PSD's own layer style". A colour input
    # cannot be blank, so a separate checkbox says whether to use the picked colour.
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
    layer_description_align = _parse_layer_align(request.form.get("layer_description_align"))
    layer_description_background = bool(request.form.get("layer_description_background"))
    layer_description_background_color = _parse_hex_color(
        request.form.get("layer_description_background_color"), default=(0, 0, 0)
    )
    layer_description_background_opacity = _parse_glow_opacity(
        request.form.get("layer_description_background_opacity")
    )
    layer_description_background_blur = _parse_band_blur(
        request.form.get("layer_description_background_blur")
    )
    layer_description_stroke_size = _parse_stroke_size(request.form.get("layer_description_stroke_size"))
    layer_description_stroke_color = _parse_hex_color(
        request.form.get("layer_description_stroke_color"), default=(0, 0, 0)
    )

    # Legal copy is a named type layer like header and description. Defaults follow
    # the description, minus glow and backing band: small print is not decorated.
    layer_legal_text = (request.form.get("layer_legal_text") or "").strip() or None
    layer_legal_font_family = (request.form.get("layer_legal_font_family") or "").strip()
    if layer_legal_font_family not in VALID_FONT_FAMILIES:
        layer_legal_font_family = ""
    layer_legal_font_size = _parse_optional_font_size(request.form.get("layer_legal_font_size"))
    layer_legal_use_custom_color = bool(request.form.get("layer_legal_use_custom_color"))
    layer_legal_text_color = _parse_hex_color(
        request.form.get("layer_legal_text_color"), default=(26, 26, 26)
    )
    layer_legal_glow = bool(request.form.get("layer_legal_glow"))
    layer_legal_glow_color = _parse_hex_color(
        request.form.get("layer_legal_glow_color"), default=(255, 255, 255)
    )
    layer_legal_glow_size = _parse_glow_size(request.form.get("layer_legal_glow_size"))
    layer_legal_glow_opacity = _parse_glow_opacity(request.form.get("layer_legal_glow_opacity"))
    layer_legal_align = _parse_layer_align(request.form.get("layer_legal_align"))
    layer_legal_background = bool(request.form.get("layer_legal_background"))
    layer_legal_background_color = _parse_hex_color(
        request.form.get("layer_legal_background_color"), default=(0, 0, 0)
    )
    layer_legal_background_opacity = _parse_glow_opacity(
        request.form.get("layer_legal_background_opacity")
    )
    layer_legal_background_blur = _parse_band_blur(
        request.form.get("layer_legal_background_blur")
    )
    layer_legal_stroke_size = _parse_stroke_size(request.form.get("layer_legal_stroke_size"))
    layer_legal_stroke_color = _parse_hex_color(
        request.form.get("layer_legal_stroke_color"), default=(0, 0, 0)
    )

    # No position field for the CTA here: the template's own button box is filled.
    layer_cta_text = (request.form.get("layer_cta_text") or "").strip() or None
    layer_cta_font_family = (request.form.get("layer_cta_font_family") or "").strip()
    if layer_cta_font_family not in VALID_FONT_FAMILIES:
        layer_cta_font_family = "sans"
    layer_cta_font_size = _parse_optional_font_size(request.form.get("layer_cta_font_size"))
    layer_cta_button_color = _parse_hex_color(
        request.form.get("layer_cta_button_color"), default=CTA_BUTTON_COLOR_DEFAULT
    )
    layer_cta_text_color = _parse_hex_color(
        request.form.get("layer_cta_text_color"), default=CTA_TEXT_COLOR_DEFAULT
    )
    layer_cta_glow = bool(request.form.get("layer_cta_glow"))
    layer_cta_glow_color = _parse_hex_color(
        request.form.get("layer_cta_glow_color"), default=(255, 255, 255)
    )
    layer_cta_glow_size = _parse_glow_size(request.form.get("layer_cta_glow_size"))
    layer_cta_glow_opacity = _parse_glow_opacity(request.form.get("layer_cta_glow_opacity"))
    layer_cta_stroke_size = _parse_stroke_size(request.form.get("layer_cta_stroke_size"))
    # Label stroke is separate from the button's. One traces the shape edge, the
    # other the letterforms, so the same number cannot serve both.
    layer_cta_text_stroke_size = _parse_stroke_size(
        request.form.get("layer_cta_text_stroke_size")
    )
    layer_cta_text_stroke_color = _parse_hex_color(
        request.form.get("layer_cta_text_stroke_color"), default=(0, 0, 0)
    )
    # None, not 0: 0 is a real answer (square corners), so "not set" has
    # to be something else or the button could never keep its pill.
    layer_cta_radius = request.form.get("layer_cta_radius")
    layer_cta_radius = (
        _parse_stroke_size(layer_cta_radius) if str(layer_cta_radius or "").strip() else None
    )
    layer_cta_stroke_color = _parse_hex_color(
        request.form.get("layer_cta_stroke_color"), default=(0, 0, 0)
    )

    # Form fields and the saved run stay English so Edit shows what was typed.
    # Translation happens here, on the way into the templates.
    copy_language = (request.form.get("copy_language") or "en").strip().lower()
    if copy_language not in COPY_LANGUAGE_NAMES:
        copy_language = "en"
    translations_noted: set = set()
    english_behind = _english_behind_translations()
    campaign_copy = None  # worked out on first need, from this run's templates
    # The pre-translation text. Saving copy into templates writes this, so a French
    # run does not turn the English masters French.
    typed_copy_english = {
        "header": layer_header_text,
        "description": layer_description_text,
        "legal": layer_legal_text,
        "cta": layer_cta_text,
    }
    if copy_language != "en":
        localized = _localize_form_copy(
            copy_language,
            {
                "header": headline,
                "message": message,
                "CTA": cta_text,
                "header layer": layer_header_text,
                "description layer": layer_description_text,
                "legal layer": layer_legal_text,
                "CTA layer": layer_cta_text,
                "AI headline": upload_ai_headline,
            },
            background_notes,
            background_warnings,
        )
        headline = localized["header"]
        message = localized["message"]
        cta_text = localized["CTA"]
        layer_header_text = localized["header layer"]
        layer_description_text = localized["description layer"]
        layer_legal_text = localized["legal layer"]
        layer_cta_text = localized["CTA layer"]
        upload_ai_headline = localized["AI headline"]

    # A hide wins over content supplied for the same layer: honouring both would
    # draw the thing just asked to disappear.
    #
    # "locked" is the template talking, not the person. That box is ticked and
    # disabled because the layer is switched off in the saved templates, and a
    # disabled checkbox posts nothing, so a hidden input carries the value. It
    # must not drive the wipe below, for two reasons: a layer that really is
    # switched off is not drawn anyway, so wiping its box achieves nothing --
    # and the lock is computed when the PAGE renders, from the templates as
    # they were BEFORE this run. Upload a corrected template that has the layer
    # back on and the same submission still carried the old lock, so the run
    # wiped out the very layer the upload was fixing. Re-uploading a fixed
    # template could never take effect on the run that fixed it; you had to run
    # it twice and nothing on the page said so.
    hidden_layer_names = {
        name for name in HIDEABLE_LAYER_NAMES
        if any(
            value and value != "locked"
            for value in request.form.getlist(f"layer_{name}_hidden")
        )
    }
    # Copy typed for a hidden layer never appears, which reads as "my change did
    # not take". Say which box is doing it.
    typed_but_hidden = [
        name for name, text in (
            ("header", layer_header_text), ("description", layer_description_text),
            ("legal", layer_legal_text), ("cta", layer_cta_text),
        ) if text and name in hidden_layer_names
    ]
    if typed_but_hidden:
        background_warnings.append(
            "You typed copy for "
            + ", ".join(typed_but_hidden)
            + f" but {'its' if len(typed_but_hidden) == 1 else 'their'} Hide box is ticked, so "
            f"{'it is' if len(typed_but_hidden) == 1 else 'they are'} not drawn on any size. "
            "Untick the box under \"Hide layers\" to see the text."
        )
    if len(hidden_layer_names) >= len(HIDEABLE_LAYER_NAMES) - 1:
        # Hide boxes carry over between runs with the rest of the form, so every layer
        # hidden is almost never meant. It reads later as "the logo is not rendering".
        background_warnings.append(
            "Every template layer is hidden on this run (Hide layers: "
            + ", ".join(sorted(hidden_layer_names))
            + "), so each size shows only its background. Untick them under \"Hide layers\" "
            "if that isn't what you meant -- they are not remembered on the next fresh form."
        )

    layer_image_overrides: dict = {}  # {"logo"/"cta"/"product"/"background": Image.Image}
    layer_upload_paths: dict = {}  # {field_name: Path} -- for form_state.json, see below
    for layer_name, field_name in (
        ("logo", "layer_logo_image"),
        ("product", "layer_product_image"),
    ):
        layer_file = request.files.get(field_name)
        layer_fresh = layer_file is not None and bool(layer_file.filename)
        if layer_fresh:
            if not _allowed(layer_file.filename, PICTURE_UPLOAD_EXTENSIONS):
                return _keep_submission(
                    f"'{layer_file.filename}' isn't a supported file type for the {layer_name} layer update. "
                    "Accepted: " + ", ".join(PICTURE_UPLOAD_EXTENSIONS)
                )
            try:
                layer_path = _flatten_psd_upload(_save_upload(layer_file, uploads_dir))
            except ValueError as exc:
                return _keep_submission(f"{layer_name} layer update: {exc}")
        elif request.form.get(f"{field_name}_clear"):
            # Explicit removal signal for a carried-forward image: a file input cannot be
            # emptied for the user, so blank has to keep meaning "keep what is there".
            layer_path = None
        else:
            layer_path = _carry_forward_upload(field_name, uploads_dir, prior_job_dir, prior_form_state)
        if layer_path is None:
            continue
        layer_upload_paths[field_name] = layer_path
        try:
            layer_image = Image.open(layer_path).convert("RGBA")
        except Exception as exc:
            return _keep_submission(f"Couldn't read the {layer_name} layer update image: {exc}")
        # Background removal on every layer image: logos, CTAs and products are commonly
        # exported flat. Not on "background" itself, that upload is the intended frame.
        if layer_name != "background":
            layer_image = auto_transparent_background(layer_image)
        if layer_name in hidden_layer_names:
            # Still kept in layer_upload_paths, so the file survives an Edit and returns
            # when the layer is unhidden.
            continue
        layer_image_overrides[layer_name] = layer_image

    # The uploaded content PSD's layers override the other sizes, so one flagship
    # PSD carries the campaign. A layer uploaded by hand wins; this fills gaps.
    propagated_layer_names = set()
    # A generated backdrop is set first and wins outright. Uploads used to outrank
    # it, but on an Edit the carried-forward hero killed every new generation.
    # Never stored in layer_upload_paths: carried forward it overwrites the new file.
    if upload_ai_image is not None:
        layer_image_overrides["background"] = upload_ai_image.convert("RGBA")
    if upload_hero_image is not None and "background" not in layer_image_overrides:
        # The hero image reaches the templates exactly as the generated
        # backdrop does -- it just yields to one when the generator is on.
        layer_image_overrides["background"] = upload_hero_image
    if content_psd_provided:
        for layer_name, layer_image in _content_psd_layer_images(content_psd_path).items():
            if layer_name in layer_image_overrides:
                continue
            layer_image_overrides[layer_name] = layer_image
            propagated_layer_names.add(layer_name)

    # For a templated campaign default_templates/ is the source of truth: the batch
    # is those sizes plus anything uploaded now, and the size selections are dropped.
    if default_templates:
        sizes = sorted(set(size_templates.keys()))
    else:
        sizes = sorted(set(sizes) | set(size_templates.keys()))
    # Approved sizes are carried over as rendered: not regenerated, not billed.
    display_sizes = list(sizes)
    kept_sizes = [size for size in sizes if size in approved_prior_sizes]
    sizes = [size for size in sizes if size not in approved_prior_sizes]

    # Generated once and reused for every size, saved into uploads/ as hero_path so
    # nothing downstream, Edit's carry-forward included, treats it specially.
    if not hero_provided and ai_hero_enabled and any((w, h) not in size_templates for w, h in sizes):
        if upload_ai_allow_text:
            brand_words = product_name or headline
            prompt = ai_hero_prompt or (
                f'advertising creative for {brand_words or "the product"}, '
                + (f'the words "{brand_words}" set as the headline, ' if brand_words else "")
                + "a hero product image or a photographic background behind the type, "
                + "bold headline typography, clean layout"
            )
        else:
            prompt = ai_hero_prompt or (
                f"professional studio product photo of {product_name or headline or 'the product'}, "
                "clean background"
            )
        try:
            hero_width, hero_height = _generation_size(sizes, (1024, 1024))
            # One lettering setting for both generators: a run that wants lettering in the
            # campaign artwork does not want it stripped from the hero beside it.
            _report_progress(progress_token, 8, f"Generating the hero image with {ai_hero_provider}")
            generated_image, prompt, hero_attempts, hero_text = _generate_text_free(
                _provider(ai_hero_provider),
                prompt,
                hero_width,
                hero_height,
                allow_text=upload_ai_allow_text,
            )
            note = (
                f"Hero image generated with AI ({ai_hero_provider}) at "
                f"{generated_image.width}x{generated_image.height} -- prompt: \"{prompt}\"."
            )
            if hero_attempts > 1:
                note += (
                    f" Regenerated {hero_attempts - 1} more time"
                    f"{'s' if hero_attempts > 2 else ''} because the first result had text in it."
                )
            background_notes.append(note)
            if upload_ai_allow_text:
                background_notes.append(
                    "Hero image: text was allowed, so no no-text instruction was sent and "
                    "nothing was checked or painted out."
                )
            if hero_text.found_text:
                generated_image, cleaned_note, cleaned_warning = _clean_text_out(
                    generated_image, hero_text, "hero image", hero_attempts
                )
                if cleaned_note:
                    background_notes.append(cleaned_note)
                if cleaned_warning:
                    background_warnings.append(cleaned_warning)
        except ImageProviderError as exc:
            # Same resilience as src/pipeline.py: a flaky free API falls back to the offline
            # placeholder and says so, rather than failing the request.
            app.logger.warning(
                "AI provider %r failed for the hero image: %s", ai_hero_provider, exc
            )
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
        # Name the AI-hero checkbox in the error, it is the one-click fix for this.
        missing_labels = ", ".join(size_label(width, height) for width, height in missing_sizes)
        return _keep_submission(
            f"These sizes need either a hero image or a matching PSD template: {missing_labels}. "
            "Or check \"Generate a hero image with AI\" under the Hero image field below to have "
            "one generated automatically instead."
        )

    video_frame_seconds = None
    if hero_provided and hero_path.suffix.lower() in VIDEO_EXTENSIONS:
        raw_seconds = (request.form.get("video_frame_seconds") or "").strip()
        if raw_seconds:
            try:
                video_frame_seconds = float(raw_seconds)
            except ValueError:
                return _keep_submission(f"'{raw_seconds}' isn't a valid number of seconds -- using the middle of the video instead.")

    hero_image = None
    if hero_provided:
        try:
            hero_image = open_as_rgb(hero_path, frame_seconds=video_frame_seconds)
        except ValueError as exc:
            flash(str(exc))

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
            return _keep_submission(
                f"Logo file '{logo_file.filename}' isn't a supported type -- use PNG or WEBP "
                "(needs transparency to composite cleanly)."
            )
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
            return _keep_submission(
                f"Badge file '{badge_file.filename}' isn't a supported type. Accepted: "
                + ", ".join(ALLOWED_BADGE_EXTENSIONS)
            )
        badge_path = _save_upload(badge_file, uploads_dir)
    else:
        badge_path = _carry_forward_upload("badge_image", uploads_dir, prior_job_dir, prior_form_state)
    if badge_path is not None:
        badge_image_obj = Image.open(badge_path).convert("RGBA")

    # OCRs uploads for well-known brand names. Warning only, never blocks, and finds
    # nothing when the tesseract binary is absent.
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

    # Full-ad mode generates once per size at that size's aspect: a laid-out ad will
    # not crop from one square without cutting the headline. Expensive, so say so.
    full_ad_templates = {}
    if upload_ai_full_ad and sizes:
        full_ad_prompt = _build_full_ad_prompt(
            product_name,
            campaign_message,
            upload_ai_headline or layer_header_text,
            layer_cta_text,
            audience,
            market,
            brand_colors=brand_colors,
        )
        if upload_ai_reference_image is not None:
            full_ad_prompt = f"{full_ad_prompt}, {reference_look_phrase(upload_ai_reference_image)}"
        # Constructing the provider is where a missing IDEOGRAM_API_KEY surfaces, and
        # unhandled it was a 500. A full ad has no offline fallback to degrade to.
        try:
            provider_for_ads = _provider(upload_ai_provider)
        except ImageProviderError as exc:
            return _keep_submission(f"Can't generate the whole ad with {upload_ai_provider}: {exc}")
        background_notes.append(
            f"Full ad mode: {len(sizes)} separate generation(s) with "
            f"{upload_ai_provider}, one per output size -- prompt: \"{full_ad_prompt}\" "
            f"[excluded: {FULL_AD_NEGATIVE_CLAUSE}]."
        )
        background_warnings.append(
            "Whole ad mode: the words in these pictures were asked for -- this box tells the model "
            "to paint the headline, product name and CTA into the image itself. For a text-free "
            "backdrop under your templates, untick \"Generate the whole ad\" and run again."
        )
        if upload_ai_provider == "ideogram" and upload_ai_speed == "TURBO":
            background_warnings.append(
                "Whole ad on Turbo: Turbo is the roughest pass for typography and tends to misspell "
                "and smear type. Fine for finding a layout; switch Rendering to Quality for the one you keep."
            )
        headline_words = len((upload_ai_headline or layer_header_text or campaign_message or "").split())
        if headline_words > FULL_AD_HEADLINE_WORDS:
            background_warnings.append(
                f"Whole ad: the headline is {headline_words} words. The model sets up to about "
                f"{FULL_AD_HEADLINE_WORDS} cleanly and starts misspelling after that -- type a shorter "
                "\"AI headline\" (or header text) for clean lettering."
            )
        render_mode = (
            # rewrite_prompt off: MagicPrompt paraphrases the quoted headline, which then
            # gets set misspelled.
            {"photographic": False, "rewrite_prompt": False}
            if getattr(provider_for_ads, "supports_render_mode", False) and not upload_ai_reference_bytes
            else {}
        )
        for ad_index, (width, height) in enumerate(sizes):
            _report_progress(
                progress_token, 10 + 25 * ad_index / max(len(sizes), 1),
                f"Generating the whole ad {size_label(width, height)} ({ad_index + 1} of {len(sizes)})",
            )
            try:
                ad_image = provider_for_ads.generate(
                    full_ad_prompt,
                    width=width,
                    height=height,
                    negative_prompt=", ".join(
                        c for c in (FULL_AD_NEGATIVE_CLAUSE, PALETTE_NEGATIVE_CLAUSE if brand_colors else None) if c
                    ),
                    **render_mode,
                    **(
                        {"style_reference": upload_ai_reference_bytes}
                        if upload_ai_reference_bytes
                        and getattr(provider_for_ads, "supports_style_reference", False)
                        else {}
                    ),
                )
            except ImageProviderError as exc:
                background_warnings.append(
                    f"{size_label(width, height)}: full ad generation failed ({exc}) -- "
                    "this size fell back to the saved template."
                )
                continue
            if ad_image.width < width or ad_image.height < height:
                ad_image = upscale_to_cover(ad_image, (width, height))
            size_templates[(width, height)] = ad_image
            # No path means no layer override, no PSD rebuild and no text drawn over the
            # model's own. The template is kept separately for the whole-ad PSD.
            full_ad_templates[(width, height)] = size_template_paths.pop((width, height), None)

    creatives = []
    for size_index, (width, height) in enumerate(sizes):
        _report_progress(
            progress_token, 35 + 57 * size_index / max(len(sizes), 1),
            f"Rendering {size_label(width, height)} ({size_index + 1} of {len(sizes)})",
        )
        # Only sizes that genuinely enlarge past the source: a 160x600 cut from 1024x1024
        # loses nothing, a 1920x1080 is stretched to near double.
        if upload_ai_source_size and (
            width > upload_ai_source_size[0] or height > upload_ai_source_size[1]
        ):
            background_warnings.append(
                f"{size_label(width, height)}: enlarged from the "
                f"{upload_ai_source_size[0]}x{upload_ai_source_size[1]} the "
                f"'{upload_ai_provider}' provider returned, so this size will look softer. "
                "It still renders -- try a provider without that cap, or supply the artwork "
                "yourself, if it matters for this size."
            )
        background_image = size_templates.get((width, height), hero_image)
        is_template_size = (width, height) in size_templates
        # Only the render_creative() path exports one; a PSD template size is already a
        # hand-built creative.
        psd_filename = None
        # psd-tools can only author rasterized pixel layers (create_pixel_layer; no
        # setter on TypeLayer.text), so a rebuild holds pictures of words. Ship the
        # source template alongside for its live type.
        source_psd_filename = None
        source_psd_download_name = None
        if (width, height) in psd_as_is_sizes:
            if request.form.get("psd_as_is_hero") and "background" in layer_image_overrides:
                background_notes.append(
                    f"{size_label(width, height)} used the template as uploaded, with the hero image put into its "
                    "background layer -- its own text and layers stand, no typed copy or hide boxes applied."
                )
            else:
                background_notes.append(
                    f"{size_label(width, height)} used your uploaded PSD exactly as uploaded (\"Use these files exactly "
                    "as uploaded\" ticked): its own background and text, with no hero image, typed copy or hide boxes applied."
                )
        elif (width, height) in superseded_rows:
            source = superseded_rows[(width, height)]
            background_notes.append(
                f"{size_label(width, height)} used the PSD you uploaded this run for {size_label(*source)}, scaled "
                f"to fit -- same proportions -- in place of the file its own row was still carrying from an "
                "earlier run. Its row now holds the new file."
            )
        elif (width, height) in ratio_matched_templates:
            source = ratio_matched_templates[(width, height)]
            background_notes.append(
                f"{size_label(width, height)} used the PSD you uploaded for {size_label(*source)}, scaled to fit "
                "-- same proportions, so the layout carries over as designed. Upload a file on a "
                f"{size_label(width, height)} row to give this size its own."
            )
        elif (width, height) in psd_templates:
            background_notes.append(
                f"{size_label(width, height)} used your uploaded PSD template as its layout -- the hero "
                "image goes into its background layer and copy typed on the form is drawn into its text "
                "layers. Tick \"Use these files exactly as uploaded\" above the rows to use it as it is."
            )
        elif (width, height) in default_templates:
            background_notes.append(
                f"{size_label(width, height)} used the saved default template as-is -- "
                "no header/message/logo/badge/CTA overlay is added on top of a template."
            )

        if is_template_size:
            # Hand back the template PSD itself rather than a rebuild: layer names and boxes
            # read reliably, per-layer edited content does not. A copy failure never blocks.
            psd_source_path = size_template_paths.get((width, height))
            if psd_source_path is not None:
                try:
                    psd_candidate_filename = f"{file_name_prefix}_{size_label(width, height)}.psd"
                    shutil.copy(psd_source_path, job_dir / psd_candidate_filename)
                    psd_filename = psd_candidate_filename
                except Exception:
                    psd_filename = None
            elif upload_ai_full_ad and (width, height) in full_ad_templates:
                # A whole-ad generation is one flat picture; its PSD is a reconstruction (see
                # src/ad_split.py) with the size's real layers hidden underneath.
                try:
                    split = split_ad(background_image)
                    psd_candidate_filename = f"{file_name_prefix}_{size_label(width, height)}.psd"
                    written = write_whole_ad_psd(
                        background_image,
                        split.layers(),
                        job_dir / psd_candidate_filename,
                        template_path=full_ad_templates.get((width, height)),
                        copy={
                            "header": upload_ai_headline or layer_header_text or campaign_message,
                            "description": layer_description_text or campaign_message,
                            "cta": layer_cta_text,
                        },
                    )
                    if written:
                        psd_filename = psd_candidate_filename
                        real = [n for n in written if n not in {name for name, _ in split.layers()}]
                        background_notes.append(
                            f"{size_label(width, height)}: whole-ad PSD -- layers {', '.join(written)}. "
                            + " ".join(split.notes)
                            + (
                                f" The template's own {', '.join(real)} are in the file switched off, "
                                "the type retyped to this brief's copy, for retouching."
                                if real else ""
                            )
                        )
                    else:
                        background_warnings.append(
                            f"{size_label(width, height)}: couldn't write the whole-ad PSD; the PNG is unaffected."
                        )
                except Exception as exc:  # noqa: BLE001
                    background_warnings.append(
                        f"{size_label(width, height)}: couldn't split the generated ad into layers ({exc}); "
                        "the PNG is unaffected."
                    )

            # A template is already a finished creative for this size, so no generic overlay
            # goes on top. The fit only covers a PSD whose pixels miss its named size.
            if fit_mode == "contain":
                final_image = resize_to_contain(background_image, (width, height))
            else:
                final_image = center_crop_to_ratio(background_image, (width, height))

            # Every override that can redraw has to open this gate: it once listed only
            # header and description, so a CTA-only run rendered the template untouched.
            # With a language set, a layer with nothing typed draws the template's own words.
            text_only_size = (width, height) in psd_as_is_sizes
            # Hide boxes are a custom-hero setting: an as-uploaded size ignores them, so its
            # PSD and clip must too, else the download opened with every layer switched off.
            size_hidden_layer_names = set() if text_only_size else hidden_layer_names
            # An as-uploaded size keeps its own text and layers, taking the hero into its
            # background layer and nothing else.
            hero_into_as_is = (
                text_only_size
                and bool(request.form.get("psd_as_is_hero"))
                and "background" in layer_image_overrides
            )
            size_image_overrides = (
                {"background": layer_image_overrides["background"]} if hero_into_as_is
                else ({} if text_only_size else layer_image_overrides)
            )
            size_text = {
                "header": None if text_only_size else layer_header_text,
                "description": None if text_only_size else layer_description_text,
                "legal": None if text_only_size else layer_legal_text,
                "cta": None if text_only_size else layer_cta_text,
            }
            # A text layer is redrawn only when its words change: a language chosen, copy
            # typed. Otherwise its pixels stay the file's pixels.
            own_text_layers: dict = {}
            # Which named layers are live type and which are pictures --
            # a rasterised header or description is a picture.
            type_layers_in_template: set = set()
            pixel_layers_in_template: set = set()
            pictures_noted: set = set()
            if (width, height) in size_template_paths:
                type_layers_in_template = set(get_psd_text_layers(size_template_paths.get((width, height))) or {})
                pixel_layers_in_template = get_psd_pixel_layer_names(size_template_paths.get((width, height)))
                # visible_only: a layer switched off in Photoshop has no
                # words on the creative to translate.
                own_text_layers = get_psd_text_layers(size_template_paths.get((width, height)), visible_only=True) or {}
                # The CTA's words sit on a layer inside its group, where get_psd_text_layers()
                # does not look. They still need the language like the other three.
                if "cta" not in own_text_layers:
                    cta_group_words = get_psd_group_text(size_template_paths.get((width, height)), "cta", visible_only=True)
                    if cta_group_words:
                        own_text_layers["cta"] = cta_group_words
                for layer_key in ("header", "description", "legal", "cta"):
                    # A hide box does not apply to an as-uploaded size, so its words are on the
                    # creative and need the language too.
                    if size_text[layer_key] or (layer_key in hidden_layer_names and not text_only_size):
                        continue
                    own_words = (own_text_layers.get(layer_key) or "").strip()
                    # Placeholder filler such as "Lorem ipsum" is not the campaign copy; draw what
                    # the other sizes carry instead.
                    if layer_key in own_text_layers and _is_placeholder_copy(own_words):
                        if campaign_copy is None:
                            campaign_copy = _campaign_copy_from_templates(size_template_paths, english_behind)
                        stand_in = campaign_copy.get(layer_key)
                        if stand_in and copy_language == "en":
                            size_text[layer_key] = stand_in
                            background_notes.append(
                                f"{size_label(width, height)}: the template's {layer_key} is placeholder copy "
                                f"(\"{own_words[:40].replace(chr(13), ' / ')}...\"), so the campaign's {layer_key} from the "
                                "other sizes is drawn instead."
                            )
                            continue
                        if stand_in:
                            own_words = stand_in
                            background_notes.append(
                                f"{size_label(width, height)}: the template's {layer_key} is placeholder copy, so the "
                                f"campaign's {layer_key} from the other sizes is translated and drawn instead."
                            )
                    if not own_words:
                        continue
                    file_words = own_words
                    # A template exported from an earlier French or Spanish run holds that language.
                    # Translations start from the English behind it, not from the file's own words.
                    english_source = _english_source_of(own_words, english_behind)
                    if english_source:
                        if (own_words, english_source) not in translations_noted:
                            translations_noted.add((own_words, english_source))
                            background_notes.append(
                                f"The template's {layer_key} is a translation this app made "
                                f"(\"{own_words.replace(chr(13), ' / ')}\"); working from the English behind it: "
                                f"\"{english_source.replace(chr(13), ' / ')}\"."
                            )
                        own_words = english_source
                    if copy_language == "en":
                        if english_source:
                            size_text[layer_key] = english_source
                        continue
                    # Translate line by line where the designer broke the lines, so the layout and
                    # per-line styling survive. Costs one line of context per call.
                    own_lines = [line.strip() for line in own_words.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
                    if len(own_lines) >= 2:
                        translated_lines = _localize_form_copy(
                            copy_language, {f"line {n}": line for n, line in enumerate(own_lines, 1)}, [], background_warnings
                        )
                        own_words = "\r".join(own_lines)
                        translated = "\r".join(translated_lines[f"line {n}"] for n in range(1, len(own_lines) + 1))
                    else:
                        translated = _localize_form_copy(
                            copy_language, {f"{layer_key} (template's own)": own_words}, [], background_warnings
                        )[f"{layer_key} (template's own)"]
                    # Already in the chosen language: leave Photoshop's own rendering, effects and
                    # all, rather than redrawing it.
                    if translated and translated.strip() != file_words.strip() and translated != own_words:
                        size_text[layer_key] = translated
                        if (own_words, translated) not in translations_noted:
                            translations_noted.add((own_words, translated))
                            background_notes.append(
                                f"{COPY_LANGUAGE_ENGLISH_NAMES.get(copy_language, copy_language)} copy -- "
                                f"the template's {layer_key}: \"{own_words.replace(chr(13), ' / ')}\" -> "
                                f"\"{translated.replace(chr(13), ' / ')}\"."
                            )
            localized_template_text = bool(
                any(size_text[k] and size_text[k] != {
                    "header": layer_header_text, "description": layer_description_text, "legal": layer_legal_text,
                    "cta": layer_cta_text,
                }[k] for k in ("header", "description", "legal", "cta"))
            )
            # The button block reads this, not the typed field, so a translated label
            # redraws the button too.
            size_cta_text = size_text["cta"]

            if (
                (
                layer_header_text
                or layer_description_text
                or layer_legal_text
                or layer_cta_text
                or layer_header_use_custom_color
                or layer_description_use_custom_color
                or layer_legal_use_custom_color
                or layer_header_glow
                or layer_description_glow
                or layer_header_shadow
                or layer_description_shadow
                or picture_layer_fx
                or layer_legal_glow
                or layer_cta_glow
                or layer_header_background
                or layer_description_background
                or layer_legal_background
                or layer_header_font_family
                or layer_description_font_family
                or layer_legal_font_family
                or layer_cta_font_family
                or layer_header_font_size
                or layer_description_font_size
                or layer_legal_font_size
                or layer_cta_font_size
                or layer_header_stroke_size
                or layer_description_stroke_size
                or layer_legal_stroke_size
                or layer_cta_stroke_size
                or layer_image_overrides
                or hidden_layer_names
                ) and not text_only_size
                or localized_template_text
                or hero_into_as_is
            ) and (width, height) in size_template_paths:
                psd_path_for_size = size_template_paths.get((width, height))
                layer_boxes = get_psd_layer_boxes(psd_path_for_size)
                applied_layers = []
                # Per-layer isolated RGBA patches, keyed by lowercased layer name, reused by the
                # layered PSD export so it shows real layers instead of one flattened image.
                export_layer_patches: dict = {}
                # The same words without the form's background box --
                # what goes inside the type layers (see below).
                words_only_patches: dict = {}
                # layer name -> the font size apply_layer_text_override()
                # settled on for it this size. See where it is filled.
                rendered_font_sizes: dict = {}
                # The CTA label's own glyphs, kept apart from the patch
                # that carries the whole button. See where it is set.
                cta_label_patch = None
                # Pillow's flattened composite, not a psd-tools recomposite: psd-tools renders
                # text and effects unlike Photoshop, so it supplies masks here, never pixels.
                pristine_final_image = final_image.copy()

                # Layer boxes are in the PSD's pixel space; final_image has been fit to (width,
                # height). A PSD a few pixels off its filename size drifts every box, leaving a
                # sliver of the original content at the edge of an override.
                psd_canvas_size = get_psd_canvas_size(psd_path_for_size) if psd_path_for_size else None
                if psd_canvas_size:
                    layer_boxes = {
                        name: map_box_through_fit(box, psd_canvas_size, (width, height), fit_mode)
                        for name, box in layer_boxes.items()
                    }
                # Drawing boxes, inset TEXT_EDGE_PADDING_PX from the canvas edge; the originals
                # stay for cleaning. Must come after the remap, or text lands at file coordinates.
                draw_boxes = _inset_boxes_to_canvas(layer_boxes, (width, height))

                def _fit_rgba_like_final_image(rgba_image):
                    # resize_to_contain() flattens to RGB for its letterbox blur, dropping alpha,
                    # so "contain" is done by hand: scale to fit, centred, on a transparent canvas.
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

                # Whether "background" was also uploaded this request (it runs first), so this
                # box's cleanup does not reintroduce the replaced backdrop. Never true for a
                # text-only size: with no new backdrop the wipe did nothing and old words showed.
                has_background_layer = "background" in {
                    n.strip().lower() for n in get_psd_layer_names(psd_path_for_size)
                } if psd_path_for_size is not None else False
                background_replaced_this_request = (
                    "background" in size_image_overrides
                    and not ("background" in propagated_layer_names and (width, height) == content_psd_size)
                )
                # The masked restore in _clean_layer_box() pulls from the original composite, so
                # it needs to know which layers must no longer come from there.
                overridden_layer_names = {
                    name
                    for name in size_image_overrides
                    if not (name in propagated_layer_names and (width, height) == content_psd_size)
                } | (set() if text_only_size else hidden_layer_names)
                # Backdrop-only image to wipe boxes to once this request has replaced the
                # background; the PSD's own backdrop is then the wrong one. None unless that ran.
                background_only_image = None

                def _clean_layer_box(target_box, layer_name, full_box=False, restore_others=False):
                    # Patch in the PSD's real pixels with the layer hidden before drawing anything
                    # new, so whatever the new content misses still reads correctly.
                    nonlocal final_image
                    if psd_path_for_size is None:
                        return
                    if full_box and background_replaced_this_request:
                        # Same whole-box wipe, but against the background this request just set.
                        if background_only_image is None:
                            return
                        final_image.paste(background_only_image.crop(target_box), target_box[:2])
                        # Then every other layer's pixels back on top inside the box, or the bare
                        # hero plate shows as a band behind the redrawn words.
                        others = get_psd_layer_foreground(
                            psd_path_for_size,
                            sorted({layer_name, f"{layer_name} (rendered)", "background"} | overridden_layer_names),
                        )
                        if others is not None:
                            others = _fit_rgba_like_final_image(others)
                            patch = others.crop(target_box)
                            final_image.paste(patch, target_box[:2], mask=patch.split()[3])
                        return
                    if background_replaced_this_request and layer_name != "background":
                        # Hide this layer and "background" together: hiding only the layer
                        # recomposes against the PSD's original backdrop and undoes the override.
                        # Wipe the box to the new background first, or old artwork survives.
                        if background_only_image is not None:
                            final_image.paste(
                                background_only_image.crop(target_box), target_box[:2]
                            )
                        # Hide every overridden layer, not just this one: the restore pulls from
                        # the original, so the last override would resurrect all the earlier ones.
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
                        # Replace, do not overprint. A text box routinely overlaps other artwork,
                        # so composite everything but the background to get the box's true backdrop.
                        clean_bg = get_psd_backdrop(psd_path_for_size)
                    if clean_bg is None:
                        clean_bg = get_psd_layer_background(psd_path_for_size, layer_name)
                    if clean_bg is None:
                        return
                    if clean_bg.size != final_image.size:
                        # Fit clean_bg with the same transform that produced final_image; a plain
                        # stretch-resize misaligns the patch from the mapped target_box.
                        if fit_mode == "contain":
                            clean_bg = resize_to_contain(clean_bg, final_image.size)
                        else:
                            clean_bg = center_crop_to_ratio(clean_bg, final_image.size)
                    final_image.paste(clean_bg.crop(target_box), target_box[:2])
                    if full_box and restore_others:
                        # The box was widened past the layer's edge to take its effects, so put
                        # the other layers' pixels back in that ring.
                        others = get_psd_layer_foreground(
                            psd_path_for_size,
                            [layer_name, f"{layer_name} (rendered)", "background"],
                        )
                        if others is not None:
                            others = _fit_rgba_like_final_image(others)
                            patch = others.crop(target_box)
                            final_image.paste(patch, target_box[:2], mask=patch.split()[3])

                # Background is processed first whatever the form order: its box is the whole
                # canvas and would wipe any logo/CTA/product override drawn before it.
                ordered_layer_names = sorted(
                    size_image_overrides.keys(), key=lambda name: name != "background"
                )
                for layer_name in ordered_layer_names:
                    override_image = size_image_overrides[layer_name]
                    box = layer_boxes.get(layer_name)
                    if (
                        layer_name == "background"
                        and (box is None or (box[2] - box[0]) * (box[3] - box[1]) < 0.01 * final_image.width * final_image.height)
                    ):
                        # No background layer, or one emptied in Photoshop, means no box of its own:
                        # the hero fills the canvas. Skipping it left the old backdrop, and the wipe
                        # below then doubled every redrawn text layer over its old words.
                        box = (0, 0, final_image.width, final_image.height)
                    if box is None:
                        continue
                    if layer_name in propagated_layer_names and (width, height) == content_psd_size:
                        # This size is the uploaded PSD; reapplying its own layers only round-trips
                        # them through a fit/crop.
                        continue
                    if layer_name == "background":
                        # Model-laid-out artwork fits rather than fills, cropping cuts the words. A
                        # plain backdrop still crops: letterbox margins on a texture look wrong.
                        background_fit = (
                            "contain" if (upload_ai_allow_text or upload_hero_fit == "contain") else "crop"
                        )
                        final_image = apply_layer_background_override(
                            final_image, box, override_image, fit=background_fit
                        )
                        background_only_image = final_image.copy()
                        # Layers return over the new backdrop with styles: their alpha says where
                        # the pixels are, the styles come from the flattened picture's difference.
                        layers_alpha_source = (
                            get_psd_layer_foreground(psd_path_for_size, layer_name, effects=False)
                            if psd_path_for_size is not None
                            else None
                        )
                        if layers_alpha_source is None and psd_path_for_size is not None and not has_background_layer:
                            # No background layer, so every visible layer is foreground.
                            layers_alpha_source = get_psd_composite_rgba(psd_path_for_size)
                        old_backdrop = (
                            get_psd_backdrop(psd_path_for_size, keep_layer_names=(layer_name,))
                            if psd_path_for_size is not None
                            else None
                        )
                        if old_backdrop is None and psd_path_for_size is not None and not has_background_layer:
                            # No background layer to compare against: the file's own
                            # picture shows transparency as white, so white is the
                            # old backdrop.
                            old_backdrop = Image.new("RGBA", psd_canvas_size or final_image.size, (255, 255, 255, 255))
                        if layers_alpha_source is not None and old_backdrop is not None:
                            layers_alpha_source = _fit_rgba_like_final_image(layers_alpha_source)
                            old_backdrop = _fit_rgba_like_final_image(old_backdrop.convert("RGBA"))
                            # Band around the layers within which differences from
                            # the old backdrop count as layer styles worth carrying.
                            styles_reach = max(
                                [get_psd_layer_effect_reach(psd_path_for_size, name) for name in get_psd_layer_names(psd_path_for_size)]
                                or [0]
                            )
                            styles_reach = int(math.ceil(styles_reach * _template_scale(psd_canvas_size, (width, height), fit_mode))) + 6
                            final_image = carry_flattened_effects(
                                pristine_final_image, old_backdrop, final_image, layers_alpha_source.split()[3],
                                reach=styles_reach,
                            )
                        elif layers_alpha_source is not None:
                            layers_alpha_source = _fit_rgba_like_final_image(layers_alpha_source)
                            final_image.paste(pristine_final_image, mask=layers_alpha_source.split()[3])
                        export_layer_patches[layer_name] = apply_layer_background_override(
                            Image.new("RGBA", final_image.size, (0, 0, 0, 0)),
                            box,
                            override_image,
                            keep_alpha=True,
                            fit=background_fit,
                        )
                    else:
                        # A layer style reaches past the layer's pixels, and the background step put
                        # that halo back. Widen the box by the reach, wipe it, restore the others.
                        reach = get_psd_layer_effect_reach(psd_path_for_size, layer_name) if psd_path_for_size else 0
                        if reach > 0:
                            pad = int(math.ceil(reach * _template_scale(psd_canvas_size, (width, height), fit_mode))) + 2
                            wide = (
                                max(0, box[0] - pad), max(0, box[1] - pad),
                                min(final_image.width, box[2] + pad), min(final_image.height, box[3] + pad),
                            )
                            _clean_layer_box(wide, layer_name, full_box=True, restore_others=True)
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

                # Wipe each hidden layer's whole box, same "replace, do not overprint" rule.
                # Skip layers the template already hides: boxes overlap, so wiping an unused
                # header's banner takes the top off the logo under it.
                visible_in_template = (
                    get_psd_visible_layers(psd_path_for_size)
                    if psd_path_for_size is not None
                    else set()
                )
                buried_in_template = (
                    get_psd_buried_layers(psd_path_for_size)
                    if psd_path_for_size is not None
                    else set()
                )
                if buried_in_template:
                    background_notes.append(
                        f"{size_label(width, height)}: "
                        + ", ".join(sorted(buried_in_template))
                        + (" sit" if len(buried_in_template) > 1 else " sits")
                        + " below the background in that size's template, so "
                        + ("they are" if len(buried_in_template) > 1 else "it is")
                        + " covered and not drawn -- as in Photoshop. Drag the layer above "
                        "the background and re-save the template to use it."
                    )
                # Form glow and drop shadow for the picture layers, drawn under the layer's own
                # pixels. The live-text file and saved template get them as real layer effects.
                picture_fx_applied: dict = {}
                for fx_key, fx_spec in (picture_layer_fx or {}).items():
                    if text_only_size or fx_key in size_hidden_layer_names or fx_key in buried_in_template:
                        continue
                    if visible_in_template and fx_key not in visible_in_template:
                        continue
                    layer_rgba = export_layer_patches.get(fx_key)
                    if layer_rgba is None and psd_path_for_size is not None:
                        own = get_psd_layer_rgba(psd_path_for_size, fx_key)
                        layer_rgba = _fit_rgba_like_final_image(own) if own is not None else None
                    if layer_rgba is None:
                        continue
                    fx_scale = _template_scale(psd_canvas_size, (width, height), fit_mode)
                    scaled = {}
                    for fx_name, fx_values in fx_spec.items():
                        scaled[fx_name] = dict(fx_values)
                        for measure in ("size", "distance"):
                            if measure in scaled[fx_name]:
                                scaled[fx_name][measure] = float(scaled[fx_name][measure]) * fx_scale
                    final_image = draw_layer_effects(final_image, layer_rgba, scaled)
                    export_layer_patches[fx_key] = draw_layer_effects(
                        Image.new("RGBA", final_image.size, (0, 0, 0, 0)), layer_rgba, scaled, keep_alpha=True
                    )
                    applied_layers.append(f"{fx_key} (" + " + ".join(sorted(fx_spec)) + ")")
                    picture_fx_applied[fx_key] = fx_spec

                # Report layers the template itself switches off: nothing else on the page
                # explains why one size came back with no product or no button.
                if visible_in_template:
                    # Check against every layer the template has: a switched-off group reports an
                    # empty bbox and drops out of the box map, so it would go unmentioned.
                    switched_off_here = sorted(
                        name
                        for name in get_psd_layer_names(psd_path_for_size)
                        if name not in visible_in_template
                        and name not in size_hidden_layer_names
                    )
                    if switched_off_here:
                        background_notes.append(
                            f"{size_label(width, height)}: "
                            + ", ".join(switched_off_here)
                            + " switched off in that size's template, so "
                            + ("they were" if len(switched_off_here) > 1 else "it was")
                            + " not drawn. Turn the layer back on in Photoshop and re-save "
                            "the template to use it."
                        )

                for layer_name in sorted([] if text_only_size else hidden_layer_names):
                    hidden_box = layer_boxes.get(layer_name)
                    if hidden_box is None:
                        # This size's template simply has no such layer.
                        continue
                    if visible_in_template and layer_name not in visible_in_template:
                        continue
                    # Not full_box: boxes overlap, and wiping the product's whole box took the
                    # corner off the CTA under it. Widen by the layer's effect reach instead,
                    # or a hidden header leaves its drop shadow behind.
                    reach = get_psd_layer_effect_reach(psd_path_for_size, layer_name) if psd_path_for_size else 0
                    if reach > 0:
                        pad = int(math.ceil(reach * _template_scale(psd_canvas_size, (width, height), fit_mode))) + 2
                        wide = (
                            max(0, hidden_box[0] - pad), max(0, hidden_box[1] - pad),
                            min(final_image.width, hidden_box[2] + pad), min(final_image.height, hidden_box[3] + pad),
                        )
                        _clean_layer_box(wide, layer_name, full_box=True, restore_others=True)
                    else:
                        _clean_layer_box(hidden_box, layer_name)
                    applied_layers.append(f"{layer_name} (hidden)")

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
                    background_blur=0,
                    stroke_size=0,
                    stroke_color=(0, 0, 0),
                    box_override=None,
                    clean=True,
                    shadow=None,
                ):
                    # Shared by every text-layer override: PSD font settings as defaults, per-field
                    # overrides, shrink-to-fit into the layer box (apply_layer_text_override()).
                    nonlocal final_image
                    if layer_key in size_hidden_layer_names:
                        # Hidden wins over any wording or styling for the same layer, and the box
                        # has already been wiped for it.
                        return
                    # box_override points this at a box other than the named layer's own: the CTA
                    # group's label sits inside the group's box.
                    box = box_override or draw_boxes.get(layer_key)
                    if box is None:
                        return
                    # A layer with this name that isn't live type is a picture: shown as-is, never
                    # wiped, never drawn into. Only in files that have live type at all.
                    if (
                        box_override is None
                        and psd_path_for_size is not None
                        and type_layers_in_template
                        and layer_key not in type_layers_in_template
                        and layer_key in pixel_layers_in_template
                    ):
                        if (layer_key, "picture") not in pictures_noted:
                            pictures_noted.add((layer_key, "picture"))
                            background_notes.append(
                                f"{size_label(width, height)}: {layer_key} is a picture in this size's template "
                                "(rasterised, not live type), so it is used as it is -- copy or styling typed for "
                                "it on the form was not applied. Keep it as a type layer in Photoshop to retype "
                                "or restyle it here."
                            )
                        return
                    if layer_key in buried_in_template:
                        # Sitting under this size's background in the stack, so it is covered. A
                        # layer merely switched off is different: typed copy turns it back on.
                        return
                    if not text:
                        # Colour or font without retyping means restyle: the text is read back from
                        # the PSD, since a colour-only override used to do nothing. visible_only:
                        # styling a hidden layer redrew it everywhere and wiped the logo.
                        text = (
                            (get_psd_text_layers(psd_path_for_size, visible_only=True) or {}).get(layer_key)
                            if psd_path_for_size
                            else None
                        )
                        if not text:
                            return
                    # A text layer is redrawn only when its WORDS change: this renderer runs wider
                    # than Photoshop and dropped the last word out of the box.
                    restyled = bool(
                        font_family or font_size or use_custom_color or glow or show_background
                        or stroke_size or shadow
                    )
                    if (
                        box_override is None
                        and not restyled
                        and psd_path_for_size is not None
                        and _same_words(own_text_layers.get(layer_key) or "", text)
                        and (own_text_layers.get(layer_key) or "").strip()
                    ):
                        if (layer_key, "unchanged") not in pictures_noted:
                            pictures_noted.add((layer_key, "unchanged"))
                            background_notes.append(
                                f"{size_label(width, height)}: the {layer_key} already says this in the template, "
                                "so it was left exactly as Photoshop drew it."
                            )
                        return
                    # full_box: a text override replaces the box contents rather than printing over
                    # them. See _clean_layer_box() for why image layers don't need it.
                    if clean:
                        # Wipe the layer's box and the '(rendered)' companion box left by an earlier
                        # export, which carries last run's words at last run's position.
                        wipe = box
                        for extra in (layer_boxes.get(layer_key), layer_boxes.get(f"{layer_key} (rendered)")):
                            if extra:
                                wipe = (
                                    min(wipe[0], extra[0]), min(wipe[1], extra[1]),
                                    max(wipe[2], extra[2]), max(wipe[3], extra[3]),
                                )
                        # Plus the layer's effect reach: a Photoshop drop shadow or glow extends
                        # past the box and left a dark frame and a band under the redrawn header.
                        reach = get_psd_layer_effect_reach(psd_path_for_size, layer_key) if psd_path_for_size else 0
                        pad = int(math.ceil(reach * _template_scale(psd_canvas_size, (width, height), fit_mode))) + 2
                        wipe = (
                            max(0, wipe[0] - pad), max(0, wipe[1] - pad),
                            min(final_image.width, wipe[2] + pad), min(final_image.height, wipe[3] + pad),
                        )
                        _clean_layer_box(wipe, layer_key, full_box=True, restore_others=True)
                    psd_text_style = (
                        get_psd_layer_text_style(psd_path_for_size, layer_key)
                        if psd_path_for_size is not None
                        else None
                    ) or {}
                    # Boxes are in the template's pixels and were scaled by the fit above; the type
                    # has to scale with them or it renders small in a big box.
                    template_scale = _template_scale(psd_canvas_size, (width, height), fit_mode)
                    if psd_text_style and template_scale != 1.0:
                        for key in ("font_size", "line_height"):
                            if psd_text_style.get(key):
                                psd_text_style[key] = max(int(round(psd_text_style[key] * template_scale)), 1)
                    if not psd_text_style:
                        # On the results page: if the PSD clearly has this text layer, this points
                        # at the environment (psd-tools missing or old), not at sizing.
                        background_notes.append(
                            f"{size_label(width, height)}: {layer_key} -- couldn't read this "
                            "template's own font settings from the PSD (font/size/color/leading "
                            "not read from a real text layer) -- using autofit sizing instead."
                        )
                    effective_family = font_family or psd_text_style.get("family") or "sans"
                    effective_bold = psd_text_style.get("bold", True)
                    # The template's typeface when installed and the form picked nothing; otherwise
                    # the nearest bundled face, reported on the results page.
                    effective_font_name = None if font_family else psd_text_style.get("font_name")
                    if align == "template":
                        align = psd_text_style.get("align") or "left"
                    if effective_font_name and not font_covers_text(effective_font_name, text):
                        # Installed but missing the glyphs this copy needs: Apple Symbols has no
                        # accented Latin, so Spanish set in it rendered as boxes.
                        if (effective_font_name, "glyphs") not in missing_fonts_reported:
                            missing_fonts_reported.add((effective_font_name, "glyphs"))
                            background_warnings.append(
                                f"The template's {layer_key} is set in {effective_font_name}, which has no glyphs "
                                f"for some of the letters in this copy (accents, most likely), so the redrawn text "
                                f"uses a bundled {effective_family} face instead. Set the layer in a font with those "
                                "letters to match the design."
                            )
                        effective_font_name = None
                    if effective_font_name and find_font_file(effective_font_name) is None:
                        if effective_font_name not in missing_fonts_reported:
                            missing_fonts_reported.add(effective_font_name)
                            background_warnings.append(
                                f"The template's {layer_key} is set in {effective_font_name}, which isn't "
                                "installed on this computer, so the redrawn text uses a bundled "
                                f"{effective_family} face instead. Install the font (or activate it in "
                                "Creative Cloud) and run again to match the design."
                            )
                        effective_font_name = None
                    # A typed font size is the ceiling for shrink-to-fit, not a demand; the
                    # 'clamped' note says when text shrank further. Copy with the template's own
                    # line breaks keeps each line's size, unless the form restyles the layer.
                    text_lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
                    text_lines = [line for line in text_lines if line]
                    styled_lines = psd_text_style.get("lines") or []
                    # Glow, outline and colour from the form are drawn line by line; a font, a size
                    # or a background box forces one style across the whole box.
                    restyled = bool(font_family or font_size or show_background)
                    if (
                        len(text_lines) >= 2
                        and len(styled_lines) == len(text_lines)
                        and not restyled
                        and all(line.get("font_size") for line in styled_lines)
                    ):
                        lines_to_draw = [
                            dict(
                                style,
                                text=words,
                                font_name=(
                                    style.get("font_name")
                                    if find_font_file(style.get("font_name")) is not None
                                    and font_covers_text(style.get("font_name"), words)
                                    else None
                                ),
                            )
                            for style, words in zip(styled_lines, text_lines)
                        ]
                        line_shadow = None
                        if shadow:
                            # The form's drop shadow, in this size's pixels.
                            line_shadow = dict(shadow)
                            line_shadow["distance"] = line_shadow["distance"] * template_scale
                            line_shadow["size"] = line_shadow["size"] * template_scale
                        elif (psd_text_style.get("effects") or {}).get("shadow"):
                            line_shadow = dict(psd_text_style["effects"]["shadow"])
                            line_shadow["distance"] = line_shadow["distance"] * template_scale
                            line_shadow["size"] = line_shadow["size"] * template_scale
                        line_glow = (
                            {"color": glow_color, "size": glow_size, "opacity": glow_opacity} if glow else None
                        )
                        line_stroke = {"color": stroke_color, "size": stroke_size} if stroke_size else None
                        # The layer's glow and outline survive unless the form restates them.
                        # Previously any shadow or colour tick redrew the words and lost them.
                        design_px = psd_text_style.get("font_size") or 0
                        template_fx_lines = psd_text_style.get("effects") or {}
                        if line_glow is None and template_fx_lines.get("glow") and design_px:
                            line_glow = {
                                "color": tuple(template_fx_lines["glow"]["color"]),
                                "size": _template_glow_percent(template_fx_lines["glow"], template_scale, design_px),
                                "opacity": int(round(template_fx_lines["glow"]["opacity"])),
                            }
                        if line_stroke is None and template_fx_lines.get("stroke") and design_px:
                            line_stroke = {
                                "color": tuple(template_fx_lines["stroke"]["color"]),
                                "size": template_fx_lines["stroke"]["size"] * template_scale,
                            }
                        line_color = text_color if use_custom_color else None
                        text_debug = {}
                        final_image = apply_layer_styled_lines(
                            final_image, box, lines_to_draw, align=align, scale=template_scale, debug=text_debug,
                            shadow=line_shadow, glow=line_glow, stroke=line_stroke, color=line_color,
                        )
                        export_layer_patches[layer_key] = apply_layer_styled_lines(
                            Image.new("RGBA", final_image.size, (0, 0, 0, 0)), box, lines_to_draw,
                            align=align, scale=template_scale, keep_alpha=True, shadow=line_shadow,
                            glow=line_glow, stroke=line_stroke, color=line_color,
                        )
                        words_only_patches[layer_key] = export_layer_patches[layer_key]
                        applied_layers.append(layer_key)
                        if text_debug.get("font_size"):
                            rendered_font_sizes[layer_key] = text_debug["font_size"]
                        background_notes.append(
                            f"{size_label(width, height)}: {layer_key} drawn line by line at the template's own "
                            f"sizes -- {', '.join(f'{px}px' for px in text_debug.get('line_sizes', []))}, "
                            f"in {text_debug.get('font_used') or '?'} (the template asks for "
                            f"{psd_text_style.get('font_name') or '?'})."
                        )
                        return
                    exact_font_size = font_size
                    ceiling_font_size = None if exact_font_size else psd_text_style.get("font_size")
                    # The PSD's leading scales against its own font size to whatever renders, even
                    # with a typed size: relative spacing beats a generic 1.2x guess.
                    effective_leading = psd_text_style.get("line_height")
                    leading_reference_size = psd_text_style.get("font_size")
                    if use_custom_color:
                        effective_color = text_color
                    else:
                        effective_color = psd_text_style.get("color", (26, 26, 26))
                    # The layer's effects, scaled with the box, unless the form restyled the layer.
                    template_fx = psd_text_style.get("effects") or {}
                    fx_scale = template_scale
                    design_shadow = None
                    if shadow:
                        design_shadow = dict(shadow)
                        design_shadow["distance"] = design_shadow["distance"] * fx_scale
                        design_shadow["size"] = design_shadow["size"] * fx_scale
                    elif template_fx.get("shadow"):
                        design_shadow = dict(template_fx["shadow"])
                        design_shadow["distance"] = design_shadow["distance"] * fx_scale
                        design_shadow["size"] = design_shadow["size"] * fx_scale
                    design_size = psd_text_style.get("font_size") or 1
                    # Which of these effects are the TEMPLATE's own live layer styles
                    # rather than something typed on this form. The distinction only
                    # matters for the picture stored back inside a type layer: that
                    # layer still carries its style, so a raster with the style
                    # already drawn into it gets the style drawn again on the next
                    # run -- one more copy every time, until a soft drop shadow is a
                    # solid black slab. Effects the FORM asked for are not on the
                    # layer, so those do belong in the raster.
                    fx_from_template = {
                        "shadow": not shadow and bool(template_fx.get("shadow")),
                        "glow": bool(template_fx.get("glow")) and not glow and bool(design_size),
                        "stroke": bool(template_fx.get("stroke")) and not stroke_size and bool(design_size),
                    }
                    if template_fx.get("glow") and not glow and design_size:
                        glow = True
                        glow_color = tuple(template_fx["glow"]["color"])
                        glow_size = _template_glow_percent(template_fx["glow"], fx_scale, design_size)
                        glow_opacity = int(round(template_fx["glow"]["opacity"]))
                    if template_fx.get("stroke") and not stroke_size and design_size:
                        stroke_size = max(1, int(round(template_fx["stroke"]["size"] * fx_scale / design_size * 100)))
                        stroke_color = tuple(template_fx["stroke"]["color"])
                    # The design's own size and box, as Photoshop shows
                    # them, whenever no size was typed for the layer.
                    keep_design_size = not exact_font_size and bool(ceiling_font_size)
                    text_debug: dict = {}
                    final_image = apply_layer_text_override(
                        final_image,
                        box,
                        text,
                        text_color=effective_color,
                        font_family=effective_family,
                        font_name=effective_font_name,
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
                        background_blur=background_blur,
                        debug=text_debug,
                        stroke_size=stroke_size,
                        stroke_color=stroke_color,
                        keep_size=keep_design_size,
                        shadow=design_shadow,
                    )
                    # Same call onto a transparent canvas with keep_alpha=True: isolates the new
                    # glyphs as their own layer for the downloadable PSD.
                    def _patch(with_background, bare=False):
                        """`bare=True` leaves out the effects that came from the
                        template's own layer styles -- see fx_from_template."""
                        return apply_layer_text_override(
                            Image.new("RGBA", final_image.size, (0, 0, 0, 0)),
                            box,
                            text,
                            text_color=effective_color,
                            font_family=effective_family,
                            font_name=effective_font_name,
                            font_size=ceiling_font_size,
                            exact_font_size=exact_font_size,
                            bold=effective_bold,
                            leading=effective_leading,
                            leading_reference_size=leading_reference_size,
                            glow=False if (bare and fx_from_template["glow"]) else glow,
                            glow_color=glow_color,
                            glow_size=glow_size,
                            glow_opacity=glow_opacity,
                            align=align,
                            show_background=with_background,
                            background_color=background_color,
                            background_opacity=background_opacity,
                            background_blur=background_blur,
                            keep_alpha=True,
                            stroke_size=0 if (bare and fx_from_template["stroke"]) else stroke_size,
                            stroke_color=stroke_color,
                            keep_size=keep_design_size,
                            shadow=None if (bare and fx_from_template["shadow"]) else design_shadow,
                        )
                    export_layer_patches[layer_key] = _patch(show_background)
                    # The picture stored inside a type layer is the words alone: a stored background
                    # box travelled into later templates as an unremovable band, and a stored drop
                    # shadow compounded with the layer's own live style on every write-back.
                    _bare_needed = any(fx_from_template.values())
                    words_only_patches[layer_key] = (
                        _patch(False, bare=True) if (show_background or _bare_needed)
                        else export_layer_patches[layer_key]
                    )
                    applied_layers.append(layer_key)
                    # The size the words were actually laid out at: glow radius and stroke width are
                    # percentages of it. A type layer has no usable size of its own, and deriving
                    # one from the box overshot whenever the text box was taller than its words.
                    if text_debug.get("font_size"):
                        rendered_font_sizes[layer_key] = text_debug["font_size"]
                    background_notes.append(
                        f"{size_label(width, height)}: {layer_key} debug -- "
                        f"read from PSD: {psd_text_style or 'none'} | "
                        f"used: {text_debug.get('font_size')}px in {text_debug.get('font_used') or text_debug.get('family')} "
                        f"(the template asks for {psd_text_style.get('font_name') or '?'}; bold={text_debug.get('bold')}), "
                        f"leading {text_debug.get('line_height')}px, "
                        f"{text_debug.get('lines')} line(s), color {effective_color}."
                    )
                    if text_debug.get("clipped_lines"):
                        background_warnings.append(
                            f"{size_label(width, height)}: {layer_key} -- at the design's {text_debug.get('font_size')}px, "
                            f"{text_debug['clipped_lines']} line(s) of this copy don't fit the layer's box and are not "
                            "shown, exactly as Photoshop's text box would clip them. Shorten the copy, or type a "
                            "smaller font size for the layer to fit it all in."
                        )
                    if text_debug.get("clamped"):
                        # Text is always shrunk to fit the box; say when that used less than
                        # requested, or the font size setting looks like it did nothing.
                        background_notes.append(
                            f"{size_label(width, height)}: {layer_key} -- requested "
                            f"{text_debug.get('requested_font_size')}px didn't fit this size's "
                            f"box at this text length -- used {text_debug.get('font_size')}px "
                            "instead so the text stays inside the box."
                        )

                # Any of these on their own is a reason to redraw the
                # layer: new words, a colour, a family, a size.
                if text_only_size:
                    if size_text["header"]:
                        _apply_text_layer_override("header", size_text["header"], "", None, False, (0, 0, 0), align="template")
                elif (
                    size_text["header"]
                    or layer_header_glow
                    or layer_header_background
                    or layer_header_use_custom_color
                    or layer_header_font_family
                    or layer_header_font_size
                    or layer_header_stroke_size
                    or layer_header_shadow
                ):
                    _apply_text_layer_override(
                        "header",
                        size_text["header"],
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
                        background_blur=layer_header_background_blur,
                        stroke_size=layer_header_stroke_size,
                        stroke_color=layer_header_stroke_color,
                        shadow=layer_header_shadow,
                    )
                if text_only_size:
                    if size_text["description"]:
                        _apply_text_layer_override("description", size_text["description"], "", None, False, (0, 0, 0), align="template")
                elif (
                    size_text["description"]
                    or layer_description_glow
                    or layer_description_background
                    or layer_description_use_custom_color
                    or layer_description_font_family
                    or layer_description_font_size
                    or layer_description_stroke_size
                    or layer_description_shadow
                ):
                    _apply_text_layer_override(
                        "description",
                        size_text["description"],
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
                        background_blur=layer_description_background_blur,
                        stroke_size=layer_description_stroke_size,
                        stroke_color=layer_description_stroke_color,
                        shadow=layer_description_shadow,
                    )
                if text_only_size:
                    if size_text["legal"]:
                        _apply_text_layer_override("legal", size_text["legal"], "", None, False, (0, 0, 0), align="template")
                elif (
                    size_text["legal"]
                    or layer_legal_glow
                    or layer_legal_background
                    or layer_legal_use_custom_color
                    or layer_legal_font_family
                    or layer_legal_font_size
                    or layer_legal_stroke_size
                ):
                    _apply_text_layer_override(
                        "legal",
                        size_text["legal"],
                        layer_legal_font_family,
                        layer_legal_font_size,
                        layer_legal_use_custom_color,
                        layer_legal_text_color,
                        glow=layer_legal_glow,
                        glow_color=layer_legal_glow_color,
                        glow_size=layer_legal_glow_size,
                        glow_opacity=layer_legal_glow_opacity,
                        align=layer_legal_align,
                        show_background=layer_legal_background,
                        background_color=layer_legal_background_color,
                        background_opacity=layer_legal_background_opacity,
                        background_blur=layer_legal_background_blur,
                        stroke_size=layer_legal_stroke_size,
                        stroke_color=layer_legal_stroke_color,
                    )
                # Any CTA setting redraws the button, not just new words. Gating on the text field
                # made the colour, radius and stroke controls do nothing.
                if (
                    not text_only_size
                    and (
                        size_cta_text
                        or layer_cta_button_color != CTA_BUTTON_COLOR_DEFAULT
                        or layer_cta_glow
                        or layer_cta_stroke_size
                        or layer_cta_text_stroke_size
                        or layer_cta_radius is not None
                        # The family dropdown always submits something ("sans" by
                        # default), so on its own it is not a request to redraw.
                        or (layer_cta_font_family and layer_cta_font_family != "sans")
                        or layer_cta_font_size
                    )
                    and "cta" not in layer_image_overrides
                    and "cta" not in size_hidden_layer_names
                ):
                    # Skipped when a CTA image was uploaded for this run: that upload is the button,
                    # and drawing one over it buries what the user supplied.
                    cta_box = layer_boxes.get("cta")
                    restyling_button_flat = bool(
                        layer_cta_button_color != CTA_BUTTON_COLOR_DEFAULT
                        or layer_cta_glow
                        or layer_cta_stroke_size
                        or layer_cta_radius is not None
                    )
                    # A CTA group has its own label layer, so only the label is rewritten and the
                    # designed button survives. Flat CTA layers fall back to the pill.
                    cta_label_box = (
                        get_psd_group_text_box(psd_path_for_size, "cta")
                        if psd_path_for_size is not None
                        else None
                    )
                    if cta_label_box is not None and cta_box is not None:
                        cta_label_box = map_box_through_fit(
                            cta_label_box, psd_canvas_size, (width, height), fit_mode
                        ) if psd_canvas_size else cta_label_box
                        # Restyling the shape and rewriting only the words need opposite treatment
                        # of what is already on the canvas.
                        restyling_button = bool(
                            layer_cta_button_color != CTA_BUTTON_COLOR_DEFAULT
                            or layer_cta_glow
                            or layer_cta_stroke_size
                            or layer_cta_radius is not None
                        )
                        if restyling_button:
                            # Clear the designer's rectangle first or it shows around a smaller new
                            # pill. full_box=True restores what was behind the group.
                            _clean_layer_box(cta_box, "cta", full_box=True)
                        else:
                            # Label-only change: erase the old word against the button itself.
                            # _clean_layer_box() would punch a page-coloured hole through it.
                            final_image.paste(
                                _reconstruct_box_background(final_image, cta_label_box),
                                cta_label_box[:2],
                            )
                        # Redraw the rectangle only when CTA settings restyle it, through the same
                        # pill routine the flat-layer path uses.
                        if restyling_button:
                            final_image = apply_layer_cta_override(
                                final_image,
                                cta_box,
                                "",
                                button_color=layer_cta_button_color,
                                text_color=layer_cta_text_color,
                                glow=layer_cta_glow,
                                glow_color=layer_cta_glow_color,
                                glow_size=layer_cta_glow_size,
                                glow_opacity=layer_cta_glow_opacity,
                                border_size=layer_cta_stroke_size,
                                border_color=layer_cta_stroke_color,
                                corner_radius=layer_cta_radius,
                            )
                        # New words span the button rather than the old label's footprint: fitting a
                        # replacement to 'Click' comes out microscopic.
                        pad_x = int((cta_box[2] - cta_box[0]) * 0.08)
                        pad_y = int((cta_box[3] - cta_box[1]) * 0.18)
                        cta_label_box = (
                            cta_box[0] + pad_x,
                            cta_box[1] + pad_y,
                            cta_box[2] - pad_x,
                            cta_box[3] - pad_y,
                        )
                        # The group's own label when none was typed, since redrawing the rectangle
                        # covers it. get_psd_text_layers() can't see inside the group.
                        cta_label_text = size_cta_text or get_psd_group_text(
                            psd_path_for_size, "cta"
                        )
                        _apply_text_layer_override(
                            "cta",
                            cta_label_text,
                            layer_cta_font_family,
                            layer_cta_font_size,
                            True,
                            layer_cta_text_color,
                            # The glow belongs to the button, not the words on it; the label's
                            # stroke is its own field.
                            stroke_size=layer_cta_text_stroke_size,
                            stroke_color=layer_cta_text_stroke_color,
                            align="center",
                            box_override=cta_label_box,
                            clean=False,
                        )
                        # Take the CTA patch from the finished canvas: _apply_text_layer_override()
                        # left just the glyphs, so the PSD download got a label with no button under
                        # it. The label patch is kept for the type layer.
                        cta_label_patch = export_layer_patches.get("cta")
                        cta_patch = Image.new("RGBA", final_image.size, (0, 0, 0, 0))
                        cta_patch.paste(final_image.crop(cta_box).convert("RGBA"), cta_box[:2])
                        export_layer_patches["cta"] = cta_patch
                    elif cta_box is not None and (size_cta_text or restyling_button_flat):
                        # A flat (pixel) CTA has no label to read back, so it is redrawn only for
                        # new words or a restyled button, never for a font choice alone.
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
                            stroke_size=layer_cta_text_stroke_size,
                            stroke_color=layer_cta_text_stroke_color,
                            border_size=layer_cta_stroke_size,
                            border_color=layer_cta_stroke_color,
                            corner_radius=layer_cta_radius,
                        )
                        final_image = apply_layer_cta_override(
                            final_image, cta_box, size_cta_text or "", **cta_kwargs
                        )
                        export_layer_patches["cta"] = apply_layer_cta_override(
                            Image.new("RGBA", final_image.size, (0, 0, 0, 0)),
                            cta_box,
                            size_cta_text or "",
                            keep_alpha=True,
                            **cta_kwargs,
                        )
                        applied_layers.append("cta")

                if applied_layers:
                    background_notes.append(
                        f"{size_label(width, height)}: updated layer(s) -- " + ", ".join(applied_layers) + "."
                    )
                    # Rebuild a real layered PSD: untouched layers from the original
                    # (get_psd_layer_stack()), touched ones from the override's isolated RGBA patch,
                    # so the download matches the preview. Best-effort.
                    try:
                        # with_effects: a drop shadow / glow / stroke is a Photoshop
                        # layer style, not pixels, so a bare recomposite loses it
                        # while the preview -- built on the file's own flattened
                        # composite -- keeps it. Without this the header opened in
                        # the download with no shadow under it.
                        #
                        # Verified in Photoshop, not by psd-tools: psd-tools'
                        # composite() draws the resulting soft alpha as a solid
                        # slab, which is a limitation of that compositor and not of
                        # the file. Anything measuring this by psd-tools composite
                        # will disagree with what a designer actually opens.
                        layer_stack = (
                            get_psd_layer_stack(psd_path_for_size, with_effects=True)
                            if psd_path_for_size is not None else None
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
                        # Saved first: the rebuild below overwrites this template copy, the only
                        # file in this job with live text layers.
                        if psd_path_for_size is not None:
                            source_candidate_filename = (
                                f"{file_name_prefix}_{size_label(width, height)}_source-template.psd"
                            )
                            shutil.copy(psd_path_for_size, job_dir / source_candidate_filename)
                            source_psd_filename = source_candidate_filename
                            # Downloaded under the template's own name so it can be dropped back
                            # over it. Saved templates only; uploads keep the campaign name.
                            source_psd_download_name = (
                                psd_path_for_size.name
                                if psd_path_for_size.parent == templates_dir()
                                else None
                            )
                            # Re-apply the overrides against the template's own layer boxes; the
                            # rendered patches are in output-canvas space. Without this the editable
                            # download showed the stock backdrop, not the hero.
                            source_layer_images = {}
                            if size_image_overrides:
                                source_boxes = get_psd_layer_boxes(psd_path_for_size)
                                source_size = get_psd_canvas_size(psd_path_for_size) or (width, height)
                                for name, override_image in size_image_overrides.items():
                                    box = source_boxes.get(name)
                                    if name == "background" and (
                                        box is None
                                        or (box[2] - box[0]) * (box[3] - box[1]) < 0.01 * source_size[0] * source_size[1]
                                    ):
                                        # An emptied background layer: the hero fills the canvas
                                        # here as it does in the preview.
                                        box = (0, 0, source_size[0], source_size[1])
                                    if box is None:
                                        continue
                                    blank = Image.new("RGBA", source_size, (0, 0, 0, 0))
                                    if name == "background":
                                        # The same fit the preview used (see background_fit above);
                                        # a different fit is a different creative.
                                        source_layer_images[name] = apply_layer_background_override(
                                            blank, box, override_image, keep_alpha=True,
                                            fit="contain" if (upload_ai_allow_text or upload_hero_fit == "contain") else "crop",
                                        )
                                    else:
                                        source_layer_images[name] = apply_layer_image_override(
                                            blank, box, override_image, keep_alpha=True
                                        )
                            if source_layer_images:
                                swapped = replace_pixel_layers(
                                    job_dir / source_candidate_filename, source_layer_images
                                )
                                if swapped:
                                    background_notes.append(
                                        f"{size_label(width, height)}: source PSD's artwork replaced -- "
                                        + ", ".join(swapped)
                                        + "."
                                    )
                            # Carry the typed copy into the live text, or this file opens with the
                            # template's placeholder words. 'cta' names the group; the label inside
                            # it is rewritten. size_text carries localized copy.
                            live_text_updates = {
                                "header": size_text["header"],
                                "description": size_text["description"],
                                "legal": size_text["legal"],
                                "cta": size_text["cta"],
                            }
                            if any(live_text_updates.values()):
                                retyped = set_type_layer_text(
                                    job_dir / source_candidate_filename, live_text_updates
                                )
                                if retyped:
                                    background_notes.append(
                                        f"{size_label(width, height)}: source PSD's live text updated -- "
                                        + ", ".join(retyped)
                                        + "."
                                    )
                            # The button's properties into the live shape, so it opens as asked and
                            # stays an editable shape. Only when something was actually set.
                            cta_shape_style = {}
                            if layer_cta_button_color != CTA_BUTTON_COLOR_DEFAULT:
                                cta_shape_style["fill"] = layer_cta_button_color
                            if layer_cta_stroke_size:
                                cta_shape_style["stroke_width_pct"] = layer_cta_stroke_size
                                cta_shape_style["stroke_color"] = layer_cta_stroke_color
                            if layer_cta_radius is not None:
                                # Rounds the path Photoshop draws from, not just the number in the
                                # Properties panel.
                                cta_shape_style["corner_radius_pct"] = layer_cta_radius
                            if cta_shape_style:
                                restyled = set_shape_layer_style(
                                    job_dir / source_candidate_filename,
                                    {"cta": cta_shape_style},
                                )
                                if restyled:
                                    background_notes.append(
                                        f"{size_label(width, height)}: source PSD's CTA shape restyled -- "
                                        + ", ".join(restyled)
                                        + " (still an editable shape)."
                                    )
                            live_text_colors = {}
                            if layer_header_use_custom_color:
                                live_text_colors["header"] = layer_header_text_color
                            if layer_description_use_custom_color:
                                live_text_colors["description"] = layer_description_text_color
                            if layer_legal_use_custom_color:
                                live_text_colors["legal"] = layer_legal_text_color
                            if layer_cta_text_color != CTA_TEXT_COLOR_DEFAULT:
                                live_text_colors["cta"] = layer_cta_text_color
                            # Glow and stroke. Colour and size live inside the type layer, but a
                            # glow is a layer EFFECT, so live text arrived as flat words with no
                            # halo. Written only here; the layered PSD stays pixels throughout.
                            live_text_effects = {}
                            for key, on, colour, size, opacity, s_size, s_colour in (
                                ("header", layer_header_glow, layer_header_glow_color,
                                 layer_header_glow_size, layer_header_glow_opacity,
                                 layer_header_stroke_size, layer_header_stroke_color),
                                ("description", layer_description_glow, layer_description_glow_color,
                                 layer_description_glow_size, layer_description_glow_opacity,
                                 layer_description_stroke_size, layer_description_stroke_color),
                                ("legal", layer_legal_glow, layer_legal_glow_color,
                                 layer_legal_glow_size, layer_legal_glow_opacity,
                                 layer_legal_stroke_size, layer_legal_stroke_color),
                                ("cta", layer_cta_glow, layer_cta_glow_color,
                                 layer_cta_glow_size, layer_cta_glow_opacity,
                                 layer_cta_text_stroke_size, layer_cta_text_stroke_color),
                            ):
                                spec = {}
                                # Measured against the size this layer's words were drawn at; with
                                # none (a layer this run didn't retype) the effect is skipped.
                                laid_out_at = rendered_font_sizes.get(key)
                                if not laid_out_at:
                                    continue
                                if on and size and opacity:
                                    # Fitting the renderer's halo to Photoshop's Outer Glow gives
                                    # Size 2.7r, Spread 45%; Size alone at r was a faint haze.
                                    halo = max(1.0, laid_out_at * (size / 100.0))
                                    spec["glow"] = {
                                        "color": colour,
                                        "radius": max(1.0, 2.7 * halo),
                                        "spread": 45,
                                        "opacity": opacity,
                                    }
                                if s_size:
                                    spec["stroke"] = {
                                        "color": s_colour,
                                        "size": max(1.0, laid_out_at * (s_size / 100.0)),
                                    }
                                form_shadow = {"header": layer_header_shadow, "description": layer_description_shadow}.get(key)
                                if form_shadow:
                                    # In the PSD's own pixels, as the
                                    # form's numbers are (see fx_scale).
                                    spec["shadow"] = dict(form_shadow)
                                if spec:
                                    live_text_effects[key] = spec
                            # ...and the picture layers' glow and shadow,
                            # already in the PSD's pixels.
                            for fx_key, fx_spec in picture_fx_applied.items():
                                spec = {}
                                if fx_spec.get("glow"):
                                    # _layer_effect_images grows the picture by size/2 and blurs by
                                    # size/2: Photoshop Size 1.5x with a third of it solid.
                                    spec["glow"] = {
                                        "color": fx_spec["glow"]["color"],
                                        "radius": max(1.0, 1.5 * float(fx_spec["glow"]["size"])),
                                        "spread": 33,
                                        "opacity": fx_spec["glow"]["opacity"],
                                    }
                                if fx_spec.get("shadow"):
                                    spec["shadow"] = dict(fx_spec["shadow"])
                                if fx_spec.get("stroke"):
                                    spec["stroke"] = dict(fx_spec["stroke"])
                                if spec:
                                    live_text_effects[fx_key] = spec
                            if live_text_effects:
                                fx = set_type_layer_effects(
                                    job_dir / source_candidate_filename, live_text_effects
                                )
                                if fx:
                                    background_notes.append(
                                        f"{size_label(width, height)}: live-text PSD's glow/stroke "
                                        "applied as Photoshop layer effects -- " + ", ".join(fx) + "."
                                    )

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
                            # A typed font size becomes the type layer's size, converted from
                            # rendered_font_sizes into the template canvas's pixels.
                            live_text_sizes = {}
                            size_to_psd = 1.0 / (_template_scale(psd_canvas_size, (width, height), fit_mode) or 1.0)
                            for key, typed in (
                                ("header", layer_header_font_size), ("description", layer_description_font_size),
                                ("legal", layer_legal_font_size), ("cta", layer_cta_font_size),
                            ):
                                if typed and rendered_font_sizes.get(key):
                                    live_text_sizes[key] = rendered_font_sizes[key] * size_to_psd
                            if live_text_sizes:
                                resized = set_type_layer_font_size(
                                    job_dir / source_candidate_filename, live_text_sizes
                                )
                                if resized:
                                    background_notes.append(
                                        f"{size_label(width, height)}: source PSD's live text resized -- "
                                        + ", ".join(resized)
                                        + "."
                                    )
                            # Rebuild the flattened composite last, or Finder and Preview show the
                            # PSD untouched. Type layers show; the renderer's pixels go in hidden as
                            # '<name> (rendered)' if Photoshop won't recompose.
                            live_text_rasters = {}
                            for key in ("header", "description", "legal"):
                                patch = words_only_patches.get(key, export_layer_patches.get(key))
                                if patch is not None:
                                    live_text_rasters[key] = patch
                            if cta_label_patch is not None:
                                live_text_rasters["cta"] = cta_label_patch
                            if live_text_rasters:
                                redrawn = pair_type_layers_with_pixels(
                                    job_dir / source_candidate_filename,
                                    live_text_rasters,
                                    prefer="text",
                                )
                                # Replace the type layer's raster: Photoshop displays that, so a
                                # rewritten string still opened reading the old placeholder.
                                set_type_layer_raster(
                                    job_dir / source_candidate_filename, live_text_rasters
                                )
                                if redrawn:
                                    background_notes.append(
                                        f"{size_label(width, height)}: live-text PSD keeps "
                                        + ", ".join(redrawn)
                                        + " as editable type layers, with the rendered version "
                                        "beside each as \"(rendered)\", switched off."
                                    )

                            # Layers hidden on the form open switched off here too; nothing is
                            # deleted, the eye is off.
                            if size_hidden_layer_names:
                                set_layer_visibility(
                                    job_dir / source_candidate_filename,
                                    {name: False for name in size_hidden_layer_names},
                                )
                            set_flattened_preview(
                                job_dir / source_candidate_filename, final_image
                            )

                            # Optionally write back into the saved template, since everything above
                            # edits a copy. Words only and only as live type, or the text could
                            # never be retyped again. default_templates/ only.
                            if (
                                update_saved_templates
                                and psd_path_for_size is not None
                                and psd_path_for_size.parent == templates_dir()
                            ):
                                template_updates = dict(typed_copy_english)
                                # Words go into the template only with a picture of those same
                                # words: a French run saved English words with a French picture.
                                held_back = [
                                    key for key, words in template_updates.items()
                                    if words and not _same_words(words, size_text.get(key) or words)
                                ]
                                for key in held_back:
                                    template_updates[key] = ""
                                if held_back:
                                    background_notes.append(
                                        f"{size_label(width, height)}: the typed {', '.join(held_back)} was not "
                                        f"saved into {psd_path_for_size.name} on a "
                                        f"{COPY_LANGUAGE_ENGLISH_NAMES.get(copy_language, copy_language)} run -- "
                                        "the template keeps English words with an English picture; run in "
                                        "English to save the copy into it."
                                    )
                                # Gate on anything to carry, not just words: a CTA restyled with
                                # nothing typed was silently dropped.
                                if (
                                    any(template_updates.values())
                                    or live_text_colors
                                    or live_text_sizes
                                    or live_text_effects
                                    or cta_shape_style
                                ):
                                    try:
                                        TEMPLATE_BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
                                        import datetime as _dt
                                        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                                        backup = (
                                            TEMPLATE_BACKUPS_DIR
                                            / f"{psd_path_for_size.stem}.{stamp}{psd_path_for_size.suffix}"
                                        )
                                        if not backup.exists():
                                            shutil.copy(psd_path_for_size, backup)
                                        retyped_template = set_type_layer_text(
                                            psd_path_for_size, template_updates
                                        )
                                        # Styling too: colour inside the type layer, glow and stroke
                                        # as layer effects, so runs needn't restate it.
                                        if live_text_colors:
                                            set_type_layer_colors(
                                                psd_path_for_size, live_text_colors
                                            )
                                        if live_text_sizes:
                                            set_type_layer_font_size(
                                                psd_path_for_size, live_text_sizes
                                            )
                                        if live_text_effects:
                                            set_type_layer_effects(
                                                psd_path_for_size, live_text_effects
                                            )
                                        # The button's shape too. Written to the copy and not the
                                        # template, a restyled CTA came back blue next run.
                                        if cta_shape_style:
                                            restyled_template = set_shape_layer_style(
                                                psd_path_for_size, {"cta": cta_shape_style}
                                            )
                                            retyped_template = (
                                                retyped_template + restyled_template
                                            )
                                        # Type layer rasters, or the template opens reading its old
                                        # placeholder. Only of the words it now holds.
                                        template_rasters = {}
                                        for raster_key, raster in live_text_rasters.items():
                                            holds = template_updates.get(raster_key) or own_text_layers.get(raster_key)
                                            shows = size_text.get(raster_key) or own_text_layers.get(raster_key)
                                            if holds and shows and _same_words(holds, shows):
                                                template_rasters[raster_key] = raster
                                        if template_rasters:
                                            set_type_layer_raster(
                                                psd_path_for_size, template_rasters
                                            )
                                        # Rebuild the flattened snapshot: Pillow reads it, not the
                                        # layers, so old placeholder text bled into later runs.
                                        refresh_flattened_preview(psd_path_for_size)
                                    except Exception:  # noqa: BLE001
                                        retyped_template = []
                                    if retyped_template:
                                        background_notes.append(
                                            f"{size_label(width, height)}: saved template "
                                            f"{psd_path_for_size.name} updated -- "
                                            + ", ".join(retyped_template)
                                            + f". The version it replaced is in _template_backups/{backup.name}."
                                        )
                        psd_candidate_filename = f"{file_name_prefix}_{size_label(width, height)}.psd"
                        # Live type can only be inherited: psd-tools authors pixel layers only. This
                        # download is the render's pixels, matching the preview; the source PSD
                        # beside it is editable. Hidden layers stay off unless redrawn here.
                        hidden_in_template = {
                            name
                            for name in (
                                psd_hidden_layer_names(psd_path_for_size)
                                if psd_path_for_size is not None else set()
                            ) | size_hidden_layer_names
                            if name not in export_layer_patches
                        }
                        save_layered_psd(
                            export_layers,
                            (width, height),
                            job_dir / psd_candidate_filename,
                            layer_names={},
                            hidden=hidden_in_template,
                        )
                        psd_filename = psd_candidate_filename
                    except Exception:
                        pass
        else:
            # Built once and shared by render_creative() and render_creative_layers() so the PNG
            # preview and the layered PSD can't drift apart.
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

            # Best-effort: a PSD write failure leaves psd_filename None and no PSD link, rather than
            # failing a render that already succeeded.
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
                "source_psd_download_name": source_psd_download_name,
                "whole_ad": bool(upload_ai_full_ad and (width, height) in full_ad_templates),
            }
        )

    carried_approvals = {}
    for width, height in kept_sizes:
        label = size_label(width, height)
        copied = {}
        for prior_file in sorted(prior_job_dir.glob(f"*_{label}*")):
            if prior_file.suffix.lower() not in (".png", ".psd"):
                continue
            try:
                shutil.copy2(prior_file, job_dir / prior_file.name)
            except OSError:
                continue
            stem = prior_file.stem
            if prior_file.suffix.lower() == ".png":
                copied["png"] = prior_file.name
            elif stem.endswith("_source-template"):
                copied["source_psd"] = prior_file.name
            elif stem.endswith(f"_{label}"):
                copied["psd"] = prior_file.name
        if "png" not in copied:
            background_warnings.append(
                f"{label} was approved in the previous run but its file couldn't be found to carry over."
            )
            continue
        creatives.append(
            {
                "filename": copied["png"],
                "label": label,
                "ratio": ratio_label(width, height),
                "name": size_name(width, height),
                "psd_filename": copied.get("psd"),
                "source_psd_filename": copied.get("source_psd"),
                "whole_ad": bool((prior_form_state.get("fields") or {}).get("upload_ai_full_ad")),
                "kept": True,
            }
        )
        carried_approvals[label] = {
            "approved": True,
            "at": _datetime.datetime.now().isoformat(timespec="seconds"),
            "carried_from": prior_job_dir.name,
        }
        background_notes.append(
            f"{label}: kept exactly as approved in run {prior_job_dir.name[:6]} -- not regenerated."
        )
    if kept_sizes:
        order = {size_label(w, h): i for i, (w, h) in enumerate(display_sizes)}
        creatives.sort(key=lambda c: order.get(c["label"], len(order)))
        try:
            _approvals_path(job_id).write_text(json.dumps(carried_approvals, indent=2))
        except OSError:
            pass

    zip_stem = (
        f"{product_name_slug}_{campaign_label}_creatives"
        if product_name_slug
        else f"{campaign_label}_creatives"
    )
    zip_path = job_dir / f"{zip_stem}.zip"
    _report_progress(progress_token, 94, "Packaging the download")
    # Zip entries nest as <Campaign Name>/<Product Name>/ so several campaigns from one session can
    # be unzipped side by side without their same-named sizes colliding.
    zip_entry_prefix = "/".join(_download_folder_parts(product_name, campaign_name, campaign_label)) + "/"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for creative in creatives:
            zf.write(
                job_dir / creative["filename"],
                arcname=f"{zip_entry_prefix}{creative['filename']}",
            )
            if creative.get("psd_filename"):
                zf.write(
                    job_dir / creative["psd_filename"],
                    arcname=f"{zip_entry_prefix}{creative['psd_filename']}",
                )
            # The source template: the only copy whose header and description are still editable
            # type layers.
            if creative.get("source_psd_filename"):
                zf.write(
                    job_dir / creative["source_psd_filename"],
                    arcname=f"{zip_entry_prefix}{creative.get('source_psd_download_name') or creative['source_psd_filename']}",
                )

    # Also copy the zip and the files themselves into downloads/<Campaign>/<Product>/, so finished
    # ads are findable without unzipping. Re-runs overwrite; best-effort, never fails a render.
    try:
        DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
        download_dir = DOWNLOADS_DIR.joinpath(*_download_folder_parts(product_name, campaign_name, campaign_label))
        download_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(zip_path, download_dir / zip_path.name)
        for creative in creatives:
            for key in ("filename", "psd_filename", "source_psd_filename"):
                name = creative.get(key)
                if not name or not (job_dir / name).is_file():
                    continue
                out_name = creative.get("source_psd_download_name") or name if key == "source_psd_filename" else name
                shutil.copy2(job_dir / name, download_dir / out_name)
    except OSError:
        pass

    # Saved so /edit/<job_id> can reload this form pre-filled and carry forward uploads that aren't
    # re-supplied (see _carry_forward_upload()). Best-effort.
    form_state_fields = {name: (request.form.get(name) or "") for name in EDIT_TEXT_FIELD_NAMES}
    form_state_fields.update(form_field_overrides)
    # As applied, not as typed: the form's select needs the normalised value to reselect.
    form_state_fields["upload_ai_speed"] = upload_ai_speed
    # Prefer the validated/defaulted Python variable over the raw form field, so a radio group's
    # default round-trips as a checked option instead of landing on ''.
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
        "upload_hero_image": upload_hero_path,
        **{slot: (upload_ai_reference_paths[i] if i < len(upload_ai_reference_paths) else None) for i, slot in enumerate(REFERENCE_SLOTS)},
        # Under a key of its own, never a user-upload key -- see the
        # "Keep this image" branch in the generator block above.
        "upload_ai_generated": upload_ai_path,
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
    # The campaign this run used goes back into briefs/, so a campaign
    # typed into the form is offered by the brief picker next time
    # instead of existing only in this job's saved fields.
    brief_note = record_run_in_briefs(form_state_fields)
    if brief_note:
        background_notes.append(brief_note)

    # Which multi-campaign page and card this job came from, for grouping on Edit. A missing or
    # blank session_id just means this job isn't grouped, never an error.
    session_id = (request.form.get("session_id") or "").strip() or uuid.uuid4().hex

    try:
        (job_dir / "form_state.json").write_text(
            json.dumps(
                {
                    "fields": form_state_fields,
                    "files": form_state_files,
                    "promoted_psd_rows": sorted(promoted_psd_rows),
                    "session_id": session_id,
                    "campaign_slot": campaign_slot,
                },
                indent=2,
            )
        )
    except OSError:
        pass
    # ...and as the run a fresh form carries its section files from.
    try:
        prefs = _load_preferences()
        prefs["last_job_id"] = job_id
        key = _product_memory_key(request.form.get("product_name"), request.form.get("campaign_name"))
        if key:
            products = _product_memories(prefs)
            entry = dict(products.get(key) or {})
            entry["last_job_id"] = job_id
            products[key] = entry
            prefs["products"] = products
        _save_preferences(prefs)
    except OSError:
        pass

    # Best-effort: record this job in its session's {slot -> job_id} index so a later Edit on any
    # campaign generated alongside it can bring all of them back.
    if prior_job_dir is not None and prior_job_dir.name.startswith("draft_"):
        # The draft this run was reopened from has done its job.
        shutil.rmtree(prior_job_dir, ignore_errors=True)

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

    # Write notes and warnings next to the job output: they exist only on the rendered page, so a
    # provider failure vanishes when the tab closes.
    spend_note = _spend_note(spend)
    if spend_note:
        background_notes.append(spend_note)

    # Job folders are pruned on a timer, so without this every AI-generated creative is temporary.
    if upload_ai_enabled and upload_ai_keep:
        background_notes.append(
            "Kept image reused, so nothing new was added to image_library/ -- the picture from the run "
            "it came from is already there."
        )
    elif upload_ai_enabled:
        kept = save_to_image_library(
            job_dir,
            creatives,
            {
                "stamp": _datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                "slug": product_name_slug or "creative",
                "job_id": job_id,
                "generated_at": _datetime.datetime.now().isoformat(timespec="seconds"),
                "product": product_name,
                "campaign": campaign_name,
                "market": market,
                "provider": upload_ai_provider,
                "prompt": upload_ai_prompt_used,
                "prompt_typed": upload_ai_prompt,
                "copy_language": copy_language,
            },
            backdrop=upload_ai_path,
        )
        if kept:
            background_notes.append(
                f"Kept in image_library/ for later training: {', '.join(kept)}."
            )
    try:
        (job_dir / "run_report.json").write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "generated_at": _datetime.datetime.now().isoformat(timespec="seconds"),
                    "notes": background_notes,
                    "warnings": background_warnings,
                },
                indent=2,
            )
        )
    except OSError:
        # Never fail a finished batch over its own logging.
        pass

    approvals = load_approvals(job_id)
    videos = load_videos(job_id)
    for creative in creatives:
        creative["approved"] = bool(approvals.get(creative["label"], {}).get("approved"))
        entry = videos.get(creative["label"])
        creative["video_filename"] = entry["filename"] if entry and (job_dir / entry["filename"]).is_file() else None

    _report_progress(progress_token, 100, "Done")
    return render_template(
        "result.html",
        job_id=job_id,
        creatives=creatives,
        spend_note=spend_note,
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


UPLOAD_THUMB_EDGE = 160
_upload_thumb_cache: dict = {}


@app.route("/upload-thumb/<job_id>/<filename>")
def serve_upload_thumb(job_id, filename):
    """A small JPEG of one job upload, whatever its format.

    The size-specific template rows keep .psd files, which no browser can
    draw, so their "kept from your last run" chip can't point at the raw
    file the way the hero chip does. This renders the same flat picture
    the run would use (open_as_rgb: the PSD's stored preview) at thumbnail
    size, so the chip shows the actual template rather than a broken
    image icon. Videos get their first frame for the same reason.
    """
    job_id = secure_filename(job_id)
    filename = secure_filename(filename)
    file_path = JOBS_DIR / job_id / "uploads" / filename
    if not file_path.is_file():
        abort(404)
    try:
        stat = file_path.stat()
        key = (str(file_path), stat.st_size, stat.st_mtime_ns)
        payload = _upload_thumb_cache.get(key)
        if payload is None:
            image = open_as_rgb(file_path)
            image.thumbnail((UPLOAD_THUMB_EDGE, UPLOAD_THUMB_EDGE))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            payload = buffer.getvalue()
            if len(_upload_thumb_cache) > 200:
                _upload_thumb_cache.clear()
            _upload_thumb_cache[key] = payload
    except Exception:  # noqa: BLE001
        abort(404)
    return send_file(io.BytesIO(payload), mimetype="image/jpeg", max_age=0)


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


def _approvals_path(job_id: str) -> Path:
    return JOBS_DIR / secure_filename(job_id) / "approvals.json"


def load_approvals(job_id: str) -> dict:
    """{size label: {"approved": bool, "at": iso time}} for a run --
    what was ticked in the preview. Empty when nothing has been."""
    try:
        data = json.loads(_approvals_path(job_id).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


@app.route("/approve/<job_id>", methods=["POST"])
def approve(job_id):
    """Tick or untick one size as approved. Called by the preview's
    checkbox; the state lives in the run's folder so it survives the
    page, and the approved set can be downloaded on its own."""
    job_id = secure_filename(job_id)
    job_dir = JOBS_DIR / job_id
    if not job_dir.is_dir():
        abort(404)
    payload = request.get_json(silent=True) or request.form
    label = (payload.get("label") or "").strip()
    if not label or not re.fullmatch(r"\d{2,5}x\d{2,5}", label):
        abort(400)
    approved = str(payload.get("approved", "")).lower() in ("1", "true", "yes", "on")
    approvals = load_approvals(job_id)
    if approved:
        approvals[label] = {"approved": True, "at": _datetime.datetime.now().isoformat(timespec="seconds")}
    else:
        approvals.pop(label, None)
    try:
        _approvals_path(job_id).write_text(json.dumps(approvals, indent=2))
    except OSError:
        abort(500)
    return {"label": label, "approved": approved, "approved_count": len(approvals)}


def _videos_path(job_id: str) -> Path:
    return JOBS_DIR / secure_filename(job_id) / "videos.json"


def load_videos(job_id: str) -> dict:
    """{size label: {"filename": ..., "seconds": ..., "at": ...}} -- the
    motion versions made for a run from the preview."""
    try:
        data = json.loads(_videos_path(job_id).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


@app.route("/motion/<job_id>", methods=["POST"])
def motion(job_id):
    """Make (or drop) the motion version of one size -- the preview's
    "Make a video of this size" box. Rendered from the size's layered
    PSD by src/motion.py: a looping MP4, no model, no spend."""
    from src.motion import DEFAULT_DURATION, render_motion_clip

    job_id = secure_filename(job_id)
    job_dir = JOBS_DIR / job_id
    if not job_dir.is_dir():
        abort(404)
    payload = request.get_json(silent=True) or request.form
    label = (payload.get("label") or "").strip()
    if not re.fullmatch(r"\d{2,5}x\d{2,5}", label):
        abort(400)
    make = str(payload.get("make", "1")).lower() in ("1", "true", "yes", "on")
    videos = load_videos(job_id)
    if not make:
        entry = videos.pop(label, None)
        if entry:
            try:
                (job_dir / entry["filename"]).unlink()
            except OSError:
                pass
        _videos_path(job_id).write_text(json.dumps(videos, indent=2))
        return {"label": label, "video": None}
    png = next(iter(sorted(job_dir.glob(f"*_{label}.png"))), None)
    if png is None:
        abort(404)
    # The layered PSD carries the stack; the flat PNG is the fallback
    # (backdrop drift only) for a size that has no layers.
    psd = job_dir / f"{png.stem}.psd"
    out = job_dir / f"{png.stem}.mp4"
    # Layers the template had switched off don't animate, even for a per-size PSD written before the
    # app wrote them off: the source-template copy beside it still records which.
    source_template = job_dir / f"{png.stem}_source-template.psd"
    hidden = psd_hidden_layer_names(source_template) if source_template.is_file() else set()
    # ...and the layers the run's own hide boxes took out.
    try:
        fields = (json.loads((job_dir / "form_state.json").read_text()).get("fields") or {})
    except Exception:  # noqa: BLE001
        fields = {}
    # ...unless the run used its templates as uploaded, where the hide
    # boxes don't apply (see size_hidden_layer_names in generate()).
    if not (fields.get("psd_as_is") and not fields.get("upload_ai_enabled")):
        hidden |= {name for name in HIDEABLE_LAYER_NAMES if fields.get(f"layer_{name}_hidden")}
    try:
        info = render_motion_clip(
            psd if psd.is_file() else None, out, fallback_image=png, hidden=hidden
        )
    except RuntimeError as exc:
        return {"error": str(exc)}, 500
    videos[label] = {
        "filename": out.name,
        "seconds": info["seconds"],
        "layers": info["layers"],
        "at": _datetime.datetime.now().isoformat(timespec="seconds"),
    }
    _videos_path(job_id).write_text(json.dumps(videos, indent=2))
    return {
        "label": label,
        "video": {
            "url": url_for("serve_output", job_id=job_id, filename=out.name),
            "download": url_for("download_video", job_id=job_id, filename=out.name),
            "seconds": info["seconds"],
            "layers": info["layers"],
        },
    }


@app.route("/download-video/<job_id>/<filename>")
def download_video(job_id, filename):
    job_id = secure_filename(job_id)
    filename = secure_filename(filename)
    if not filename.lower().endswith(".mp4"):
        abort(404)
    file_path = JOBS_DIR / job_id / filename
    if not file_path.is_file():
        abort(404)
    stamped = f"{file_path.stem}_{job_id[:6]}{file_path.suffix}"
    return send_file(file_path, as_attachment=True, download_name=stamped, mimetype="video/mp4")


@app.route("/download/<job_id>/approved")
def download_approved(job_id):
    """A zip of only the sizes ticked as approved -- the PNG and any
    PSDs for each -- plus approvals.json saying who approved what when."""
    job_id = secure_filename(job_id)
    job_dir = JOBS_DIR / job_id
    approvals = load_approvals(job_id)
    if not job_dir.is_dir() or not approvals:
        abort(404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(job_dir.iterdir()):
            if path.suffix.lower() not in (".png", ".psd", ".mp4"):
                continue
            if any(f"_{label}" in path.stem for label in approvals):
                zf.write(path, path.name)
        zf.writestr("approvals.json", json.dumps(approvals, indent=2))
    buf.seek(0)
    stem = next((z.stem for z in job_dir.glob("*.zip")), f"{job_id[:6]}_creatives")
    return send_file(buf, as_attachment=True, download_name=f"{stem}_approved.zip", mimetype="application/zip")


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
    # Restricted to .psd: serve_output() above already serves any file in a job folder, so this is
    # just a download-forced, download_name'd entry point, not a wider attack surface.
    job_id = secure_filename(job_id)
    filename = secure_filename(filename)
    if not filename.lower().endswith(".psd"):
        abort(404)
    file_path = JOBS_DIR / job_id / filename
    if not file_path.is_file():
        abort(404)
    # Stamp the filename with six characters of the job id. Every run writes the same name, so a
    # second download lands as '... (2).psd' and an older run gets stared at as if nothing applied.
    stamped = f"{file_path.stem}_{job_id[:6]}{file_path.suffix}"
    # A live-text file goes out under its template's own name when the results page asks
    # (?as=tester-720x480.psd), unstamped, so it can be dropped straight back over that template.
    wanted = secure_filename(request.args.get("as") or "")
    if wanted and wanted.lower().endswith(".psd") and SIZE_IN_NAME_RE_LOOSE.search(wanted):
        stamped = wanted
    return send_file(file_path, as_attachment=True, download_name=stamped)


def _free_port(preferred: int) -> int:
    """The preferred port if nothing is listening on it, else the first
    free one from the same fallback list run.sh / run.ps1 use. On a Mac
    port 5000 is usually AirPlay's, and a packaged build has no shell
    script in front of it to notice."""
    import socket

    for candidate in (preferred, 5050, 8080, 8000, 8765):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return candidate
    return preferred


def _open_browser_when_up(url: str) -> None:
    """Open the default browser once the server answers, from a thread
    so the server itself is never held up."""
    import threading
    import time
    import urllib.request
    import webbrowser

    def wait_then_open():
        for _ in range(40):
            time.sleep(0.5)
            try:
                urllib.request.urlopen(url, timeout=2).close()
            except Exception:
                continue
            webbrowser.open(url)
            return

    threading.Thread(target=wait_then_open, daemon=True).start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Every product in briefs/ gets its templates folder now, unpacked
    # from the backup zip, so they are there before the first run.
    try:
        for folder in seed_product_template_folders():
            print(f"[webapp] made {folder.relative_to(BASE_DIR)}/ from the backup zip")
        _seeded_product_folders = True
    except Exception as exc:  # noqa: BLE001
        print(f"[webapp] could not seed product template folders: {exc}")
    if FROZEN:
        # A packaged build is double-clicked, not launched from a shell: pick a free port and open
        # it. The reloader restarts by re-running a script path a frozen build has no.
        port = _free_port(port)
        url = f"http://127.0.0.1:{port}"
        print(f"Creative Automation Pipeline -> {url}   (close this window to stop)")
        print(f"Files live beside the app in: {BASE_DIR}")
        _open_browser_when_up(url)
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
        raise SystemExit(0)
    # Auto-reload on by default for this local dev tool: otherwise an edited template or module
    # keeps serving old behaviour with no visible clue. FLASK_RELOAD=0 off; FLASK_DEBUG=1 also
    # enables the interactive debugger, which is a different and less safe thing.
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(
        host="127.0.0.1",
        port=port,
        debug=debug,
        use_reloader=debug or os.environ.get("FLASK_RELOAD", "1") != "0",
    )
