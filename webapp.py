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
import os
import random
import sys
import time
from pathlib import Path

# Packaged builds (windows/build_exe.ps1, PyInstaller) run from a folder
# that holds the exe, and everything the user owns or the app writes --
# .env, default_templates/, outputs/, downloads/ -- lives beside the exe
# where they can see it. The read-only files that ship inside the bundle
# (templates/, fonts/) live in PyInstaller's extraction dir instead.
FROZEN = bool(getattr(sys, "frozen", False))
BASE_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", BASE_DIR))

try:  # optional, same as src/main.py -- a missing python-dotenv isn't fatal
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None
else:
    # The CLI has always read .env; the web app never did, so a token or
    # model set there was silently ignored and "HUGGINGFACE_API_TOKEN is
    # not set" came back from a form that had it configured all along.
    # The path is explicit because dotenv's own search starts from this
    # file, which inside a packaged build is not where the user's .env is.
    load_dotenv(BASE_DIR / ".env")
import re
import secrets
import shutil
import uuid
import zipfile

from flask import Flask, abort, flash, redirect, render_template, request, send_file, url_for
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
from src.localization import localize_message
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

# Sane bounds for a user-supplied font size, in pixels -- just a safety
# valve against nonsense input (0, negative, absurdly huge); the autofit
# path (font size left blank) isn't bound by this at all.
MIN_CUSTOM_FONT_SIZE = 4
MAX_CUSTOM_FONT_SIZE = 2000

DEFAULT_LOGO_SCALE_PERCENT = 16
DEFAULT_LOGO_OPACITY_PERCENT = 100

DEFAULT_BADGE_SCALE_PERCENT = 35
DEFAULT_BADGE_OPACITY_PERCENT = 100

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
# Campaign brief files (briefs/sample_campaign.json and friends): the
# same files the command-line pipeline runs from, offered on the form so
# a campaign can be started from one -- pick a product, the brief
# fields fill in.
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
            # Seeded from the backup zip -- the master set every product
            # starts from and every Reset goes back to. Only when there
            # is no zip are the loose shared PSDs copied instead.
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
# Where a saved template goes before this app writes over it. Editing
# the templates is the one action here that changes a file the user made
# by hand, and it cannot be undone from the results page -- so the
# version being replaced is kept, timestamped, every time.
TEMPLATE_BACKUPS_DIR = BASE_DIR / "_template_backups"

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
# ...and a PSD, flattened on the way in (transparency kept, so a logo or
# product cut out on its own layer arrives as a cut-out). These are the
# picture fields: the hero image, the logo/product layer updates, and
# the mood board. What is picked out of a PSD is its composite, not its
# layers -- a layered design belongs in "Size-specific PSD templates".
PICTURE_UPLOAD_EXTENSIONS = ALLOWED_LAYER_IMAGE_EXTENSIONS + (".psd",)

# Size-specific PSD templates: each one becomes the background for exactly
# the output size it's paired with (a Pillow-rendered flattened preview of
# the PSD, nothing more -- no layer extraction or role recognition). Kept
# separate from SUPPORTED_EXTENSIONS so the general hero image field stays
# plain-image/video only; PSD only enters through this dedicated section.
ALLOWED_PSD_TEMPLATE_EXTENSIONS = (".psd",)
MAX_PSD_TEMPLATES = 12
# Rows open on the form to begin with; the rest sit hidden behind the
# "+ Add another size" button (see index.html).
PSD_TEMPLATE_ROWS_SHOWN = 4

# The "quick campaign" single-input mode: upload just this one flagship
# size and every other exported size comes from default_templates/.
# 1920x1080: the flagship is the widescreen size the saved templates are
# designed at, and the floor for how big a backdrop is generated.
CONTENT_PSD_SIZE = (1920, 1080)
CONTENT_PSD_LABEL = f"{CONTENT_PSD_SIZE[0]}x{CONTENT_PSD_SIZE[1]}"

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
# Layers offered a "hide" checkbox. Deliberately not "background": it
# sits behind everything and hiding it leaves a hole rather than a
# cleaner creative -- the way to change a backdrop is to replace it,
# which the tool already does in three other places.
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

# Euclidean RGB distance under which a pixel counts as "matching" a brand
# color for find_missing_brand_colors() -- see that function's docstring
# for why an exact-match check would be too strict.
# The CTA button's own colour, and the value the form ships. Named so the
# render can ask whether somebody actually chose a colour -- a CTA built
# as a group is left as its designer drew it unless they did.
CTA_BUTTON_COLOR_DEFAULT = (0, 87, 184)
# White, which is also what a CTA label is in every template shipped so
# far -- so "not this" is the test for whether a colour was actually
# asked for, the same test the other three text layers use.
CTA_TEXT_COLOR_DEFAULT = (255, 255, 255)

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
SIZE_IN_NAME_RE_LOOSE = re.compile(r"\d{2,5}x\d{2,5}")


# Nothing useful comes of asking a provider for more than this, and the
# wait grows with the pixels.
MAX_GENERATED_EDGE = 2048

# Appended to a background prompt unless the user turns it off.
#
# Generative models are worst at exactly the things a backdrop doesn't
# need: faces, hands, crowds and lettering. A prompt like "marathon
# runners" asks for all four at once, and the melted faces and garbled
# race bibs that come back are what reads as "distortion" -- the model
# doing its worst on subjects that were never wanted, behind a template
# that already has its own product, logo and text.
#
# Steering away from those keeps the subject matter and drops the failure
# modes.
#
# It deliberately does NOT ask for defocus. An earlier version of this
# said "shallow depth of field, softly out of focus" -- reasoning that a
# soft backdrop composites better behind text, which is true. But it also
# meant a prompt reading "high resolution image of runners" was sent with
# "softly out of focus" stapled to it, and the blur that came back was
# the model doing exactly as asked. Legibility behind text is what the
# band and glow controls are for; the backdrop itself should be sharp.
# Appended to EVERY generated-image prompt, whatever else is switched on.
# Lettering is the one thing a backdrop can never want -- the template's
# own header, description and CTA sit on top of it -- and asking costs
# nothing, so it isn't left to a checkbox the way the styling guidance
# below is. Prevention only, though: models ignore this often enough that
# it is verified afterwards rather than trusted (see NO_TEXT_ESCALATION
# and the OCR check in src/text_check.py).
NO_TEXT_CLAUSE = (
    "no text, no words, no lettering, no numbers, no watermarks, no signage, "
    # A brand name in the prompt invites a wordmark, and a campaign
    # backdrop always has one in it somewhere -- so the exclusion has to
    # name that case specifically rather than trusting "no text" to cover
    # it.
    "no brand name, no wordmark, no logo, no packaging text, no labels"
)

# Used on a retry, once a generation has actually come back with text in
# it. Blunter and more repetitive on purpose: the polite phrasing above
# has already demonstrably failed for this prompt.
NO_TEXT_ESCALATION = (
    "absolutely no text anywhere in the image, no words, no letters, no numbers, "
    "no captions, no labels, no signs, no posters, no packaging text, no watermark, "
    "a completely textless photographic background"
)

# Things a prompt author types to say what they DON'T want. In the
# positive prompt they do the opposite of what was meant: "no text, no
# words" hands the model the tokens "text" and "words" to steer by, and
# it letters the picture (one run came back with "NEVFT WORDS" painted
# across it). The results page shows the prompt it used as
# `... [excluded: no text, no words]`, which is exactly the sort of line
# that gets copied back into the box, so the guard has to recognise its
# own output as well as plain English.
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
# Any noun, not only the text words: "no bottles or drink" typed into the
# prompt came back as four bottles, for the same reason "no text" came
# back lettered. A term is a short run of plain words up to the next
# comma, full stop, "just"/"only" or the end.
_ANY_TERM = r"(?:[A-Za-z][A-Za-z'-]*)(?:\s+(?!no\b|just\b|only\b|with\b)[A-Za-z][A-Za-z'-]*){0,3}"
PROMPT_NEGATION_PHRASE = re.compile(
    r"\b(?:no|without|free\s+of|zero|not\s+any|absolutely\s+no)\s+" + _NEGATION_QUALIFIER + _ANY_TERM
    # "no X or Y" / "no X and Y" continue the same exclusion; after a
    # comma only another explicit "no ..." does, so "no bottles, just
    # people having fun" keeps the part that was wanted.
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
    # Whatever is left that still talks about text goes too, clause by
    # clause. "no bottles or drink with text on it" loses its "no
    # bottles or drink" to the pattern above and would leave "with text
    # on it" standing in the prompt -- an order for text. A clause that
    # mentions lettering at all has no business in a picture prompt:
    # it is either an exclusion (kept, on the negative side) or a
    # remark about one (dropped).
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


# How many times to regenerate when the check finds text. Each retry is
# another API call -- real money on a paid provider -- so this is
# deliberately small, and 0 turns the retries off while leaving the
# warning in place.
AI_TEXT_RETRY_LIMIT = max(0, _env_int("AI_TEXT_RETRIES", 2))
# ...and how many of those a PAID provider gets (each one is an image).
AI_PAID_TEXT_RETRIES = max(0, _env_int("AI_PAID_TEXT_RETRIES", 2))

# Into the prompt itself on a retry -- what the model paints comes from
# the prompt, and the objects it paints come with markings unless told
# otherwise: a volleyball with its maker's name on it, a can with a
# label, a shirt with a slogan.
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

# Not a word about text in here, even to ask for room for it: "space
# for overlaid text" reads to a typography model as "put text here",
# and one run came back with a caption sitting in exactly that space.
BACKGROUND_PROMPT_GUIDANCE = (
    "sharp focus, crisp fine detail, high resolution, professional photography, "
    "no faces, no logos, no product shot, no bottle, no can, no packaging, no labels, "
    "even lighting, plenty of clean empty negative space, uncluttered composition"
)


def _keep_the_product_out_of_a_backdrop_prompt(prompt: str, product_name: str):
    """A backdrop prompt with the product in it, minus the product.

    The template's own product layer supplies the bottle, and a model
    asked for "HydroBoost sports drink" draws a bottle with HydroBoost
    written on it -- lettering the text check then has to smear out,
    which is the "bad image" that came back. Returns (prompt, what was
    taken out) -- the names and the words that ask for the product
    itself; the scene ("beach volleyball, sand and water") stays."""
    if not prompt:
        return prompt, []
    removed = []
    names = []
    if product_name:
        compact = re.sub(r"\s+", "", product_name)
        names = [re.escape(product_name.strip())]
        if compact.lower() != product_name.strip().lower():
            names.append(re.escape(compact))
    for pattern in names + [
        r"(?:sports?|energy|soft)\s+drinks?", r"\bbottles?\b", r"\bcans?\b", r"\bpackaging\b",
        r"\bproduct(?:\s+shot)?\b",
    ]:
        for match in re.finditer(pattern, prompt, flags=re.IGNORECASE):
            removed.append(match.group(0))
        prompt = re.sub(pattern, " ", prompt, flags=re.IGNORECASE)
    # Tidy what the removals leave behind: doubled spaces, empty
    # comma-separated parts, a leading comma.
    parts = [part.strip() for part in re.split(r"[,;]", prompt)]
    parts = [re.sub(r"\s{2,}", " ", part) for part in parts if part and re.search(r"[A-Za-z]", part)]
    return ", ".join(parts), removed

# The same guidance for a run that WANTS type in the picture. The
# no-logos/leave-room-for-text half of the clause above exists to keep a
# backdrop out of the template's way; asked to set type, the model is no
# longer painting a backdrop and that half is working against the brief.
BACKGROUND_PROMPT_GUIDANCE_WITH_TEXT = (
    "sharp focus, crisp fine detail, high resolution, professional graphic design, "
    "clean legible typography, balanced composition"
)


# Rough plain-language names for the brand-colour clause below. A hex
# triplet alone is a weak instruction to an image model -- it reads as
# text, not as a colour -- so each one is sent as "#0057b8 (blue)", which
# gives the model a word it actually steers on and keeps the exact value
# for anyone reading the prompt back.
_COLOUR_WORDS = (
    ((0, 0, 0), "black"), ((255, 255, 255), "white"), ((128, 128, 128), "grey"),
    ((255, 0, 0), "red"), ((0, 128, 0), "green"), ((0, 0, 255), "blue"),
    ((255, 255, 0), "yellow"), ((255, 165, 0), "orange"), ((128, 0, 128), "purple"),
    ((255, 192, 203), "pink"), ((165, 42, 42), "brown"), ((0, 255, 255), "cyan"),
    ((0, 128, 128), "teal"), ((245, 245, 220), "cream"), ((25, 25, 112), "navy"),
    # Extra anchors where one entry per hue misnames the colours brands
    # actually use: a mid blue like #0057b8 sits numerically nearer pure
    # teal than pure blue and came back "teal", which is not a word
    # anyone would steer this palette with.
    ((0, 87, 184), "blue"), ((65, 105, 225), "blue"), ((135, 206, 235), "light blue"),
    ((50, 205, 50), "bright green"), ((255, 122, 0), "orange"), ((220, 20, 60), "red"),
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


# What the automatic backdrop prompt must not collapse into. With
# nothing but a product name to go on, and asked for "the words set as
# the headline", Ideogram produced the same thing on every run whatever
# the seed: a wordmark, the product name as a logo with the second word
# filled with water, on a flat field. Different files, identical idea.
# The exclusion names that idea.
LOGO_NEGATIVE_CLAUSE = (
    "logo design, wordmark, lettermark, brand mark, typographic poster, "
    "plain flat background, text-only layout"
)

# Compositions for the automatic backdrop to draw from, one per run.
# The same brief on the same model with only the seed changing gives
# the same concept -- the seed varies the rendering, not the idea. Each
# of these is a different idea.
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


# Words that carry no picture. What is left of a campaign message once
# these are gone is its subject matter -- which is the part a backdrop
# can use, without the sentence a typography model would want to set.
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
    # On a text-free run the audience is "people" and the product is
    # "the product": every distinctive word in the prompt has come back
    # as lettering on one run or another -- ADULTS on a shirt from the
    # audience, HYDRO BOOST on a shirt from the product name. A backdrop
    # goes UNDER the template's own product and copy layers, so it does
    # not need either named; it needs a scene.
    who = "people" if textless else (audience or "the target audience")
    scene = scene.replace("the audience", who)
    if textless:
        scene = scene.replace("the product's", "the object's").replace("the product", "a plain unlabeled object")
    parts = [scene]
    if product_name and not textless:
        parts.append(f"featuring {product_name}")
    if audience and not pictured and not textless:
        # A scene without people in it still has an audience: it sets
        # the styling, so the brief's audience always reaches Ideogram.
        # Not on a text-free run: "styled for Active Adults 18-34" came
        # back with ADULTS lettered on the picture.
        parts.append(f"styled for {audience}")
    if campaign_message:
        if textless:
            # Not even as keywords. "mood: rehydrate, refreshing,
            # summer" came back with REHYDRATE and SUMMER set as type:
            # a typography model letters any distinctive word it is
            # given. The scene and the product are the whole brief here.
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


# A finished ad dropped on the mood board is the commonest reference
# there is -- and the worst one to hand Ideogram whole. A style
# reference carries layout and typography as much as palette, so a
# reference with a headline across it comes back as a backdrop with a
# headline across it, in a language of the model's own; the
# no-text clause loses to the picture every time. For a text-free run
# the words are cut out of the reference first: painted out when they
# are small, cropped away when a headline is too big to paint out
# convincingly. The look survives; the lettering does not.
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
    # Too much to paint: a headline, a lockup. Keep the tallest band of
    # rows with no text in it, if that is enough of the picture to be
    # worth matching.
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


def _brand_palette_phrase(brand_colors) -> str:
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
    if len(words) == 1:
        return (
            f"{words[0]} as the dominant colour of the scene, in the lighting, "
            "surfaces and background"
        )
    named = ", ".join(words[:-1]) + " and " + words[-1]
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
    parts.append("a hero shot of the product as the focus")
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


# What a whole-ad generation must not add of its own accord: the
# audience and market lettered as copy, and the fine print, disclaimers
# and pseudo-legal lines models like to fill a bottom edge with -- which
# come out as gibberish, since there are no real words to set.
FULL_AD_NEGATIVE_CLAUSE = (
    "extra text, fine print, disclaimer, lorem ipsum, gibberish lettering, "
    "misspelled words, cropped text"
)


# The margin every whole-ad size is asked to keep, in pixels of the
# final creative -- on top of whatever the provider's crop takes.
FULL_AD_EDGE_PADDING_PX = 10

# The same breathing room for text the app draws itself into a
# template's layer boxes. A designer's box that runs to the canvas edge
# (the skyscraper's description starts at x=0) would otherwise put the
# first letter of every line against the edge.
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
    # Where the "no lettering" instruction goes depends on the provider.
    # A real negative-prompt field is the right channel and the only
    # strong one: diffusion models handle negation in the positive prompt
    # badly, and "no text, no words" there feeds the tokens "text" and
    # "words" straight into what is steering the image. Ideogram's own
    # documentation says the positive prompt takes precedence over the
    # negative one, which makes stuffing it in the positive prompt not
    # merely weak but counterproductive.
    #
    # Providers without such a field (Pollinations' GET endpoint) fold it
    # into the prompt themselves -- weaker, but it's that or nothing.
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
        # A photograph, not a design -- Ideogram's design mode is a
        # poster generator and sets type on principle -- and the prompt
        # exactly as written: its MagicPrompt rewrite has been seen to
        # add a caption to a prompt that asked for none.
        extra.update({"photographic": True, "rewrite_prompt": False})
    if allow_text:
        # One call, taken as it comes: nothing to verify, so the retry
        # budget stays unspent. Lettering is wanted here, so the no-text
        # clause stays out -- but a caller's own exclusion (the palette
        # not becoming a swatch chart) still goes, on the channel that
        # works for it.
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
    # The offline placeholder draws the prompt across its own gradient on
    # purpose -- that is what makes it recognisable as a placeholder. It
    # would fail the check every single time, burn the whole retry budget
    # regenerating an image that is text by design, and pay for several
    # seconds of OCR to learn nothing.
    verify = getattr(provider, "name", "") != "mock"
    retry_limit = AI_TEXT_RETRY_LIMIT if verify else 0
    if getattr(provider, "cost_per_image", 0):
        # Every retry here is a paid image. Two more goes when the
        # first came back with lettering: a fresh picture beats a
        # painted-out one ("if there is text, create another image"),
        # and the paint-out afterwards is the last resort, not the plan.
        retry_limit = min(retry_limit, AI_PAID_TEXT_RETRIES)
    # Every attempt is kept: if none comes back clean, the one with the
    # least lettering is the one to paint out, not simply the last.
    tried = []
    while attempts <= retry_limit:
        # First attempt asks politely; every retry escalates, since the
        # polite phrasing has by then demonstrably failed for this prompt.
        # Escalation goes into the prompt itself as well as the negative
        # prompt: a volleyball comes with a brand on it, and the
        # negative prompt alone did not take the brand off the ball.
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

    Painting out means inventing what was behind the words, so it is only
    convincing on small, isolated lettering; remove_text() refuses
    anything larger rather than trading readable text for an obvious
    smear, and this reports that refusal plainly instead of implying the
    image was fixed.
    """
    attempt_word = f"{attempts} attempt{'s' if attempts != 1 else ''}"
    # Whatever it takes: paint out, re-read, paint again, and crop the
    # lettering off if painting can't finish it. A backdrop with a soft
    # patch on it is a backdrop; a backdrop with a headline on it is
    # not. What was done is reported, and so is anything still left.
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
    # The same choice the render makes: with two files for one size
    # (hydroboost-1920x1080 and tester-1920x1080, say) the later one in
    # name order is the one that renders, so it is the one whose
    # backdrop is taken.
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


# Flattened templates, keyed by (path, size, mtime). Drawing a PSD from
# its layers is right (see open_psd_flat) and slow -- a second for a
# large one -- and a page load asks for every template several times
# over. The key changes the moment the file does, so a template
# re-saved in Photoshop is re-read.
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
        # Two ways to be a near miss: both sides close (3480x2160 for
        # 3840x2160), or one side right and the other its digits
        # shuffled (1290x1080 for 1920x1080) -- the latter ranks first.
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
# Printed on startup and shown in the page footer. With auto-reload on
# (the default, see __main__ below) this tracks your latest save; if it
# ever lags behind an edit you just made, the process is serving stale
# code and everything you're looking at is from the old build.
print(f"[webapp] code build stamp: {BUILD_STAMP} (auto-reload on unless FLASK_RELOAD=0)")

SIZE_PRESET_CHOICES = [
    ("default", "Social defaults -- 1080x1080, 1080x1920, 1920x1080"),
    ("web-top7", "Web ad sizes (9) -- Leaderboard, Medium Rectangle, Skyscraper, etc."),
    ("broadcast", "Broadcast/video frame sizes (3) -- 1080p, 720p, 4K UHD"),
]

app = Flask(__name__, template_folder=str(BUNDLE_DIR / "templates"))
app.secret_key = os.environ.get("WEBAPP_SECRET_KEY", secrets.token_hex(16))
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB -- generous enough for a short product video
# Jinja compiles a template once and caches it for the life of the
# process. The dev reloader only watches .py files, so without this an
# edit to templates/index.html changed nothing in a running server --
# silently, with the old markup still being served. That reads as "the
# fix didn't work" rather than "the server hasn't seen the fix", and cost
# a real debugging session.
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
        # The form was remembered at the top of the request, before the
        # drops were looked at: overwrite those entries so the next
        # form opens with the switches off too.
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
        # force=True composites the layers rather than handing back the
        # cached preview, which is flat RGB: it is what keeps a cut-out
        # logo's transparency.
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


# What the mood board takes, beyond the PNG/JPEG/WebP Ideogram itself
# accepts: a video (its middle frame is the reference), iPhone HEIC,
# TIFF, GIF, BMP -- converted to PNG on the way in.
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
    prior_path = prior_job_dir / "uploads" / prior_rel
    if not prior_path.is_file():
        return None
    dest_path = uploads_dir / prior_rel
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # A link, not a copy: a carried-forward 4K template is 30-50 MB,
    # and every Edit copied every one again -- a thousand runs later the
    # output folder was 108 GB and the disk full. The file is never
    # rewritten in place (a fresh upload gets a new name), so the link
    # is safe; a filesystem that can't link gets the copy.
    try:
        if dest_path.exists():
            dest_path.unlink()
        os.link(prior_path, dest_path)
    except OSError:
        shutil.copy2(prior_path, dest_path)
    return dest_path


# The output folder is working space, not an archive: runs older than
# the newest JOB_KEEP_COUNT go, and older still if what's left is over
# JOB_DISK_BUDGET_GB. Sessions referenced by the session index and the
# run a fresh form carries files from are kept regardless.
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
    # The job's own saved form is the fallback's content: a draft kept
    # from a failed submission has no session index, and a run from
    # before session tracking has none either -- both still have their
    # fields to reopen.
    try:
        own = json.loads((JOBS_DIR / fallback_job_id / "form_state.json").read_text())
        fallback[0]["prefill"] = own.get("fields") or {}
        fallback[0]["prefill_files"] = own.get("files") or {}
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
    template_paths = _default_template_paths()
    for path in template_paths.values():
        for name in get_psd_text_layers(path, visible_only=True):
            editable.add(name)
    return editable


def _switched_off_layers() -> set:
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
    seen, visible = set(), set()
    template_paths = _default_template_paths()
    for path in template_paths.values():
        try:
            psd = PSDImage.open(path)
        except Exception:
            continue
        for layer in psd.descendants():
            name = layer.name.strip().lower()
            seen.add(name)
            if layer.visible:
                visible.add(name)
    return seen - visible


def _present_text_layers() -> set:
    """Every named text layer the saved templates have at all -- switched
    on or off.

    The companion to _editable_text_layers(), and the distinction matters
    for whether a section is offered: a layer that is PRESENT but off can
    be turned on in Photoshop, so its controls are worth showing greyed
    out with an explanation. A layer that isn't in any template has
    nothing to explain and no path to enabling it, so its controls aren't
    shown at all rather than sitting there permanently dead.
    """
    present = set()
    template_paths = _default_template_paths()
    for path in template_paths.values():
        for name in get_psd_text_layers(path):
            present.add(name)
    return present


ENV_FILE = BASE_DIR / ".env"


def _ideogram_key_status() -> dict:
    """What the page may say about the Ideogram key: whether one is set
    and its last four characters. Never the key itself."""
    key = (os.environ.get("IDEOGRAM_API_KEY") or "").strip()
    return {"set": bool(key), "hint": key[-4:] if len(key) >= 8 else ""}


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
        "market_copy_languages": market_copy_languages(),
        "ideogram_key": _ideogram_key_status(),
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


def restore_templates_from_backup(zip_path: Path | None = None, dest_dir: Path | None = None) -> tuple[list[str], list[str], str | None]:
    """Put the saved templates back to the copies in the backup zip.

    Every tester-WxH.psd in the zip replaces the one in `dest_dir` --
    default_templates/ itself, or a product's own folder under it (the
    one being replaced is moved to _template_backups/ first, stamped, so
    nothing is lost). Sizes the zip does not carry are left as they are.
    Returns (restored names, untouched sizes, error message)."""
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
        # This product only: its entry becomes just the brief; every
        # other product's memory is untouched. The top level follows
        # it, since it was the product last used.
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


@app.route("/reset", methods=["POST"])
def reset_form():
    """The "Reset form" button. A blank form -- every remembered field and
    every file kept from the last run forgotten -- and the saved
    templates put back to the ones in default_templates/template-backup.zip:
    the way to undo a run of drops and Photoshop edits and start over
    from the known-good set."""
    # The card's Reset posts the whole card, so the product it is for
    # comes along: its own folder is what gets restored. No product --
    # the shared set.
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


# Form settings remembered from one run to the next even on a fresh form
# (the Edit page carries a whole run forward; this is for the handful of
# things that belong to the brand rather than to a run). Kept in a small
# JSON next to the jobs, so a test's temp JOBS_DIR gets its own.
BRAND_COLOR_FIELD_NAMES = tuple(
    name for i in (1, 2, 3) for name in (f"brand_color_{i}", f"brand_color_{i}_enabled")
)
# Everything in the "custom hero image" and "size-specific PSD" sections:
# with an ad in progress, a fresh form should open on the same layout,
# copy, hide boxes and rows as the last run, not blank.
SECTION_FIELD_PREFIXES = ("layer_", "psd_", "upload_hero", "upload_custom_hero", "upload_ai_enabled")
# The hide boxes are NOT remembered: a run is the design, and hiding a
# layer is a one-off decision for that run. Carried over, every box
# ticked once stayed ticked, and a later run came back as nine blank
# canvases with a note nobody reads -- four times in one day.
REMEMBERED_SECTION_FIELDS = tuple(
    name for name in EDIT_TEXT_FIELD_NAMES + EDIT_CHECKBOX_FIELD_NAMES
    if name.startswith(SECTION_FIELD_PREFIXES) and not name.endswith("_hidden")
)
# The campaign brief too: product, market, audience and message are the
# same from run to run of one campaign, and retyping them was the
# first thing every fresh form asked for.
CAMPAIGN_BRIEF_FIELD_NAMES = ("campaign_name", "product_name", "market", "audience", "campaign_message")
# psd_make_saved starts ticked and stays as last set: ticked, a dropped
# PSD replaces the saved template for its size; unticked, drops are
# one-offs -- and the results page says so every time, since an
# untick that outlives the run it was meant for otherwise reads as
# "doesn't look like it updated".
REMEMBERED_FIELD_NAMES = (
    CAMPAIGN_BRIEF_FIELD_NAMES + BRAND_COLOR_FIELD_NAMES + ("copy_language", "psd_make_saved") + REMEMBERED_SECTION_FIELDS
)
# The files those sections hold (hero image, layer images, PSD rows)
# are carried from the last run on a fresh form too: a form that
# remembers the hide boxes but forgets the hero would be half a memory.
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
    # Two tries: the translator is a free public endpoint that drops the
    # odd call, and line-by-line copy means more calls per run -- one
    # dropped line left a header half translated.
    for _attempt in range(2):
        translated, ok = localize_message(text, language)
        if ok and translated:
            # Words that came back unchanged are already in the language
            # (or the translator gave up quietly): not a translation, so
            # not cached as one -- a cached "French of the French" was
            # standing in front of the real English behind those words.
            if translated.strip() != text.strip():
                cache[key] = translated
            return translated, True
        time.sleep(0.5)
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
        # A phrase the translator handed back unchanged maps to itself;
        # that entry says nothing about the English and, taken first,
        # hid the real one -- the header stayed French with English
        # chosen because "RÉHYDRATER..." pointed at "RÉHYDRATER...".
        if translated.strip() == english.strip():
            continue
        reverse.setdefault(translated.strip(), english.strip())
    return reverse


def _same_words(a, b) -> bool:
    """Whether two pieces of copy say the same thing, ignoring how the
    lines are broken and spaced (a type layer breaks lines with \\r, the
    form with \\n, and either may carry a trailing space)."""
    def norm(text):
        lines = [
            " ".join(line.split())
            for line in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        ]
        return "\n".join(line for line in lines if line)
    return norm(a) == norm(b)


def _english_source_of(words: str, reverse: dict):
    """The English behind `words` (line by line, so a header translated
    line by line reverses the same way), or None if `words` isn't a
    translation this machine made."""
    lines = [line.strip() for line in words.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
    if not lines or not reverse:
        return None

    def back(text):
        # One step back; None when `text` isn't a translation. Followed
        # repeatedly below: a Spanish template translated again on a
        # later run is Spanish-of-Spanish, and the English is two steps
        # behind it.
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
    out = {}
    for name, text in fields.items():
        translated, ok = _translate_copy(text, language, cache)
        out[name] = translated
        if text and ok and translated != text:
            notes.append(
                f"{COPY_LANGUAGE_ENGLISH_NAMES.get(language, language)} copy -- {name}: \"{text}\" -> \"{translated}\"."
            )
        elif text and not ok:
            failed.append(name)
    if failed:
        warnings.append(
            f"Couldn't translate the {', '.join(failed)} into "
            f"{COPY_LANGUAGE_ENGLISH_NAMES.get(language, language)} -- the translator (Google Translate via "
            "deep-translator) didn't answer, so those are drawn in English. Check the connection and run again."
        )
    if cache != before:
        try:
            _translation_cache_path().write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    return out


def _preferences_path() -> Path:
    return JOBS_DIR / "preferences.json"


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
    files = {
        name: filename
        for name, filename in (state.get("files") or {}).items()
        if name.startswith(REMEMBERED_FILE_PREFIXES) and filename
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
        editable_text_layers=_editable_text_layers(),
        present_text_layers=_present_text_layers(),
        switched_off_layers=_switched_off_layers(),
        hideable_layers=HIDEABLE_LAYER_NAMES,
    )


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
    # Every entry of "From a brief file" is a campaign: a card each,
    # filled from the brief, with whatever that campaign's product has
    # been given since (its memory) laid over the top.
    for choice in _brief_choices():
        key = _product_memory_key(choice.get("product_name"), choice.get("campaign"))
        if not key or key in seen:
            continue
        seen.add(key)
        prefill = _brief_prefill(choice)
        if key in products:
            prefill.update(_remembered_prefill(key))
        job_id, files = _remembered_files(key) if key in products else (None, {})
        cards.append({"prefill": prefill, "prefill_files": files, "edit_job_id": None, "carry_job_id": job_id})
    for key in products:
        if key in seen:
            continue
        seen.add(key)
        job_id, files = _remembered_files(key)
        cards.append({
            "prefill": _remembered_prefill(key),
            "prefill_files": files,
            "edit_job_id": None,
            "carry_job_id": job_id,
        })
    # The form as it always was -- the top-level memory -- when it is
    # not already one of the cards above (the last run was for no
    # product, or was remembered before products had entries), or when
    # there is nothing else to show.
    prefs = _load_preferences()
    top_key = _product_memory_key(prefs.get("product_name"), prefs.get("campaign_name"))
    top_prefill = _remembered_prefill()
    if not cards or (top_key not in seen and (top_prefill or prefs.get("last_job_id"))):
        job_id, files = _remembered_files()
        cards.append({
            "prefill": top_prefill,
            "prefill_files": files,
            "edit_job_id": None,
            "carry_job_id": job_id,
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
        session_id=session_id,
        editable_text_layers=_editable_text_layers(),
        present_text_layers=_present_text_layers(),
        switched_off_layers=_switched_off_layers(),
        hideable_layers=HIDEABLE_LAYER_NAMES,
    )


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
                # Most of these were already saved once by the time the
                # failure happened, which leaves the stream at its end --
                # and a second save from there is an empty file.
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
    # Editing a prior job (see /edit/<job_id>) carries a hidden
    # edit_job_id field -- load that job's saved form_state.json so file
    # fields the user didn't re-upload this time can be carried forward
    # (see _carry_forward_upload()) instead of forcing a re-upload.
    edit_job_id = (request.form.get("edit_job_id") or "").strip() or None
    # A fresh form remembers the last run's section files (see
    # _remembered_files()): they are carried forward exactly as on Edit,
    # but nothing else about that run -- its approvals in particular --
    # comes along.
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
    # Sizes ticked as approved in the run being edited. An approved
    # creative is finished: it is carried over untouched rather than
    # regenerated, and in normal mode its backdrop is pinned for the
    # other sizes -- see approved_prior_sizes below.
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
    # "Keep this image": re-run the batch against the artwork the last run
    # generated instead of asking the provider for a fresh one. Every
    # generation is a different picture even from an identical prompt, so
    # without this, adjusting a font size or a glow colour means losing
    # the backdrop you were adjusting it against -- and paying for the
    # replacement.
    upload_ai_keep = bool(request.form.get("upload_ai_keep"))
    approval_pinned_backdrop = False
    if (
        approved_prior_sizes
        and not upload_ai_keep
        and (prior_form_state.get("files") or {}).get("upload_ai_generated")
        and not request.form.get("upload_ai_full_ad")
    ):
        # Someone approved a size in the last run. The backdrop in it is
        # the backdrop they approved, so the other sizes are updated
        # against that same picture rather than a fresh generation that
        # would make the set inconsistent (and cost an image).
        upload_ai_keep = True
        approval_pinned_backdrop = True
    # Let the model set type. Off by default: a generated backdrop sits
    # under the template's own header, description and CTA, and lettering
    # there is noise competing with them. Ticked, the picture is being
    # asked to BE the creative -- a logo lockup, a headline, a laid-out
    # poster -- and every no-text defence downstream has to stand down or
    # it will spend the retry budget destroying what was asked for.
    upload_ai_allow_text = bool(request.form.get("upload_ai_allow_text"))
    # Let the model build the whole ad, not the backdrop under one. The
    # brief already holds the copy -- product, message, headline, CTA --
    # so it is composed into the prompt and the result IS the creative:
    # no template layers are drawn over it, because the model has already
    # drawn them and a second headline on top of the first is the one
    # thing this must not produce. Implies allow_text: an ad with the
    # lettering suppressed is a photograph.
    # Ticking the custom route is itself a statement that this batch is
    # built from the saved templates -- even with no file chosen yet. The
    # templates are a complete design on their own; a hero image only
    # replaces their background layer, and the layer overrides in that
    # same section are reason enough to run without one.
    upload_custom_hero_enabled = bool(request.form.get("upload_custom_hero_enabled"))
    # Write this run's copy back into the saved templates themselves, not
    # just into its own downloads. Off by default and deliberately so:
    # a template is a design meant to be reused, and folding one
    # campaign's words into it means every later campaign starts from
    # them. Ticked, it is the fastest way to make a change stick --
    # retype once, and every future run begins with the new copy.
    update_saved_templates = bool(request.form.get("update_saved_templates"))
    upload_ai_full_ad = bool(request.form.get("upload_ai_full_ad"))
    # The headline for a generated ad, kept separate from
    # layer_header_text on purpose. That one is an override for the
    # template's own header layer, and it greys out when the layer is
    # switched off in Photoshop -- correct for a layer nobody will draw,
    # and wrong here, where the words are not going into a layer at all
    # but into the prompt. Falls back to the layer field, then to the
    # campaign message, so a brief that is already filled in needs
    # nothing typed twice.
    upload_ai_headline = (request.form.get("upload_ai_headline") or "").strip() or None
    if upload_ai_full_ad:
        upload_ai_allow_text = True
    upload_ai_prompt = (request.form.get("upload_ai_prompt") or "").strip() or None
    # Exclusions typed into the prompt ("no text", "excluded: no words")
    # come out of it here and go on the negative channel instead, where
    # they work; see split_prompt_negations(). A "no text" typed with the
    # whole-ad or allow-text box ticked is the more specific instruction
    # of the two, so the run becomes a text-free backdrop -- with the
    # no-text clause, the OCR check and the retry back on -- rather
    # than an ad with the words "no text" set in it.
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
    # ALL_PROVIDER_NAMES, not PROVIDER_NAMES: the offline placeholder is
    # deliberately absent from the dropdowns but still accepted if asked
    # for by name, which is how the tests render without a network.
    if upload_ai_provider not in ALL_PROVIDER_NAMES:
        upload_ai_provider = "pollinations"
    # Defaults ON -- see BACKGROUND_PROMPT_GUIDANCE. The checkbox is
    # absent from a form that predates it, so the default has to survive
    # "not submitted", which is why this reads the marker field.
    upload_ai_background_style = bool(
        request.form.get("upload_ai_background_style")
        or not request.form.get("upload_ai_background_style_seen")
    )

    ai_hero_enabled = bool(request.form.get("ai_hero_enabled"))
    ai_hero_prompt = (request.form.get("ai_hero_prompt") or "").strip() or None
    ai_hero_provider = request.form.get("ai_hero_provider", "pollinations")
    if ai_hero_provider not in ALL_PROVIDER_NAMES:
        ai_hero_provider = "pollinations"

    # The two generators each have their own provider select, but they
    # are one choice: nobody means "the paid model here, the free one
    # there". They used to be able to disagree, and the consequence was
    # nasty to read -- pick Ideogram in Upload Creative, leave Manual
    # Creative on its Pollinations default, and the results page reports
    # a Pollinations failure while the form plainly shows Ideogram
    # selected. The form now keeps them in step; this covers what the
    # form can't reach: a saved batch from before that was true, a
    # disabled select that posted nothing, a non-browser client.
    #
    # The non-default value wins in a disagreement, because "pollinations"
    # is exactly what an unset or unsubmitted field yields -- so the other
    # one is the deliberate choice, whichever section it came from.
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

    # Campaign brief -- product name / market / audience / campaign
    # message. Purely informational context about *this* batch: it isn't
    # composited into the creatives (there's no template placeholder for
    # it), just carried through to the results page and saved/carried
    # forward like every other field, so a batch's intent stays attached
    # to it when reviewing or editing later.
    product_name = (request.form.get("product_name") or "").strip() or None
    # This product's own templates: default_templates/<product>/, made
    # from the shared set the first time the product runs. From here on
    # every scan, drop and style saved back in this request uses it.
    from flask import g as _g
    campaign_name = (request.form.get("campaign_name") or "").strip() or None
    _g.templates_dir = product_templates_dir(product_name, create=True, campaign_name=campaign_name)
    product_templates_created = (
        _g.templates_dir != DEFAULT_TEMPLATES_DIR
        and any(_g.templates_dir.glob("*.psd"))
        and (time.time() - _g.templates_dir.stat().st_mtime) < 5
    )
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
        return _keep_submission(
            "Campaign brief is required -- please fill in: "
            + ", ".join(missing_brief_fields)
            + "."
        )

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
        return _keep_submission(
            "That contains language we can't allow through -- please edit: "
            + ", ".join(flagged_fields)
            + "."
        )

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
    _remember_form_fields(request.form)

    headline = (request.form.get("header") or "").strip() or None
    message = (request.form.get("description") or "").strip() or None
    fit_mode = request.form.get("fit_mode", "crop")
    if fit_mode not in ("crop", "contain"):
        fit_mode = "crop"
    # How the hero image goes into each size's background layer: filled
    # (cropped to the layer's shape -- right for a photo) or fitted whole
    # (no cropping, the layer's edges padded with the picture's own edge
    # colour -- right for a finished composition nothing may be cut off).
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
    # Sizes whose uploaded PSD is to be used exactly as it is: no hero
    # in its background, no copy from the form drawn over its text, no
    # hide boxes. For someone uploading a finished creative for one
    # size, the file IS the creative -- and every "update" they make to
    # it was being painted over by the same hero and the same typed
    # description on every run, which read as the upload not taking.
    psd_as_is_sizes: set = set()
    # (label, Path) for every PSD actually uploaded *this request* (not
    # carried forward from a prior edit) -- scanned for profanity in their
    # text layers below, once content_psd's own fresh-upload is known too.
    fresh_psd_uploads = []
    # Sizes whose file was chosen THIS run, and which row holds each
    # size: a file uploaded now carries onto same-shape sizes whose own
    # file is only a carry-over from an earlier run (see below).
    fresh_psd_sizes: set = set()
    psd_row_by_size: dict = {}
    # Rows whose file is (or became) the saved template for its size --
    # written to form_state.json so a later run that carries the row
    # knows the saved template supersedes the file in it.
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
        # The "x" button next to a row on the Edit page (see index.html)
        # sets this hidden field so a carried-forward template can be
        # cancelled outright -- otherwise there'd be no way to say "stop
        # using a template here, go back to the hero image for this size"
        # short of overwriting it with a different .psd.
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
        # A size a few pixels off one the app knows is that size: a
        # 728x480 canvas dropped for the 720x480 slot goes to 720x480,
        # the same snap the content PSD has always had. (Otherwise the
        # row opened a 728x480 of its own and the 720x480 slot kept
        # rendering from the saved template -- "it's not using the
        # current PSD".)
        snapped = _snap_row_size(
            (psd_width, psd_height), set(_default_template_paths()) | set(SIZE_NAMES) | set(DEFAULT_SIZES)
        )
        if snapped != (psd_width, psd_height):
            psd_size_snaps.append((i, (psd_width, psd_height), snapped))
            psd_width, psd_height = snapped
        # A row remembered from an earlier run whose file WAS made the
        # saved template for this size (recorded in that run's
        # form_state as promoted) stays on the form as the chip that
        # says what was dropped, but the saved template is what
        # renders: it is that file, plus whatever later runs wrote into
        # it, and the older copy in the row would only shadow it. A row
        # that was never promoted -- dropped with the box unticked, or
        # carried from before -- renders from its own file, exactly as
        # uploaded. A freshly dropped file is the new template.
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

    # "Quick campaign" single-input mode: upload just the one flagship
    # 728x480 PSD and disregard the Output sizes / Custom sizes selections
    # entirely -- the exported batch becomes this size plus whatever's
    # already saved in default_templates/, nothing else.
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
        # The upload renders as its own size (see the size_templates merge
        # below) *and* pulls in whatever's already saved in
        # default_templates/ -- "Output sizes"/"Custom sizes" and the
        # general hero image are still ignored in this mode either way.
        sizes = []

    # The campaign hero image. This is the ordinary way in: rather than
    # designing and uploading a whole flagship PSD, drop in one picture
    # and it becomes the backdrop of the 728x480 template already saved
    # in default_templates/, which then carries onto every other saved
    # size. Same destination as the AI generator below -- the background
    # layer of every template -- just supplied by hand instead.
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

    # Reference pictures for the AI backdrop -- a mood board of up to
    # REFERENCE_LIMIT: files, pictures dragged off web pages, or both.
    # Sent to Ideogram together as style references, and described in
    # words in the prompt for every provider (see reference_look_phrase()).
    # Kept as numbered slots (upload_ai_reference, _2, _3) so Edit carries
    # each forward like any other upload, and one (x) drops one picture.
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
        # Dragged from a web page, or an address pasted in. Fetched now
        # and kept as a file from here on, so Edit carries it forward
        # like any upload and the page never has to fetch it twice.
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
    # A mood board is a LOOK -- palette, lighting, mood -- and that is
    # what goes to the model, in words. The pictures themselves go as
    # Ideogram style references only when asked: a style reference is
    # copied as a whole, layout and typography included, so a board of
    # finished ads came back as a poster with invented brand names on
    # it, and no amount of "no text" in the prompt could stop that.
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
                    # A reference that had words on it is an ad, and an
                    # ad handed to Ideogram as a style reference comes
                    # back as an ad -- a sign, a board, a label, with
                    # the words painted back in. Its LOOK still counts:
                    # the cleaned copy goes into the mood board that is
                    # described in the prompt (palette, lighting), but
                    # the picture itself is not sent to the model.
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
    # What the provider actually handed back, before the shortfall was
    # made up. Kept so the render loop can say which SIZES are softened
    # by it -- the enlargement is real for a size above this and a
    # non-event for one below, and one run-wide warning can't tell them
    # apart.
    upload_ai_source_size = None
    # Both are raised before background_notes/background_warnings exist,
    # so they wait here and are flushed onto those lists below.
    background_notes_pending = None
    background_warnings_pending = []
    kept_ai_path = None
    # NOT gated on upload_ai_enabled. Ticking "Keep this image" now
    # unticks and disables the generate checkbox in the form -- keeping
    # the previous image and generating a new one are opposite
    # instructions, and leaving both lit invited a run that did one while
    # appearing to promise the other. So by the time this arrives,
    # upload_ai_enabled is False whenever keep is ticked, and gating on
    # it here would make the checkbox a no-op that silently dropped the
    # artwork it was asked to preserve.
    if upload_ai_keep:
        # Carried forward from the previous run's job folder under its own
        # key, so it can only ever come back when it was explicitly asked
        # for. (A generated image carried forward *silently* is poison: it
        # outranks the fresh generation and overwrites its file, so a new
        # prompt returns a byte-identical result and the generator looks
        # broken. That is exactly the bug this key exists to keep fenced
        # off -- the checkbox is the fence.)
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
        # Ticked with nothing to carry forward: a first run, or a batch
        # whose previous job folder is gone. Silence here would be a
        # campaign rendered without the artwork the checkbox implied it
        # was preserving.
        background_warnings_pending.append(
            "\"Keep this image\" is ticked but there's no previous image to keep -- nothing was "
            "generated and the templates kept their own backdrops. Untick it and tick "
            "\"Generate the hero image with AI\" to make one."
        )
    # Full ad mode replaces every template with its own generation, so a
    # campaign backdrop generated here is paid for, waited on, and then
    # thrown away -- an extra call and up to 40s on top of the one-per-
    # size the mode already costs.
    if upload_ai_enabled and upload_ai_image is None and not upload_ai_full_ad:
        # A backdrop, not a product shot. This used to ask for
        # "professional studio product photo of X", which is the wrong
        # brief entirely for a layer that sits *behind* the template's
        # own product, logo and CTA -- two competing subjects in one
        # frame.
        #
        # Not "abstract BRANDED backdrop suggesting <product name>",
        # which this was for a long time. That is a request for a brand
        # mark, with a brand name handed over to render -- and models
        # oblige, with an invented logo and a wordmark under it. Every
        # other defence in this file is downstream cleanup for a problem
        # asked for right here, and none of them can win: Ideogram's own
        # documentation says the prompt takes precedence over the
        # negative prompt, and the OCR check can't read a blurred,
        # half-occluded mark well enough to flag it.
        #
        # The product name stays: it steers mood and subject matter,
        # which is the useful part. "unbranded" is what stops it being
        # read as a logo brief.
        if upload_ai_allow_text:
            # "unbranded" is the whole point of the default auto-prompt
            # and the opposite of what this run is for, so it doesn't
            # appear here.
            # The product name goes in QUOTED, as the words to set. Named
            # only as a subject ("creative for Hydro Boost") a model
            # treats it as art direction and letters whatever it likes,
            # or nothing; quoted, it renders those characters -- which is
            # the point of asking a typography model for type at all.
            # Type AND a picture. Asking only for headline typography
            # gets a typographic poster back -- words on a field of
            # colour, nothing behind them -- which is not a campaign
            # creative. The brief is a hero shot, or failing that a real
            # photographic scene, with the brand set over it.
            # The scene first and the type second. Led by 'the words
            # "Hydro Boost" set as the headline', Ideogram made the
            # headline THE picture: a wordmark on a flat field, every
            # run, whatever the seed. Led by a photograph, it makes a
            # photograph and sets the words over it.
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
        # The ticked brand colours, whichever prompt is in play. This
        # went only into the full-ad prompt, so a backdrop generated
        # with three swatches ticked came back in whatever palette the
        # model felt like -- and the brand-colour check on the results
        # page then flagged every size as missing them, which read as
        # the check being broken rather than the request never sent.
        # Appended to a typed prompt as well as the automatic one: the
        # swatches are a separate, explicit instruction, not something a
        # prompt author should have to restate.
        palette = _brand_palette_phrase(brand_colors)
        if palette:
            upload_ai_prompt_text = f"{upload_ai_prompt_text}, {palette}"
        if upload_ai_reference_image is not None:
            upload_ai_prompt_text = (
                f"{upload_ai_prompt_text}, {reference_look_phrase(upload_ai_reference_image)}"
            )
        if upload_ai_background_style:
            if upload_ai_prompt and not upload_ai_allow_text:
                # A backdrop goes UNDER the template's own product layer,
                # so the prompt mustn't ask for the product: asked for
                # "HydroBoost sports drink" the model drew a labelled
                # bottle, and the label is text.
                trimmed, left_out = _keep_the_product_out_of_a_backdrop_prompt(
                    upload_ai_prompt_text, product_name
                )
                if left_out:
                    # A prompt that was only the product ("a bottle of
                    # Hydro Boost") has no scene left once it goes: the
                    # automatic backdrop scene stands in.
                    if len(trimmed.split()) < 3:
                        trimmed = (
                            f"{_backdrop_scene(product_name, campaign_message, audience, textless=True)}, "
                            "unbranded, open uncluttered space, soft lighting"
                        )
                    upload_ai_prompt_text = trimmed
                    upload_ai_prompt_notes.append(
                        "Backdrop mode: "
                        + ", ".join(f'"{w}"' for w in dict.fromkeys(left_out))
                        + " left out of your prompt -- the product comes from the template's own "
                        "product layer, and a model asked to draw it letters the label, which is text. "
                        "Untick \"generate a background\" to have the picture include the product."
                    )
            upload_ai_prompt_text = (
                f"{upload_ai_prompt_text}, "
                f"{BACKGROUND_PROMPT_GUIDANCE_WITH_TEXT if upload_ai_allow_text else BACKGROUND_PROMPT_GUIDANCE}"
            )
        try:
            upload_ai_width, upload_ai_height = _generation_size(
                _default_template_sizes(), CONTENT_PSD_SIZE
            )
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
                        ", ".join(upload_ai_prompt_negations) or None,
                    ) if clause
                ) or None,
                style_reference=upload_ai_reference_bytes,
            )
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
                # Providers cap their output. Worth saying out loud:
                # anything smaller than the batch needs gets upscaled into
                # the bigger sizes, and "why is this blurry" is otherwise
                # a mystery with no visible cause.
                # Make the shortfall up once, here, rather than letting
                # every output size enlarge from the same small source
                # separately and unsharpened.
                # Read the shortfall off the image BEFORE upscaling it --
                # the upscale sets the size to exactly what was asked
                # for, so measuring afterwards reports "returned
                # 1920x1920 for a requested 1920x1920" and reads as a
                # warning about nothing.
                returned_width, returned_height = upload_ai_image.width, upload_ai_image.height
                upload_ai_source_size = (returned_width, returned_height)
                upload_ai_image = upscale_to_cover(
                    upload_ai_image, (upload_ai_width, upload_ai_height)
                )
                # Recorded once, as a note, so the run report still says
                # what the provider gave. The warning proper is raised
                # per size below, against the sizes it actually costs
                # something -- run-wide, it read as "this whole batch is
                # soft" on a batch whose smaller half was untouched.
                background_notes_pending = (background_notes_pending or "") + (
                    f" The '{upload_ai_provider}' provider capped this at "
                    f"{returned_width}x{returned_height} against a requested "
                    f"{upload_ai_width}x{upload_ai_height}."
                )
        except ImageProviderError as exc:
            # Same resilience as the hero generator: a flaky free API
            # degrades to the offline placeholder rather than failing the
            # whole run, and says so instead of quietly looking worse.
            app.logger.warning(
                "AI provider %r failed for campaign artwork: %s", upload_ai_provider, exc
            )
            upload_ai_image = MockImageProvider().generate(upload_ai_prompt_text)
            # A warning, not a note. This was filed under "Details", which
            # is collapsed -- so the single most consequential thing that
            # can happen to a run (your artwork is a labelled placeholder,
            # not the image you paid for) sat folded away while the
            # placeholder itself was the most visible thing on the page.
            background_warnings_pending.append(
                f"Campaign artwork: the '{upload_ai_provider}' AI provider failed ({exc}) -- used the "
                f"offline placeholder generator instead. Prompt: \"{upload_ai_prompt_text}\"."
            )
        upload_ai_path = uploads_dir / AI_GENERATED_CAMPAIGN_FILENAME
        upload_ai_image.save(upload_ai_path)
    elif upload_ai_image is None:
        # Generator off entirely. Not "the reuse branch already filled
        # it in" -- that path has its own note to report and must not be
        # cleared here.
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
    missing_fonts_reported: set = set()  # each uninstalled template font is reported once, not once per size
    if background_notes_pending_hero:
        background_notes.append(background_notes_pending_hero)
    if reference_overflow_warning:
        background_warnings.append(reference_overflow_warning)
    if background_notes_pending:
        background_notes.append(background_notes_pending)
    background_warnings.extend(background_warnings_pending)

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
    # Full ad mode drives a templated batch as well: it renders one image
    # per saved size and hands each in as that size's template. Without
    # it here the saved sizes are never brought in, and a run that needs
    # no hero image at all is rejected for not having one.
    # A layer-image override is a statement about the templates too: a
    # replacement background, logo, CTA or product only means anything
    # composited into a template's named layer, so supplying one is
    # asking for a templated batch as surely as ticking the route is.
    # Read straight off the request rather than from
    # layer_image_overrides, which isn't built until further down.
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
        # The saved templates ARE the live design once uploads are
        # promoted into them (see psd_make_saved below): a run with
        # that on always starts from them.
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
    # "Exactly as uploaded" covers the saved templates too: once uploads
    # are promoted into default_templates/ the saved files ARE the
    # current designs, and putting the hero behind them or typed copy
    # over them is the same unwanted repaint it was for a row's file.
    if request.form.get("psd_as_is") and not upload_ai_enabled:
        psd_as_is_sizes.update(default_templates.keys())
        if content_psd_size is not None:
            psd_as_is_sizes.add(content_psd_size)
    elif request.form.get("psd_as_is") and upload_ai_enabled:
        # The generator is on: its backdrop goes into every template.
        # "Exactly as uploaded" is the custom-hero mode's setting and
        # comes back the moment that mode is picked again.
        background_notes.append(
            "AI hero image on: the generated backdrop goes into every template. \"Use these files exactly as "
            "uploaded\" applies in the custom hero mode, not here."
        )

    # A size-specific upload carries onto every other size in the batch
    # with the same proportions that has no upload of its own: a fresh
    # 1080x1080 PSD is the new 1200x1200 too, and a 1080x1920 the new
    # 720x1280, scaled to fit (same ratio, so no crop). Without this the
    # square that was just updated sat next to the saved square from
    # last time, and both had to be uploaded to change one design.
    # A size typed on a row that is a near miss for a size the batch or
    # the app already knows (3480x2160 for 3840x2160) is almost always a
    # typo -- and a costly one, since it exports as a size of its own
    # (29:18) instead of updating the one meant, and carries onto
    # nothing. Flagged, not corrected: the row is used as typed.
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
    # A file chosen this run beats a same-shape row still carrying last
    # run's file: with a 720x1280 and a 1080x1920 row both kept from
    # earlier runs, dropping a new file on one of them is meant to
    # update both -- otherwise the other row quietly kept the old
    # design and "the 9:16s don't update each other".
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

    # "Make these the saved templates": a file uploaded this run becomes
    # the template in default_templates/ for its size -- and for every
    # same-shape size it carried onto -- so the saved set is always the
    # current design and every future run (a fresh form included)
    # starts from it. The file each one replaces is kept in
    # _template_backups/. The row is then let go: the saved template
    # is the live one, and a row holding the same file would only
    # shadow it.
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
            # One file per size, always named for the size:
            # tester-720x480.psd. Whatever else in the folder claims
            # that size -- an older upload under its own name, a
            # hydroboost-... left from before -- is moved to the
            # backups folder, so the size has exactly one template and
            # its name says which.
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
            # A layer that exists but sits entirely off the canvas (dragged
            # past the artboard edge, or empty) counts as absent above --
            # say which, since "missing" sends someone looking for a layer
            # that is right there in the Layers panel.
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

    # Layer overrides -- lives in the PSD section only: swap the
    # description text, logo image, CTA image, or product image, applied
    # to EVERY template-covered size that has a matching named layer (each
    # size's own PSD has its own layout, so the same override lands at a
    # different position/scale per size, driven by that size's own layer
    # bbox). Only meaningful when there's at least one template-covered
    # size to apply them to; parsed once, applied per-size in the render
    # loop below via get_psd_layer_boxes()/apply_layer_*_override().
    # A PSD dropped this run is the design: the styling the form carries
    # for the layers -- glow, shadow, outline, colour, font, size, for
    # the text layers and the pictures -- is switched off for this run
    # and forgotten, so the file's own styling shows. Without this the
    # round trip app -> Photoshop -> app never closed: a glow recoloured
    # in Photoshop came back drawn over by the green one the form still
    # remembered from the run that made the file. Tick a control again
    # to override the file from here on.
    if fresh_psd_drops:
        switched_off = _switch_off_form_layer_styling()
        if switched_off:
            background_notes.append(
                "A PSD was dropped this run (" + ", ".join(fresh_psd_drops) + "), so the form's own layer "
                "styling -- " + ", ".join(switched_off) + " -- was switched off and the file's styling is used. "
                "Tick a control again to override the file."
            )
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

    # The "legal" text layer -- disclaimers, terms, the small print a
    # campaign is required to carry. Handled exactly like header and
    # description rather than as a special case: it is a named type layer
    # in every saved template, it gets swapped per campaign more often
    # than either of them, and the whole point of the layer-override
    # feature is that the words in a template are not baked in.
    #
    # Defaults differ where the role does. Left-aligned like the
    # description, and the colour default matches it, but nothing about
    # this layer wants a glow or a backing band by default -- small print
    # is meant to be legible and unobtrusive, not decorated.
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

    # The CTA layer's own text override. No position field, unlike the
    # hero-image tool's CTA: a template's button sits in the box its
    # designer drew, and that box is what gets filled.
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
    # The label's own stroke, separate from the button's. One traces the
    # shape's edge and the other the letterforms -- the same number means
    # two different things, so they cannot share a field.
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

    # The language the copy is drawn in. Everything typed stays English
    # on the form (and in the saved run, so Edit shows what was typed);
    # the translation happens here, on the way into the templates.
    copy_language = (request.form.get("copy_language") or "en").strip().lower()
    if copy_language not in COPY_LANGUAGE_NAMES:
        copy_language = "en"
    # (source text -> translation) pairs already shown on this run's
    # results page, so a template phrase shared by several sizes is
    # listed once.
    translations_noted: set = set()
    english_behind = _english_behind_translations()
    # What was typed, before translation: the "save this copy into the
    # templates" path writes THIS into the saved templates, so a French
    # run doesn't quietly turn the English masters French.
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

    # Layers to leave out of every size entirely. A hide wins over any
    # content supplied for the same layer: "hide it" and "put this in it"
    # are contradictory, and honouring both would draw the thing that was
    # just asked to disappear.
    hidden_layer_names = {
        name for name in HIDEABLE_LAYER_NAMES if request.form.get(f"layer_{name}_hidden")
    }
    # Copy typed for a layer whose hide box is ticked: the hide wins (see
    # above), and the words never appear -- which reads as "my change
    # didn't take". Say which box is doing it.
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
        # Every layer hidden is almost never meant: the hide boxes carry
        # over from run to run with the rest of the form, and a set
        # ticked once for an experiment quietly empties every creative
        # after it -- which then reads as "the logo isn't rendering".
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
            return _keep_submission(f"Couldn't read the {layer_name} layer update image: {exc}")
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
        if layer_name in hidden_layer_names:
            # Kept in layer_upload_paths above, so the file survives an
            # Edit and comes back the moment the layer is unhidden -- it
            # just isn't drawn while the hide is on.
            continue
        layer_image_overrides[layer_name] = layer_image

    # The uploaded content PSD's own layers become overrides for every
    # other template size -- the point of the quick-campaign field is
    # "upload one flagship PSD and get the campaign", which means the
    # other sizes have to actually take on its artwork instead of
    # rendering from their saved templates untouched. A layer the user
    # uploaded by hand above wins; this only fills the gaps.
    propagated_layer_names = set()
    # Order matters, and it is the opposite of what it was. The generated
    # backdrop is set FIRST and wins outright: a ticked generator means
    # "make me a new background", every time, and the only thing that
    # stops it is "Keep this image" -- which prevents the generation from
    # happening at all, further up, rather than discarding it here.
    #
    # It used to be the other way round, with an uploaded hero outranking
    # a generation on the theory that an explicit file beats an invented
    # one. That reasoning fails on an Edit, which is where this tool is
    # actually used: the hero upload is carried forward automatically, so
    # it silently outranked every later generation. Ticking the generator
    # on an edit appeared to do nothing at all -- it ran, cost an API
    # call, and had its result thrown away.
    #
    # Deliberately NOT recorded in layer_upload_paths: that dict is what
    # gets carried forward on an Edit, and a generated image carried
    # forward is poison. It would come back as though the user had
    # uploaded it and overwrite the new file on its way past -- so
    # changing the prompt and re-running produced a byte-identical
    # result. A generated image belongs to the run that generated it.
    #
    # Set before the content PSD's own layers are propagated below, and
    # that loop skips any layer already overridden, so the generated
    # backdrop wins over the PSD's too. Ticking the generator with a PSD
    # uploaded can only mean "keep this design, change the backdrop".
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
    # Approved sizes are carried over from the previous run as they are
    # -- not regenerated, not re-rendered, not billed -- and put back in
    # their place among the results below.
    display_sizes = list(sizes)
    kept_sizes = [size for size in sizes if size in approved_prior_sizes]
    sizes = [size for size in sizes if size not in approved_prior_sizes]

    # AI-generated hero image, if the box was checked and there's an
    # actual gap for it to fill (a hero image was uploaded, or every
    # requested size already has a matching PSD template -- either way,
    # nothing to generate). Generated once, reused for every size that
    # needs it, exactly like an uploaded hero image would be -- saved
    # into uploads/ as hero_path so nothing downstream needs to treat it
    # any differently, including the Edit page's carry-forward.
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
            # One setting across both generators, the same way the
            # provider is one choice for both: the switch says whether
            # THIS RUN wants lettering, and a run that wants it in the
            # campaign artwork does not want it stripped out of the hero
            # image standing in the same creative.
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
            # Mirrors src/pipeline.py's own resilience (never let a flaky
            # free API turn into a hard failure) -- falls back to the
            # offline placeholder generator and says so plainly, so it
            # reads as "this specific provider had a bad moment," not as
            # a silent quality regression.
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
        # Reaching this means the AI-hero fallback above either wasn't
        # checked or wasn't applicable -- point directly at that checkbox
        # rather than leaving "upload something" as the only way forward,
        # since it's the one-click fix for exactly this situation.
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

    # Full-ad mode: one generation PER SIZE, at that size's own aspect.
    # A laid-out ad cannot be cropped from a single square the way a
    # backdrop can -- cropping is what takes the right-hand third off a
    # headline. It is also why this is the expensive option, and why the
    # count is said out loud rather than discovered on the bill.
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
        # Constructing the provider is where a missing key surfaces
        # (IdeogramProvider() reads IDEOGRAM_API_KEY and refuses without
        # one). Unhandled, that was a 500 with a traceback in the
        # terminal and a blank error page in the browser -- for a
        # configuration problem the message already explains how to fix.
        # It goes back to the form as a notice instead. The backdrop
        # generator above degrades to the offline placeholder in this
        # case; a full ad has nothing to degrade to, since the model IS
        # the creative here.
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
            # A designed layout, and the prompt exactly as written: the
            # MagicPrompt rewrite has been seen to paraphrase the quoted
            # headline, which is then set misspelled.
            {"photographic": False, "rewrite_prompt": False}
            if getattr(provider_for_ads, "supports_render_mode", False) and not upload_ai_reference_bytes
            else {}
        )
        for width, height in sizes:
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
            # No path for it, so no layer override, no PSD rebuild, and
            # no text drawn over the model's own -- see the
            # size_template_paths lookups in the render loop. The
            # template is remembered separately: the whole-ad PSD
            # written in the render loop carries its real elements
            # hidden beneath the model's picture.
            full_ad_templates[(width, height)] = size_template_paths.pop((width, height), None)

    creatives = []
    for width, height in sizes:
        # Only the sizes that genuinely enlarge past what came back. A
        # 160x600 cut from a 1024x1024 source loses nothing; a 1920x1080
        # is stretched to nearly twice the width it was drawn at, and
        # that is the one worth flagging.
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
            elif upload_ai_full_ad and (width, height) in full_ad_templates:
                # A whole-ad generation is one flat picture. The PSD for
                # it is a reconstruction -- background, subject and
                # painted text pulled apart after the fact -- with the
                # size's real logo, product and live type hidden beneath
                # for retouching. See src/ad_split.py for what that can
                # and can't do.
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
            # Every override that can redraw something has to be able to
            # get in here. This gate listed the header and description
            # text and nothing else, so a run whose only instruction was
            # a CTA label -- or legal copy, or a colour, or an outline --
            # skipped the whole block and rendered the template
            # untouched. It went unnoticed because the AI generator puts
            # a background override in layer_image_overrides, which held
            # the gate open for every run that used it.
            # Localized copy for THIS size. In English every layer draws
            # what was typed (nothing, mostly). In another language a
            # layer with nothing typed draws the template's own words
            # translated -- the file's header, description and legal in
            # French or Spanish, in the template's own font, size and
            # colour -- so choosing a language changes the creative even
            # when the form is otherwise blank. A size marked "exactly as
            # uploaded" gets only that: its own words in the language,
            # nothing typed and no other override.
            text_only_size = (width, height) in psd_as_is_sizes
            # The hide boxes are a custom-hero-mode setting: an as-uploaded
            # size ignores them on the preview, so its PSDs and clip
            # ignore them too -- with every box ticked, the download was
            # opening with every layer switched off.
            size_hidden_layer_names = set() if text_only_size else hidden_layer_names
            # "...but put the hero image in their background": an
            # as-uploaded size keeps its own text and layers and takes
            # the hero into its background layer, nothing else.
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
            # The rule every size is held to: a text layer is redrawn
            # only when its WORDS change (a language chosen, copy typed).
            # Otherwise its pixels are the file's pixels, untouched --
            # the design is the PSD, not this app's rendering of it.
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
                for layer_key in ("header", "description", "legal"):
                    # A hide box doesn't apply to an as-uploaded size, so
                    # its words are on the creative and get the language
                    # like any other -- skipping them here left half the
                    # sizes in English with every hide box ticked.
                    if size_text[layer_key] or (layer_key in hidden_layer_names and not text_only_size):
                        continue
                    own_words = (own_text_layers.get(layer_key) or "").strip()
                    if not own_words:
                        continue
                    file_words = own_words
                    # A template that is an export from an earlier
                    # French or Spanish run holds that language in its
                    # type layers. The English it came from is what
                    # every language starts from -- set back to English,
                    # the English is drawn; set to French, the French
                    # comes from the English, not from the Spanish.
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
                    # Line by line when the designer broke the lines:
                    # the translation then keeps the same lines, each in
                    # the style its English line had. (One line of
                    # context per call -- the trade for keeping the
                    # layout, and why the pairs are worth a read.)
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
                    # Already saying this in the chosen language -- a
                    # file saved from Photoshop after a Spanish run, say
                    # -- is left exactly as saved: Photoshop's own
                    # rendering of it, effects and all, beats a redraw.
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
                }[k] for k in ("header", "description", "legal"))
            )

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
                # Each overridden layer's own isolated RGBA patch (box-
                # positioned, transparent everywhere else) -- keyed by
                # lowercased layer name, e.g. "background"/"logo"/
                # "header". Populated below as each override is applied,
                # and reused when building the downloadable layered PSD
                # (see the "if applied_layers" block further down) so
                # that export shows exactly the new content on its own
                # layer instead of a single flattened image.
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
                # Where the type is DRAWN: the same boxes, kept
                # TEXT_EDGE_PADDING_PX in from the canvas edge. The
                # originals stay for cleaning, since the designer's own
                # pixels reach wherever the designer's box did. Taken
                # AFTER the remap above: taken before it, a 1280x720
                # file rendering as 1920x1080 had its text drawn at the
                # file's own coordinates -- a description landing on top
                # of the logo, in a box two-thirds the width it should be.
                draw_boxes = _inset_boxes_to_canvas(layer_boxes, (width, height))

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
                # ...and never on a text-only ("exactly as uploaded")
                # size: its background is not replaced, so the box wipe
                # below must use the file's own backdrop. With this
                # wrongly True the wipe found no new backdrop to wipe
                # to and quietly did nothing, and the template's old
                # words showed through under the new ones.
                # Whether the template has a background layer at all --
                # even an empty one (see the override loop below).
                has_background_layer = "background" in {
                    n.strip().lower() for n in get_psd_layer_names(psd_path_for_size)
                } if psd_path_for_size is not None else False
                background_replaced_this_request = (
                    "background" in size_image_overrides
                    and not ("background" in propagated_layer_names and (width, height) == content_psd_size)
                )
                # Every layer actually being replaced on THIS size. The
                # masked restore in _clean_layer_box() below reaches back
                # into the original composite, so it has to know which
                # layers are no longer supposed to come from there.
                overridden_layer_names = {
                    name
                    for name in size_image_overrides
                    if not (name in propagated_layer_names and (width, height) == content_psd_size)
                } | (set() if text_only_size else hidden_layer_names)
                # final_image as it stood with the new background painted
                # in but before every other layer was composited back on
                # top -- the "clear the whole box" case below needs a
                # backdrop-only image to wipe to, and once the background
                # has been replaced this request the PSD's own backdrop is
                # the wrong one to use. Stays None unless that happens.
                background_only_image = None

                def _clean_layer_box(target_box, layer_name, full_box=False, restore_others=False):
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
                        # there rather than the PSD's original one...
                        if background_only_image is None:
                            return
                        final_image.paste(background_only_image.crop(target_box), target_box[:2])
                        # ...and then every OTHER layer's own pixels back
                        # on top of it, inside the box: the panel behind
                        # the description, a gradient plate, a rule --
                        # the design's own layers that the new hero
                        # sits under. Without this the box showed the
                        # bare hero plate (its dark edge colour, where
                        # the hero was fitted rather than cropped) as a
                        # band behind the redrawn words.
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
                    if full_box and restore_others:
                        # The box was widened past the layer's own edge
                        # (to take its effects with it), so put every
                        # OTHER layer's pixels back in that ring: the
                        # composite with this layer, its rendered
                        # companion and the backdrop hidden, through its
                        # own alpha.
                        others = get_psd_layer_foreground(
                            psd_path_for_size,
                            [layer_name, f"{layer_name} (rendered)", "background"],
                        )
                        if others is not None:
                            others = _fit_rgba_like_final_image(others)
                            patch = others.crop(target_box)
                            final_image.paste(patch, target_box[:2], mask=patch.split()[3])

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
                    size_image_overrides.keys(), key=lambda name: name != "background"
                )
                for layer_name in ordered_layer_names:
                    override_image = size_image_overrides[layer_name]
                    box = layer_boxes.get(layer_name)
                    if (
                        layer_name == "background"
                        and (box is None or (box[2] - box[0]) * (box[3] - box[1]) < 0.01 * final_image.width * final_image.height)
                    ):
                        # ...or no background layer at all (deleted
                        # rather than emptied): same thing, the hero
                        # fills the canvas, and the live-text file gets
                        # a background layer put in at the bottom.
                        # The background layer is there but empty -- its
                        # picture deleted in Photoshop so the hero can
                        # take its place -- so it has no box of its own.
                        # The background IS the canvas: the hero fills
                        # it. (Skipping it left the old backdrop and, with
                        # the wipe below expecting a new one, every
                        # redrawn text layer doubled over its old words.)
                        box = (0, 0, final_image.width, final_image.height)
                    if box is None:
                        continue
                    if layer_name in propagated_layer_names and (width, height) == content_psd_size:
                        # This size IS the uploaded PSD -- re-applying its
                        # own layers back onto itself would round-trip
                        # them through a fit/crop for no gain.
                        continue
                    if layer_name == "background":
                        # Artwork the model laid out has to fit, not
                        # fill: cropping a generated headline to a
                        # size's shape is what takes the end off the
                        # words. A plain backdrop still crops to fill --
                        # there is nothing in it to lose, and letterbox
                        # margins on a texture look like a mistake.
                        background_fit = (
                            "contain" if (upload_ai_allow_text or upload_hero_fit == "contain") else "crop"
                        )
                        final_image = apply_layer_background_override(
                            final_image, box, override_image, fit=background_fit
                        )
                        background_only_image = final_image.copy()
                        # The layers go back over the new backdrop as
                        # Photoshop drew them -- pixels AND layer styles
                        # -- lifted from the file's own flattened picture
                        # (pristine_final_image): see
                        # carry_flattened_effects(). The layers' own
                        # alpha (no approximated effects) says where the
                        # pixels are; the styles come from how the
                        # picture differs from the bare backdrop.
                        layers_alpha_source = (
                            get_psd_layer_foreground(psd_path_for_size, layer_name, effects=False)
                            if psd_path_for_size is not None
                            else None
                        )
                        if layers_alpha_source is None and psd_path_for_size is not None and not has_background_layer:
                            # Nothing to hide: with no background layer
                            # every visible layer is foreground, and all
                            # of it goes back over the hero.
                            layers_alpha_source = get_psd_composite_rgba(psd_path_for_size)
                        old_backdrop = (
                            get_psd_backdrop(psd_path_for_size, keep_layer_names=(layer_name,))
                            if psd_path_for_size is not None
                            else None
                        )
                        if old_backdrop is None and psd_path_for_size is not None and not has_background_layer:
                            # No background layer to compare against: the
                            # file's own picture shows its transparency
                            # as white, so white is the old backdrop.
                            old_backdrop = Image.new("RGBA", psd_canvas_size or final_image.size, (255, 255, 255, 255))
                        if layers_alpha_source is not None and old_backdrop is not None:
                            layers_alpha_source = _fit_rgba_like_final_image(layers_alpha_source)
                            old_backdrop = _fit_rgba_like_final_image(old_backdrop.convert("RGBA"))
                            final_image = carry_flattened_effects(
                                pristine_final_image, old_backdrop, final_image, layers_alpha_source.split()[3]
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
                        _clean_layer_box(box, layer_name)
                        final_image = apply_layer_image_override(final_image, box, override_image)
                        export_layer_patches[layer_name] = apply_layer_image_override(
                            Image.new("RGBA", final_image.size, (0, 0, 0, 0)),
                            box,
                            override_image,
                            keep_alpha=True,
                        )
                    applied_layers.append(layer_name)

                # Wipe each hidden layer's box back to the backdrop.
                # full_box=True on purpose: hiding means the whole box
                # goes, not just the layer's own pixels within it, which
                # is the same "replace, don't overprint" reasoning the
                # text overrides use.
                # A layer the template already switches off needs no
                # hiding, and wiping its box does real damage: boxes
                # overlap, so clearing an unused header's banner takes
                # the top off the logo underneath it, and clearing an
                # unused legal line erases the label out of the CTA
                # sitting in the same strip. Nothing was drawn there to
                # remove.
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
                # The form's glow and drop shadow on the picture layers
                # (logo, product): drawn under the layer's own pixels --
                # this run's upload if there was one, the template's
                # otherwise -- in this size's pixels, and into the
                # layered PSD's patch for the layer. The live-text file
                # and the saved template get them as real layer effects
                # (see live_text_effects below).
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

                # Say so when a template has a layer switched off. A
                # designer's eye-icon in Photoshop silently removes that
                # layer from this size and nothing else on the page
                # explains the hole -- the creative just comes back
                # missing its product, or its button, while every other
                # size has one. Worth a line: it is a two-second fix in
                # the template and an unanswerable mystery without it.
                if visible_in_template:
                    # Against every layer the template HAS, not just the
                    # ones with a usable box: a switched-off group
                    # reports an empty bounding box and drops out of the
                    # box map, so a CTA turned off in Photoshop -- the
                    # exact case worth reporting -- would go unmentioned.
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
                    # NOT full_box. That wipes the whole box back to the
                    # bare backdrop, and boxes overlap: the product shot
                    # in the 720x480 template sits across the top-left
                    # corner of the CTA, so hiding the product took that
                    # corner of the button with it and the label went on
                    # a button missing its left third. Hiding a layer
                    # means "draw everything except this" -- which is
                    # what the plain path does: the PSD recomposited
                    # with just this layer off (or, once the background
                    # has been replaced, the other layers restored from
                    # the original through a mask that leaves this one
                    # out). Its neighbours keep every pixel they had.
                    # ...widened by the layer's own effect reach, and
                    # with every other layer put back in that ring: a
                    # drop shadow Photoshop drew around a header reaches
                    # past the header's box, and hiding the header
                    # left its shadow behind -- "a shadow on a box".
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
                    # Shared by every text-layer override (description,
                    # header, ...) -- reads that named layer's own PSD
                    # font settings as the default, lets font
                    # family/size/color be overridden per field, and
                    # shrink-to-fits the text into the layer's box. See
                    # apply_layer_text_override() for the actual
                    # shrink-to-fit/leading-scaling behavior.
                    nonlocal final_image
                    if layer_key in size_hidden_layer_names:
                        # Hidden wins over any styling or wording set for
                        # the same layer. The two instructions contradict
                        # each other, and drawing the text would mean
                        # rendering exactly what was asked to disappear
                        # -- on top of a box already wiped clean for it.
                        return
                    # box_override lets a caller point this at something
                    # other than the named layer's own box -- the CTA
                    # group's label sits inside the group's box, not at
                    # it.
                    box = box_override or draw_boxes.get(layer_key)
                    if box is None:
                        return
                    # A layer with this name that is NOT live type -- a
                    # description rasterised in Photoshop, say -- is a
                    # picture, and is treated like one: shown exactly as
                    # it is, never wiped, never drawn into. Typed copy
                    # and styling for it are noted and left alone.
                    # (Only in a file that has live type at all -- a
                    # Photoshop file. A template built from pixel layers
                    # alone still takes its header and description as
                    # text boxes, as it always has.)
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
                        # Sitting under this size's background in the
                        # stack: covered, exactly as Photoshop shows it.
                        # (A layer merely switched off is different --
                        # typed copy turns it back on, as it always has.)
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
                    if clean:
                        # Wipe wherever this layer's words have ever been
                        # drawn, not just where they go now: the layer's
                        # designed box, and the box of the "(rendered)"
                        # companion an earlier export left beside it. A
                        # template that is itself an export carries last
                        # run's words in its stored preview, at last
                        # run's box -- wiping only the current box left
                        # the part that stuck out, so the description
                        # showed twice.
                        wipe = box
                        for extra in (layer_boxes.get(layer_key), layer_boxes.get(f"{layer_key} (rendered)")):
                            if extra:
                                wipe = (
                                    min(wipe[0], extra[0]), min(wipe[1], extra[1]),
                                    max(wipe[2], extra[2]), max(wipe[3], extra[3]),
                                )
                        # ...plus the layer's own effects: a drop shadow
                        # or glow Photoshop drew around the old words
                        # reaches past the box, and left a dark frame
                        # and a band under the redrawn header.
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
                    # The template's sizes are in ITS pixels. When the
                    # output is a different size -- a 1280x720 file
                    # standing in for 1920x1080, a 1080x1080 upload
                    # carried onto 1200x1200, a 1920x1080 onto 4K -- the
                    # boxes were scaled through the fit above, and the
                    # type has to scale with them or it comes out small
                    # in a big box.
                    template_scale = _template_scale(psd_canvas_size, (width, height), fit_mode)
                    if psd_text_style and template_scale != 1.0:
                        for key in ("font_size", "line_height"):
                            if psd_text_style.get(key):
                                psd_text_style[key] = max(int(round(psd_text_style[key] * template_scale)), 1)
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
                    # The template's own typeface, when it is installed on
                    # this machine and no other family was picked on the
                    # form. Otherwise the nearest bundled face stands in
                    # -- and the results page says so, since "why is the
                    # font different" is otherwise unanswerable.
                    effective_font_name = None if font_family else psd_text_style.get("font_name")
                    if align == "template":
                        align = psd_text_style.get("align") or "left"
                    if effective_font_name and not font_covers_text(effective_font_name, text):
                        # Installed, but without the letters this copy
                        # needs -- Apple Symbols has no accented Latin,
                        # so Spanish set in it came out as boxes.
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
                    # Copy with the template's own line breaks, on a
                    # layer whose lines were set at their own sizes (a
                    # header set "REHYDRATE / with a new summer /
                    # REFRESHING / DRINK"): each line is drawn at its own
                    # size, so a translation keeps the layout it had in
                    # English. Only when nothing on the form restyles
                    # the layer -- typed styling means one style.
                    text_lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
                    text_lines = [line for line in text_lines if line]
                    styled_lines = psd_text_style.get("lines") or []
                    # A glow, an outline or a colour from the form is
                    # drawn line by line too; only a font, a size or a
                    # background box means one style for the whole box.
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
                        # The layer's own glow and outline stay when the
                        # form doesn't restate them -- the same rule the
                        # shadow above follows. A yellow glow given to the
                        # header in Photoshop was lost the moment the
                        # form ticked a shadow or a colour, since either
                        # of those redraws the words.
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
                    # The layer's own effects, scaled with the box, unless
                    # the form restyled the layer: the drop shadow the
                    # design has stays on a translated header, the glow
                    # and stroke too -- no trip through Photoshop needed.
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
                    # Same call again, but onto a transparent canvas with
                    # keep_alpha=True -- isolates just the new glyphs as
                    # their own layer (see export_layer_patches above),
                    # for the downloadable PSD.
                    def _patch(with_background):
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
                            glow=glow,
                            glow_color=glow_color,
                            glow_size=glow_size,
                            glow_opacity=glow_opacity,
                            align=align,
                            show_background=with_background,
                            background_color=background_color,
                            background_opacity=background_opacity,
                            background_blur=background_blur,
                            keep_alpha=True,
                            stroke_size=stroke_size,
                            stroke_color=stroke_color,
                            keep_size=keep_design_size,
                            shadow=design_shadow,
                        )
                    export_layer_patches[layer_key] = _patch(show_background)
                    # The picture kept INSIDE a type layer (and its
                    # "(rendered)" twin) is the words alone: a box drawn
                    # behind them is this form's setting, not the
                    # layer's, and stored in the layer it travelled into
                    # every later template as a band behind the text no
                    # setting could switch off.
                    words_only_patches[layer_key] = _patch(False) if show_background else export_layer_patches[layer_key]
                    applied_layers.append(layer_key)
                    # The size the words were ACTUALLY laid out at, which
                    # is what the renderer measures a glow radius and a
                    # stroke width against. A percentage is meaningless
                    # to the live-text PSD without it: a type layer keeps
                    # its point size in the document's resource defaults
                    # scaled by the layer's own transform, so there is
                    # nothing on the layer to take a percentage OF, and
                    # guessing from the box overshot badly whenever the
                    # designer's text box was taller than its words.
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
                # Any CTA setting is a reason to redraw the button, not
                # just new words. Someone who picks a colour, a corner
                # radius or a stroke and leaves the label alone means
                # "this button, like that" -- gating the whole block
                # behind the text field made every one of those controls
                # do nothing until something was typed into a field they
                # have no relationship with.
                if (
                    not text_only_size
                    and (
                        layer_cta_text
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
                    # Skipped when a CTA image was uploaded for this run:
                    # that upload IS the button, and drawing one over it
                    # would bury what the user just supplied.
                    cta_box = layer_boxes.get("cta")
                    restyling_button_flat = bool(
                        layer_cta_button_color != CTA_BUTTON_COLOR_DEFAULT
                        or layer_cta_glow
                        or layer_cta_stroke_size
                        or layer_cta_radius is not None
                    )
                    # A CTA built as a group -- the designer's rounded
                    # rectangle with its label on top -- has a text layer
                    # of its own, and that is the thing being changed.
                    # Rewriting just the label keeps the button that was
                    # actually designed; painting the whole box and
                    # drawing a generic pill throws it away to change
                    # three words. Falls back to the pill when the CTA is
                    # a flat layer with no label inside it.
                    cta_label_box = (
                        get_psd_group_text_box(psd_path_for_size, "cta")
                        if psd_path_for_size is not None
                        else None
                    )
                    if cta_label_box is not None and cta_box is not None:
                        cta_label_box = map_box_through_fit(
                            cta_label_box, psd_canvas_size, (width, height), fit_mode
                        ) if psd_canvas_size else cta_label_box
                        # Is the shape itself being restyled, or only the
                        # words on it? The two need opposite treatment of
                        # what is already on the canvas.
                        restyling_button = bool(
                            layer_cta_button_color != CTA_BUTTON_COLOR_DEFAULT
                            or layer_cta_glow
                            or layer_cta_stroke_size
                            or layer_cta_radius is not None
                        )
                        if restyling_button:
                            # The designer's rectangle is being replaced,
                            # so it has to go first. Anything less leaves
                            # it showing around the new button wherever
                            # that is smaller or rounder -- a blue frame
                            # around an orange pill. full_box=True puts
                            # back what was BEHIND the whole group, giving
                            # the redraw a clean plate.
                            _clean_layer_box(cta_box, "cta", full_box=True)
                        else:
                            # Only the label changes, so the button stays
                            # exactly as drawn and just the old word is
                            # erased -- against the button it sits on, not
                            # against the template's backdrop.
                            # _clean_layer_box() would restore what was
                            # behind the whole GROUP, punching a hole the
                            # colour of the page through the middle of the
                            # button. Sampling just outside the word's own
                            # box gives the button's own colour.
                            final_image.paste(
                                _reconstruct_box_background(final_image, cta_label_box),
                                cta_label_box[:2],
                            )
                        # The rectangle itself, when the CTA settings say
                        # to restyle it. Left alone the designer's button
                        # is kept as drawn; give it a colour, a glow, a
                        # stroke or a corner radius and it is redrawn with
                        # them, the label going back on top afterwards.
                        # Drawn through the same pill routine the
                        # flat-layer path uses, with no text, so there is
                        # one implementation of what a button looks like.
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
                        # ...and the new words go across the button, not
                        # into the box the old ones happened to occupy.
                        # "Click" is four characters; a replacement fitted
                        # to its footprint comes out microscopic.
                        pad_x = int((cta_box[2] - cta_box[0]) * 0.08)
                        pad_y = int((cta_box[3] - cta_box[1]) * 0.18)
                        cta_label_box = (
                            cta_box[0] + pad_x,
                            cta_box[1] + pad_y,
                            cta_box[2] - pad_x,
                            cta_box[3] - pad_y,
                        )
                        # The group's own label when none was typed --
                        # redrawing the rectangle covers it, so it has to
                        # go back. get_psd_text_layers() can't see it:
                        # the words are on a layer inside the group.
                        cta_label_text = layer_cta_text or get_psd_group_text(
                            psd_path_for_size, "cta"
                        )
                        _apply_text_layer_override(
                            "cta",
                            cta_label_text,
                            layer_cta_font_family,
                            layer_cta_font_size,
                            True,
                            layer_cta_text_color,
                            # The glow belongs to the button, not to the
                            # words on it -- haloing both leaves the
                            # label wearing the shape's styling. The
                            # label's stroke is its own field.
                            stroke_size=layer_cta_text_stroke_size,
                            stroke_color=layer_cta_text_stroke_color,
                            align="center",
                            box_override=cta_label_box,
                            clean=False,
                        )
                        # _apply_text_layer_override() left the patch as
                        # the label's glyphs alone, so the PSD download
                        # got the words with no button under them -- the
                        # one thing the whole CTA section is for. Take
                        # the patch from the finished canvas instead:
                        # whatever is in that box IS the button, however
                        # it was arrived at (restyled pill, or the
                        # designer's own shape left alone with new words
                        # on it). Opaque over the backdrop it was cleaned
                        # to, which composites identically and can't drift
                        # from the render the way a re-derived patch can.
                        # ...but the label on its own is still wanted:
                        # in the live-text PSD the words are a type layer
                        # INSIDE the group, and giving that layer a
                        # picture of the whole button would bury the
                        # shape it sits on.
                        cta_label_patch = export_layer_patches.get("cta")
                        cta_patch = Image.new("RGBA", final_image.size, (0, 0, 0, 0))
                        cta_patch.paste(final_image.crop(cta_box).convert("RGBA"), cta_box[:2])
                        export_layer_patches["cta"] = cta_patch
                    elif cta_box is not None and (layer_cta_text or restyling_button_flat):
                        # A flat (pixel) CTA has no label to read back: redrawn only
                        # with words to put on it or a restyled button, never over a
                        # font choice alone.
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
                            final_image, cta_box, layer_cta_text or "", **cta_kwargs
                        )
                        export_layer_patches["cta"] = apply_layer_cta_override(
                            Image.new("RGBA", final_image.size, (0, 0, 0, 0)),
                            cta_box,
                            layer_cta_text or "",
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
                            # Downloaded under the template's own name
                            # (tester-720x480.psd), so the file that
                            # comes out is visibly the file that goes
                            # back in -- a drop of it replaces that
                            # template. Only a saved template's name;
                            # a per-run upload keeps the campaign name.
                            source_psd_download_name = (
                                psd_path_for_size.name
                                if psd_path_for_size.parent == templates_dir()
                                else None
                            )
                            # The artwork, in the template's own
                            # coordinate space. This file was a straight
                            # copy of the template, which meant the one
                            # download made to be edited showed the
                            # template's stock backdrop instead of the
                            # hero image the creative was built from --
                            # correct as a "source template", useless as
                            # a copy of what you just made. The patches
                            # built for the rendered PSD are in the
                            # OUTPUT canvas's space, so they can't be
                            # reused here; each override is re-applied
                            # against the template's own layer boxes,
                            # which needs no fit mapping at all.
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
                                        # An emptied background layer: the
                                        # hero fills the canvas here just
                                        # as it does in the preview, so
                                        # the live-text file carries it.
                                        box = (0, 0, source_size[0], source_size[1])
                                    if box is None:
                                        continue
                                    blank = Image.new("RGBA", source_size, (0, 0, 0, 0))
                                    if name == "background":
                                        # The same fit the preview used
                                        # (see background_fit above) --
                                        # a file that fits the hero
                                        # differently from the preview
                                        # is a different creative.
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
                            # Carry the typed copy into the live text.
                            # This file is a straight copy of the
                            # template -- the one download that still has
                            # every text layer editable and the CTA as a
                            # live group -- so without this it opens
                            # showing the template's placeholder words no
                            # matter what was typed into the form: the
                            # file someone opens *to edit the words* was
                            # the only one that didn't have them. "cta"
                            # names the group; the label inside it is
                            # what actually gets rewritten (see
                            # _named_type_layers()).
                            # size_text, not the typed fields: it holds
                            # this size's localized copy too -- the
                            # template's own header in French, say -- so
                            # the live type layers read what the creative
                            # shows, in the layer's own font, size and
                            # colour (the rewrite keeps the run's style).
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
                            # ...and the button's own properties into
                            # the live shape under that label. The
                            # rendered PSD has the restyled button drawn
                            # as pixels; here the fill and the stroke are
                            # still the ones Photoshop's shape toolbar
                            # edits, so the button that opens is the one
                            # asked for AND still a shape. Only when
                            # something was actually asked for -- an
                            # untouched CTA keeps the design as drawn.
                            cta_shape_style = {}
                            if layer_cta_button_color != CTA_BUTTON_COLOR_DEFAULT:
                                cta_shape_style["fill"] = layer_cta_button_color
                            if layer_cta_stroke_size:
                                cta_shape_style["stroke_width_pct"] = layer_cta_stroke_size
                                cta_shape_style["stroke_color"] = layer_cta_stroke_color
                            if layer_cta_radius is not None:
                                # Rounds the path Photoshop draws from,
                                # not just the number in the Properties
                                # panel -- see set_shape_layer_style().
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
                            if layer_legal_use_custom_color:
                                live_text_colors["legal"] = layer_legal_text_color
                            if layer_cta_text_color != CTA_TEXT_COLOR_DEFAULT:
                                live_text_colors["cta"] = layer_cta_text_color
                            # The glow and the stroke. Colour, size and
                            # weight are text styling and live inside the
                            # type layer; a glow is not styling at all in
                            # Photoshop but a layer EFFECT hanging off
                            # the layer, so live text the renderer had
                            # drawn with a green halo arrived here as
                            # flat green words and the file stopped
                            # looking like the creative. Written only
                            # into this file, never the layered one --
                            # that stays pixels throughout and correct
                            # whatever a given Photoshop makes of an
                            # effect authored here.
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
                                # Against the size this layer's words
                                # were actually drawn at, matching the
                                # renderer exactly. With no such size --
                                # a layer this run didn't retype -- the
                                # percentage has nothing to measure
                                # against and the effect is skipped
                                # rather than guessed at.
                                laid_out_at = rendered_font_sizes.get(key)
                                if not laid_out_at:
                                    continue
                                if on and size and opacity:
                                    # The renderer's halo (see
                                    # apply_layer_styled_lines) thickens
                                    # the glyphs by 0.8r, blurs by r and
                                    # boosts the result. Photoshop's Outer
                                    # Glow is Size + Spread; fitting the
                                    # two on real type puts the same halo
                                    # at Size 2.7r, Spread 45%, opacity
                                    # as set. Size alone at r was a faint
                                    # haze next to the preview.
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
                                    # _layer_effect_images grows the
                                    # picture by size/2 and blurs by
                                    # size/2: in Photoshop's terms, Size
                                    # 1.5x with a third of it solid.
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
                            # A typed font size becomes the type layer's
                            # own size -- live text, not pixels. In the
                            # PSD's pixels: the run drew at
                            # rendered_font_sizes (this size's pixels),
                            # and the template's canvas may be another.
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
                            # Last, after every layer edit above: the
                            # snapshot every viewer except Photoshop
                            # shows. Without it a correctly edited PSD
                            # looks untouched in Finder, Preview and
                            # quick-look, because those read the cached
                            # composite the template shipped with rather
                            # than redrawing the layers.
                            # Live type layers, with the drawn words
                            # kept beside them switched off.
                            #
                            # This file is the editable one -- that is
                            # the whole reason it exists next to the
                            # layered download -- so the type layer is
                            # what shows, carrying this run's copy, its
                            # colour and its effects. The renderer's own
                            # pixels go in as "<name> (rendered)",
                            # hidden: if a given Photoshop declines to
                            # recompose the type (which is its decision,
                            # not the file's, and is what made this
                            # awkward) the correct picture is one click
                            # away instead of a re-render away.
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
                                # And the type layer's OWN picture. A type
                                # layer carries a rasterized copy of how
                                # its words last looked, and that copy is
                                # what Photoshop puts on screen -- so a
                                # layer whose string had been replaced
                                # still opened reading the template's
                                # placeholder, in the template's black,
                                # while every pixel layer beside it
                                # updated. Replacing that picture with the
                                # words as this run drew them is what
                                # makes the string and the screen agree.
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

                            # Layers hidden on the form open switched off
                            # in the editable file too, so it matches the
                            # preview; nothing is deleted, the eye is off.
                            if size_hidden_layer_names:
                                set_layer_visibility(
                                    job_dir / source_candidate_filename,
                                    {name: False for name in size_hidden_layer_names},
                                )
                            set_flattened_preview(
                                job_dir / source_candidate_filename, final_image
                            )

                            # ...and, if asked, back into the saved
                            # template itself. Everything above edits a
                            # COPY: the template is read, copied into the
                            # job folder, and the copy is what gets the
                            # new words -- which is why retyping a
                            # description has never changed anything in
                            # default_templates/. That is the right
                            # default (a template is a design meant to
                            # outlive one campaign) but it is not what
                            # someone wants when the placeholder copy is
                            # simply wrong and should stay fixed.
                            #
                            # Only the words, and only as live type: a
                            # template whose text had been flattened to
                            # pixels could never be retyped again, which
                            # would make the next run's override
                            # impossible. Only templates in
                            # default_templates/ -- never a PSD uploaded
                            # for one request, which the user does not
                            # think of as a saved design.
                            if (
                                update_saved_templates
                                and psd_path_for_size is not None
                                and psd_path_for_size.parent == templates_dir()
                            ):
                                template_updates = dict(typed_copy_english)
                                # Words go into the template only with a
                                # picture of those same words beside
                                # them (see template_rasters below). On
                                # a French run the picture is French and
                                # the typed words English: saving the
                                # words alone left the template reading
                                # one thing and showing another, and
                                # every later run showed the picture.
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
                                # Anything at all to carry, not just
                                # words: restyling the CTA button and
                                # typing nothing is a perfectly ordinary
                                # thing to want to make permanent, and
                                # gating on the text alone silently
                                # dropped it.
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
                                        # The styling too, on the same
                                        # terms as the live-text
                                        # download: colour inside the
                                        # type layer, glow and stroke as
                                        # real layer effects. A template
                                        # that kept the new words in the
                                        # template's old colours would
                                        # still need every run to restate
                                        # the styling, which is most of
                                        # what makes retyping tedious.
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
                                        # ...and the button's own shape.
                                        # Written to the copy above and
                                        # not to the template, so a CTA
                                        # restyled and saved came back
                                        # the template's blue on the next
                                        # run -- the one part of the
                                        # button that didn't stick.
                                        if cta_shape_style:
                                            restyled_template = set_shape_layer_style(
                                                psd_path_for_size, {"cta": cta_shape_style}
                                            )
                                            retyped_template = (
                                                retyped_template + restyled_template
                                            )
                                        # The type layers' pictures, or
                                        # the saved template opens in
                                        # Photoshop reading its old
                                        # placeholder copy despite holding
                                        # the new string -- see the
                                        # live-text download above.
                                        # ...but only a picture that shows
                                        # the words the template now holds.
                                        # The run drew size_text (the
                                        # French, say) and the template got
                                        # the typed English, or kept its own
                                        # words unretyped: a picture of
                                        # other words beside them is what
                                        # left templates reading one thing
                                        # and showing another.
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
                                        # Last: the file's own flattened
                                        # snapshot, rebuilt from the layers
                                        # just edited. Pillow's Image.open
                                        # -- which is how this app loads a
                                        # template to render on -- reads
                                        # that snapshot, not the layers.
                                        # Left as Photoshop wrote it, the
                                        # next run drew the new copy on
                                        # top of the old placeholder text
                                        # still sitting in the picture.
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
                        # Inherit the template's live text layers instead
                        # of baking this size's copy into pixels like
                        # every other layer. psd-tools can author pixel
                        # layers and nothing else, so a type layer can
                        # only ever be carried over from a file that
                        # already has one -- which makes the template the
                        # single source of editable text in the pipeline.
                        # A typed description override becomes that
                        # layer's new words; with no override it keeps the
                        # template's own. Anything the template already
                        # holds as flattened art (the CTA, in every
                        # default_templates PSD today) can't come back as
                        # text and is simply left as pixels.
                        # Every layer as this render drew it. It used to
                        # inherit the template's live type layers here
                        # instead, which quietly threw away the styling
                        # that makes the creative look like itself: the
                        # green fill, the glow, the size the words were
                        # actually laid out at. Worse, Photoshop shows a
                        # type layer from its cached raster until you
                        # click into it, so a layer whose string had been
                        # rewritten still READ as the template's old copy
                        # -- a download that looked, in every way a person
                        # can see, as though nothing had been applied.
                        #
                        # So the two downloads now each do one job
                        # properly: this one is the creative, layered, and
                        # matches the preview exactly; the source PSD
                        # beside it is the editable one, with the same
                        # copy in live type layers and the CTA still a
                        # live group.
                        # A layer switched off in the template, or hidden
                        # by this run's own hide box, stays off here --
                        # unless this run drew a new one over it, in
                        # which case the new one is what the preview
                        # shows and what should be in the file.
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
                    arcname=f"{zip_entry_prefix}{creative.get('source_psd_download_name') or creative['source_psd_filename']}",
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
    form_state_fields.update(form_field_overrides)
    # As applied, not as typed: the form's select needs the normalised value to reselect.
    form_state_fields["upload_ai_speed"] = upload_ai_speed
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

    # Best-effort, like the write above: record this job into its
    # session's {slot -> job_id} index so a later Edit on ANY campaign
    # generated alongside it (see _load_session_campaigns()) can bring
    # all of them back, not just this one.
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

    # The notes and warnings exist only on the rendered page, which means
    # a provider failure -- the single most useful thing to know about a
    # run -- is gone the moment the tab is closed, and diagnosing one
    # depends on somebody transcribing red text from a screenshot. Write
    # them next to the job's own output instead.
    spend_note = _spend_note(spend)
    if spend_note:
        background_notes.append(spend_note)
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
    # Layers the template had switched off don't animate, even in a
    # per-size PSD written before the app wrote them switched off: the
    # source-template copy beside it still says which they were.
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
    # Stamped with the run it came from. Every run writes the same
    # filename (the product and size decide it, and neither changes
    # between runs), so a second download lands in Downloads as
    # "... (2).psd" -- macOS renames it silently, the tab in Photoshop
    # looks near enough identical, and the file being stared at is an
    # older run's. That is indistinguishable from the app not applying
    # the edits, which is exactly how it reads. Six characters of the
    # job id make two downloads impossible to confuse, and match the
    # run shown on the results page.
    stamped = f"{file_path.stem}_{job_id[:6]}{file_path.suffix}"
    # A live-text file goes out under its template's own name when the
    # results page asks (?as=tester-720x480.psd): the file that comes
    # out is the file that goes back in. No run stamp on it -- the
    # name IS the point.
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
        # A packaged build is double-clicked, not launched from a shell:
        # pick a free port, say where the app is, and open it. The
        # reloader is off because it restarts the process by re-running
        # the interpreter with the script's path, which does not exist in
        # a frozen build -- and there is no source to reload anyway.
        port = _free_port(port)
        url = f"http://127.0.0.1:{port}"
        print(f"Creative Automation Pipeline -> {url}   (close this window to stop)")
        print(f"Files live beside the app in: {BASE_DIR}")
        _open_browser_when_up(url)
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
        raise SystemExit(0)
    # Auto-reload on by default. This is a local dev tool, and the cost of
    # not reloading is invisible: an edited template or module keeps
    # serving the old behaviour, every symptom points at the change that
    # was just made, and the only clue is a build stamp in the footer that
    # nobody thinks to check. Flask's reloader watches the source and
    # restarts on save, so what's on disk is what's being served.
    #
    # FLASK_RELOAD=0 turns it off; FLASK_DEBUG=1 additionally enables the
    # interactive debugger, which is a different (and far less safe)
    # thing.
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(
        host="127.0.0.1",
        port=port,
        debug=debug,
        use_reloader=debug or os.environ.get("FLASK_RELOAD", "1") != "0",
    )
