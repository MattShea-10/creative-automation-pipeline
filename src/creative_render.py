"""Compose a single finished creative from a hero image + overlays.

This is the one place that decides "what does a rendered ad actually look
like" -- fit the hero image into the target frame, then stack the header,
logo, and message overlays in the right order with the right collision
handling. `CreativePipeline` (the full campaign-brief pipeline, driven by
the CLI) and the web UI (a quick single-image generator, driven by a
browser form) both call this same function, so a creative looks and
behaves identically no matter which front end produced it.
"""

from __future__ import annotations

from typing import Optional, Tuple

from PIL import Image

from .image_ops import (
    VALID_LOGO_POSITIONS,
    add_badge_image,
    add_cta_button,
    add_header_banner,
    add_logo_watermark,
    add_message_banner,
    center_crop_to_ratio,
    header_banner_height,
    logo_render_size,
    message_banner_height,
    resize_to_contain,
)

VALID_FIT_MODES = ("crop", "contain")


def render_creative(
    hero_image: Image.Image,
    size: Tuple[int, int],
    *,
    message: Optional[str] = None,
    headline: Optional[str] = None,
    fit_mode: str = "crop",
    logo: Optional[Image.Image] = None,
    logo_position: str = "top-right",
    logo_scale: float = 0.16,
    logo_opacity: float = 1.0,
    logo_offset_x: int = 0,
    logo_offset_y: int = 0,
    header_text_color: Tuple[int, int, int] = (255, 255, 255),
    header_show_background: bool = True,
    header_glow: bool = False,
    header_glow_color: Tuple[int, int, int] = (255, 255, 255),
    header_align: str = "center",
    header_font_size: Optional[int] = None,
    message_text_color: Tuple[int, int, int] = (255, 255, 255),
    message_show_background: bool = True,
    message_glow: bool = False,
    message_glow_color: Tuple[int, int, int] = (255, 255, 255),
    message_align: str = "left",
    message_font_size: Optional[int] = None,
    badge_image: Optional[Image.Image] = None,
    badge_position: str = "top-right",
    badge_scale: float = 0.35,
    badge_opacity: float = 1.0,
    cta_text: Optional[str] = None,
    cta_position: str = "bottom-center",
    cta_button_color: Tuple[int, int, int] = (0, 87, 184),
    cta_text_color: Tuple[int, int, int] = (255, 255, 255),
    cta_font_size: Optional[int] = None,
    cta_font_family: str = "sans",
    cta_glow: bool = False,
    cta_glow_color: Tuple[int, int, int] = (255, 255, 255),
    cta_above_message: bool = False,
) -> Tuple[Image.Image, bool]:
    """Render one finished creative at `size` from `hero_image`.

    Order of operations: fit the hero image into the frame (crop or
    contain), then the header band (top), then the brand logo (composited
    after the header so it sits on top of that band, if `logo_position` is
    a top corner -- see below), then the message banner (bottom). The
    header's available width automatically shrinks to leave room for the
    logo when both are present *and* the logo sits in a top corner, so a
    long headline can't run underneath it.

    `logo_position` is one of add_logo_watermark()'s VALID_LOGO_POSITIONS,
    defaulting to "top-right" (the original, only-ever-supported
    placement). A top-corner logo ("top-left"/"top-right") is composited
    right after the header, same as always, and the header reserves space
    for it on that side. The three "below-header-*" positions
    (left/center/right) are also composited right after the header,
    aligned to that side, but offset down to clear the header's actual
    rendered height instead of sharing its row -- if there's no
    `headline`, they just sit at the top like the corresponding
    top/centered logo, since there's no header to be below. A logo at
    "bottom-left"/"bottom-right"/"center" has no header to avoid, so it's
    instead composited *last* -- after the message banner and badge image
    -- so it isn't covered by either. `logo_scale`/`logo_opacity` size and
    blend it the same way `badge_scale`/`badge_opacity` do for the badge
    image. `logo_offset_x`/`logo_offset_y` then nudge the logo right/down
    (negative for left/up) by that many pixels from wherever `logo_position`
    already placed it -- a manual fine-tune on top of the preset, for every
    position including the below-header ones (added to, not instead of,
    the automatic below-header vertical offset).

    `header_text_color`/`message_text_color` and `header_show_background`/
    `message_show_background` control each banner's text color and whether
    it sits on a semi-transparent black plate (the default/original look)
    or floats directly over the image with an automatic outline for
    legibility instead -- see `add_header_banner()`/`add_message_banner()`
    in image_ops.py for the details. Both default to the original
    white-on-black-plate look, so existing callers are unaffected.

    `header_glow`/`header_glow_color` and `message_glow`/`message_glow_color`
    add an optional soft colored halo behind each banner's text instead of
    (or alongside) the background plate -- also independently configurable
    per banner, and off by default.

    `header_align`/`message_align` (each "left"/"center"/"right") control
    which edge of the banner's text area each line is anchored to --
    default to "center" for the header and "left" for the message, matching
    the original fixed behavior. `header_font_size`/`message_font_size`
    pin that banner's text to an exact pixel size instead of the automatic
    largest-that-fits sizing -- also independently settable per banner, and
    unset (autofit) by default.

    `badge_image`, if given, is a second image composited independently
    of the brand `logo` -- e.g. a "Sale" sticker, seasonal seal, or a
    full-frame tint/texture -- at `badge_position` (one of
    add_badge_image()'s VALID_BADGE_POSITIONS), sized by
    `badge_scale` and blended at `badge_opacity`. A "full" position is
    composited right after the hero image is fit into the frame, *before*
    the header/logo/message overlays, so it reads as a backdrop treatment
    text can sit on top of; every other position is composited *last*, on
    top of everything else, so it reads as a badge sitting on the finished
    creative.

    `cta_text`, if given, adds a filled, pill-shaped call-to-action button
    (e.g. "Shop Now") at `cta_position` (one of add_cta_button()'s
    VALID_CTA_POSITIONS, defaulting to "bottom-center" -- the common
    "sticky action bar" placement), colored by `cta_button_color`/
    `cta_text_color` and sized by `cta_font_size` (autofit to the frame if
    omitted). `cta_font_family` picks the button label's typeface (one of
    add_cta_button()'s VALID_FONT_FAMILIES). `cta_glow`/`cta_glow_color`
    add a soft colored halo around the whole button shape for a "lit up"
    look. It's composited last of all -- after the badge image too --
    since a CTA needs to stay visible/actionable no matter what else is on
    the creative.

    `cta_above_message`, when True, lifts a bottom-positioned button
    ("bottom-left"/"bottom-right"/"bottom-center") clear of the message
    banner instead of letting the two overlap -- both are otherwise drawn
    independently and land in the same bottom strip. Has no effect for a
    top/center `cta_position`, or when there's no `message` to avoid.
    Off by default, matching the original (overlapping) behavior.

    Returns (final_image, logo_composited) -- the boolean is handed
    straight to the brand-compliance check, which needs to know
    deterministically whether a logo was actually placed.
    """
    if fit_mode not in VALID_FIT_MODES:
        raise ValueError(f"fit_mode must be one of {VALID_FIT_MODES}, got {fit_mode!r}")
    if logo is not None and logo_position not in VALID_LOGO_POSITIONS:
        raise ValueError(f"logo_position must be one of {VALID_LOGO_POSITIONS}, got {logo_position!r}")

    width, height = size
    if fit_mode == "contain":
        canvas = resize_to_contain(hero_image, (width, height))
    else:
        canvas = center_crop_to_ratio(hero_image, (width, height))

    if badge_image is not None and badge_position == "full":
        canvas = add_badge_image(
            canvas, badge_image, position="full", opacity=badge_opacity
        )

    # A top-corner logo shares the header row, so it's composited right
    # after the header (which reserves space for it) -- same as it's
    # always worked. "below-header" doesn't share the row (no reserved
    # space needed) but is also composited right after the header, offset
    # down by the header's actual rendered height, so it sits just under
    # it rather than overlapping. Any other position has no header to
    # avoid, so it's composited last instead (with the badge/CTA), so it
    # isn't covered by the message banner or badge image that come after
    # the header.
    logo_in_top_corner = logo is not None and logo_position in ("top-left", "top-right")
    logo_below_header = logo is not None and logo_position in (
        "below-header-left", "below-header-center", "below-header-right",
    )
    logo_composited_late = logo is not None and not logo_in_top_corner and not logo_below_header

    if headline:
        reserved_right = 0
        reserved_left = 0
        if logo_in_top_corner:
            logo_w, _, logo_margin = logo_render_size((width, height), logo, scale_frac=logo_scale)
            if logo_position == "top-right":
                reserved_right = logo_w + logo_margin
            else:
                reserved_left = logo_w + logo_margin
        canvas = add_header_banner(
            canvas,
            headline,
            reserved_right=reserved_right,
            reserved_left=reserved_left,
            text_color=header_text_color,
            show_background=header_show_background,
            glow=header_glow,
            glow_color=header_glow_color,
            align=header_align,
            font_size=header_font_size,
        )

    logo_composited = False
    if logo_in_top_corner or logo_below_header:
        logo_y_offset = 0
        if logo_below_header and headline:
            gap = max(int(min(width, height) * 0.03), 6)
            logo_y_offset = header_banner_height((width, height), headline, font_size=header_font_size) + gap
        canvas = add_logo_watermark(
            canvas,
            logo,
            position=logo_position,
            scale=logo_scale,
            opacity=logo_opacity,
            x_offset=logo_offset_x,
            y_offset=logo_y_offset + logo_offset_y,
        )
        logo_composited = True

    if message:
        canvas = add_message_banner(
            canvas,
            message,
            text_color=message_text_color,
            show_background=message_show_background,
            glow=message_glow,
            glow_color=message_glow_color,
            align=message_align,
            font_size=message_font_size,
        )

    if badge_image is not None and badge_position != "full":
        canvas = add_badge_image(
            canvas,
            badge_image,
            position=badge_position,
            scale=badge_scale,
            opacity=badge_opacity,
        )

    if logo_composited_late:
        canvas = add_logo_watermark(
            canvas,
            logo,
            position=logo_position,
            scale=logo_scale,
            opacity=logo_opacity,
            x_offset=logo_offset_x,
            y_offset=logo_offset_y,
        )
        logo_composited = True

    if cta_text:
        cta_y_offset = 0
        if cta_above_message and message and cta_position in ("bottom-left", "bottom-right", "bottom-center"):
            gap = max(int(min(width, height) * 0.03), 6)
            cta_y_offset = message_banner_height((width, height), message, font_size=message_font_size) + gap
        canvas = add_cta_button(
            canvas,
            cta_text,
            position=cta_position,
            button_color=cta_button_color,
            text_color=cta_text_color,
            font_size=cta_font_size,
            font_family=cta_font_family,
            glow=cta_glow,
            glow_color=cta_glow_color,
            y_offset=cta_y_offset,
        )

    return canvas, logo_composited


def render_creative_layers(
    hero_image: Image.Image,
    size: Tuple[int, int],
    *,
    message: Optional[str] = None,
    headline: Optional[str] = None,
    fit_mode: str = "crop",
    logo: Optional[Image.Image] = None,
    logo_position: str = "top-right",
    logo_scale: float = 0.16,
    logo_opacity: float = 1.0,
    logo_offset_x: int = 0,
    logo_offset_y: int = 0,
    header_text_color: Tuple[int, int, int] = (255, 255, 255),
    header_show_background: bool = True,
    header_glow: bool = False,
    header_glow_color: Tuple[int, int, int] = (255, 255, 255),
    header_align: str = "center",
    header_font_size: Optional[int] = None,
    message_text_color: Tuple[int, int, int] = (255, 255, 255),
    message_show_background: bool = True,
    message_glow: bool = False,
    message_glow_color: Tuple[int, int, int] = (255, 255, 255),
    message_align: str = "left",
    message_font_size: Optional[int] = None,
    badge_image: Optional[Image.Image] = None,
    badge_position: str = "top-right",
    badge_scale: float = 0.35,
    badge_opacity: float = 1.0,
    cta_text: Optional[str] = None,
    cta_position: str = "bottom-center",
    cta_button_color: Tuple[int, int, int] = (0, 87, 184),
    cta_text_color: Tuple[int, int, int] = (255, 255, 255),
    cta_font_size: Optional[int] = None,
    cta_font_family: str = "sans",
    cta_glow: bool = False,
    cta_glow_color: Tuple[int, int, int] = (255, 255, 255),
    cta_above_message: bool = False,
) -> list:
    """Render the same creative render_creative() would, but return it as a
    stack of separate named layers instead of one flattened image -- used
    to build an editable/inspectable layered PSD (see
    src/psd_export.py's build_layered_psd()) alongside the usual flattened
    PNG.

    Takes *exactly* the same parameters as render_creative() (this
    mirrors its control flow step for step, on purpose, so the two never
    disagree about what gets drawn or in what order -- see
    test_render_creative_layers_matches_render_creative_flattened_output
    for the cross-check that keeps them honest) and returns
    List[Tuple[str, Image.Image]]: (layer_name, RGBA image) pairs, each
    the full canvas size, in back-to-front stacking order (index 0 is the
    bottom-most layer -- the fitted hero image itself -- and each
    following entry is meant to be alpha-composited on top of the ones
    before it). A layer is only included when render_creative() would
    actually have drawn it (e.g. no "Header" layer when there's no
    `headline`), so the layer list -- and the resulting PSD -- only ever
    has what the creative actually uses.

    Every overlay layer (everything but "Background") is built by calling
    the same add_header_banner()/add_message_banner()/add_logo_watermark()/
    add_badge_image()/add_cta_button() helpers render_creative() itself
    uses, but against a fully transparent canvas with their internal
    `_rgba=True` escape hatch instead of the real, growing composite --
    since none of those helpers' drawn content ever depends on what's
    already under them (only on the canvas size), this yields pixel-for-
    pixel the same overlay content, just isolated on transparency instead
    of already flattened onto everything drawn before it.
    """
    if fit_mode not in VALID_FIT_MODES:
        raise ValueError(f"fit_mode must be one of {VALID_FIT_MODES}, got {fit_mode!r}")
    if logo is not None and logo_position not in VALID_LOGO_POSITIONS:
        raise ValueError(f"logo_position must be one of {VALID_LOGO_POSITIONS}, got {logo_position!r}")

    width, height = size

    def transparent():
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))

    layers = []

    if fit_mode == "contain":
        background = resize_to_contain(hero_image, (width, height))
    else:
        background = center_crop_to_ratio(hero_image, (width, height))
    layers.append(("Background", background.convert("RGBA")))

    if badge_image is not None and badge_position == "full":
        layers.append((
            "Badge",
            add_badge_image(
                transparent(), badge_image, position="full", opacity=badge_opacity, _rgba=True
            ),
        ))

    logo_in_top_corner = logo is not None and logo_position in ("top-left", "top-right")
    logo_below_header = logo is not None and logo_position in (
        "below-header-left", "below-header-center", "below-header-right",
    )
    logo_composited_late = logo is not None and not logo_in_top_corner and not logo_below_header

    if headline:
        reserved_right = 0
        reserved_left = 0
        if logo_in_top_corner:
            logo_w, _, logo_margin = logo_render_size((width, height), logo, scale_frac=logo_scale)
            if logo_position == "top-right":
                reserved_right = logo_w + logo_margin
            else:
                reserved_left = logo_w + logo_margin
        layers.append((
            "Header",
            add_header_banner(
                transparent(),
                headline,
                reserved_right=reserved_right,
                reserved_left=reserved_left,
                text_color=header_text_color,
                show_background=header_show_background,
                glow=header_glow,
                glow_color=header_glow_color,
                align=header_align,
                font_size=header_font_size,
                _rgba=True,
            ),
        ))

    if logo_in_top_corner or logo_below_header:
        logo_y_offset = 0
        if logo_below_header and headline:
            gap = max(int(min(width, height) * 0.03), 6)
            logo_y_offset = header_banner_height((width, height), headline, font_size=header_font_size) + gap
        layers.append((
            "Logo",
            add_logo_watermark(
                transparent(),
                logo,
                position=logo_position,
                scale=logo_scale,
                opacity=logo_opacity,
                x_offset=logo_offset_x,
                y_offset=logo_y_offset + logo_offset_y,
                _rgba=True,
            ),
        ))

    if message:
        layers.append((
            "Message",
            add_message_banner(
                transparent(),
                message,
                text_color=message_text_color,
                show_background=message_show_background,
                glow=message_glow,
                glow_color=message_glow_color,
                align=message_align,
                font_size=message_font_size,
                _rgba=True,
            ),
        ))

    if badge_image is not None and badge_position != "full":
        layers.append((
            "Badge",
            add_badge_image(
                transparent(),
                badge_image,
                position=badge_position,
                scale=badge_scale,
                opacity=badge_opacity,
                _rgba=True,
            ),
        ))

    if logo_composited_late:
        layers.append((
            "Logo",
            add_logo_watermark(
                transparent(),
                logo,
                position=logo_position,
                scale=logo_scale,
                opacity=logo_opacity,
                x_offset=logo_offset_x,
                y_offset=logo_offset_y,
                _rgba=True,
            ),
        ))

    if cta_text:
        cta_y_offset = 0
        if cta_above_message and message and cta_position in ("bottom-left", "bottom-right", "bottom-center"):
            gap = max(int(min(width, height) * 0.03), 6)
            cta_y_offset = message_banner_height((width, height), message, font_size=message_font_size) + gap
        layers.append((
            "CTA",
            add_cta_button(
                transparent(),
                cta_text,
                position=cta_position,
                button_color=cta_button_color,
                text_color=cta_text_color,
                font_size=cta_font_size,
                font_family=cta_font_family,
                glow=cta_glow,
                glow_color=cta_glow_color,
                y_offset=cta_y_offset,
                _rgba=True,
            ),
        ))

    return layers
