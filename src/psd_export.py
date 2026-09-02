"""Build an editable, layered PSD from a rendered creative.

A Photoshop-native companion to the flattened PNG preview: instead of one
flat image, this assembles render_creative_layers()'s (name, RGBA image)
stack into a real multi-layer .psd, so the pieces a creative is actually
made of -- background, header, logo, message, CTA, badge -- can be hidden,
moved, or restyled independently in Photoshop instead of starting over from
a flat image.

Every layer here is a rasterized (pixel) layer, not live Photoshop type --
the header/message/CTA text is drawn the same way it is for the PNG, just
kept on its own transparent layer instead of already flattened onto
everything under it. That means the text itself isn't editable as
characters in Photoshop (moving/hiding/recoloring the whole layer still
works fine), which is a real limitation worth knowing about, not something
this module tries to hide.

Layers are also renamed and tightly cropped so the exported PSD can be
re-uploaded straight back into the app as a "Size-specific PSD template" (or
the 728x480 quick-campaign content PSD) and have it actually recognized --
see REUPLOAD_LAYER_NAMES below.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PIL import Image
from psd_tools import PSDImage

# webapp.py's PSD-template upload flow (both the per-size "Size-specific
# PSD templates" rows and the 728x480 quick-campaign content PSD field --
# they share the same validation/parsing code) only recognizes layers
# named (case-insensitively) "logo", "description", and "product" as
# required, plus "cta" as an optional fourth -- see REQUIRED_PSD_LAYERS and
# get_psd_layer_boxes() in webapp.py/src/image_ops.py. Mapping our own
# render_creative_layers() names onto those means a PSD downloaded here,
# tweaked in Photoshop, and uploaded right back in is already recognized --
# no manual renaming needed first.
#
# "Header" and "Badge" have no matching role in the upload flow today (it
# doesn't recognize a header or badge layer at all) -- they keep their own
# names. That's not a regression: an unrecognized layer just gets baked
# into the flattened background on re-upload, the same as it always has
# been for any layer name the upload flow doesn't look for.
REUPLOAD_LAYER_NAMES = {
    "Background": "product",
    "Logo": "logo",
    "Message": "description",
    "CTA": "cta",
}


def set_type_layer_colors(psd_path, colors: dict) -> list:
    """Recolour live Photoshop type layers in the PSD at `psd_path`, in
    place, keeping them editable text.

    `colors` maps a lowercased layer name to an (r, g, b) tuple. Only
    layers that are real type layers and are named in it are touched;
    everything else in the file is left exactly as it was.

    A type layer's colour lives in its text engine data, as a FillColor
    per style run -- so a run of text with mixed colours has several, and
    all of them are set. The floats are stored as psd-tools' own Float
    objects and mutated in place: replacing the list with plain Python
    numbers parses fine but blows up on save, since the writer expects
    objects that know how to serialize themselves.

    Returns the names of the layers actually recoloured. Best-effort by
    design -- a file psd-tools can't parse, or a type layer whose engine
    data is shaped unexpectedly, returns [] rather than raising, because
    this only ever decorates a download that is already correct.

    One caveat worth knowing: this rewrites the text's styling, not the
    rasterized preview Photoshop caches alongside it. Photoshop re-renders
    the type layer on open, so the colour is right there; a viewer that
    only reads the cached composite may still show the old colour.
    """
    try:
        from psd_tools import PSDImage
    except ImportError:
        return []
    try:
        psd = PSDImage.open(psd_path)
    except Exception:
        return []

    wanted = {name.strip().lower(): rgb for name, rgb in colors.items()}
    recoloured = []
    for layer in psd:
        name = layer.name.strip().lower()
        if name not in wanted or getattr(layer, "kind", None) != "type":
            continue
        red, green, blue = wanted[name]
        try:
            runs = layer.engine_dict["StyleRun"]["RunArray"]
        except Exception:
            continue
        touched = False
        for run in runs:
            try:
                fill = run["StyleSheet"]["StyleSheetData"]["FillColor"]
                values = fill["Values"]
            except Exception:
                continue
            # Values are [alpha, r, g, b] as 0..1 floats for an RGB fill.
            for index, component in enumerate((1.0, red / 255, green / 255, blue / 255)):
                if index < len(values):
                    values[index].value = component
            touched = True
        if touched:
            recoloured.append(layer.name)

    if not recoloured:
        return []
    try:
        psd.save(psd_path)
    except Exception:
        return []
    return recoloured


def _tight_bbox_crop(layer_img: Image.Image) -> Tuple[Image.Image, int, int]:
    """Crop `layer_img` (RGBA, full canvas size) down to its own non-empty
    content's bounding box, returning (cropped_image, left, top).

    This matters for more than file size: the app's PSD-template upload
    flow uses a recognized layer's bounding box as the exact region a
    replacement image gets pasted into (see apply_layer_image_override()/
    apply_layer_text_override() in src/image_ops.py) -- a "logo" or "cta"
    layer left at the full canvas size would report that entire canvas as
    its box, so a replacement logo/CTA image uploaded against it would get
    stretched to fill the whole creative instead of landing where the
    logo/button actually sits. A fully opaque layer (the background/
    "product" layer, always) naturally crops to the full canvas anyway,
    since there's no transparent margin to trim.

    Falls back to the untouched image at (0, 0) when there's nothing at
    all to crop to (a fully transparent layer -- shouldn't happen for a
    layer render_creative_layers() actually included, but handled rather
    than left to error).
    """
    bbox = layer_img.getbbox()
    if bbox is None:
        return layer_img, 0, 0
    left, top, right, bottom = bbox
    return layer_img.crop(bbox), left, top


def build_layered_psd(
    layers: List[Tuple[str, Image.Image]],
    size: Tuple[int, int],
    *,
    layer_names: Optional[dict] = None,
) -> PSDImage:
    """Assemble a render_creative_layers() stack into a psd_tools PSDImage.

    `layers` is the same (name, RGBA image) list render_creative_layers()
    returns -- in back-to-front order, index 0 first. Each entry becomes
    its own Photoshop pixel layer, added in that same order, so the
    resulting PSD's layer stack (bottom to top in Photoshop's own layers
    panel) matches exactly what render_creative() would have flattened
    them into.

    Each layer is renamed via `layer_names` (a {render_creative_layers
    name: psd layer name} mapping, defaulting to REUPLOAD_LAYER_NAMES --
    pass `{}` to keep the original render_creative_layers() names
    unchanged) and cropped to its own tight bounding box -- see
    _tight_bbox_crop() -- rather than left at the full canvas size, so a
    recognized layer's box is actually the region it visually occupies.

    Raises ValueError if `layers` is empty -- there's always at least a
    "Background" layer for a real creative, so an empty list almost
    certainly means the caller passed the wrong thing.
    """
    if not layers:
        raise ValueError("layers must contain at least one (name, image) entry")
    if layer_names is None:
        layer_names = REUPLOAD_LAYER_NAMES

    # "RGBA" (not "RGB") -- with an RGB-mode document, psd_tools stores a
    # layer's alpha as a separate "user layer mask" channel instead of a
    # normal transparency channel, and Pillow's own (much simpler) PSD
    # reader -- what webapp.py's PSD-template upload flow uses to find
    # "logo"/"description"/"product"/"cta" -- silently refuses to parse
    # any layer with more than 4 channels, i.e. it would silently drop
    # every layer here. RGBA keeps the alpha as a normal 4th channel, which
    # Pillow does understand -- see test_exported_psd_is_accepted_by_the_apps_own_template_upload_check.
    psd = PSDImage.new("RGBA", size)
    for name, layer_img in layers:
        rgba = layer_img if layer_img.mode == "RGBA" else layer_img.convert("RGBA")
        cropped, left, top = _tight_bbox_crop(rgba)
        psd_layer_name = layer_names.get(name, name)
        psd.create_pixel_layer(cropped, name=psd_layer_name, top=top, left=left)
    return psd


def save_layered_psd(
    layers: List[Tuple[str, Image.Image]],
    size: Tuple[int, int],
    dest_path,
    *,
    layer_names: Optional[dict] = None,
) -> None:
    """build_layered_psd() and write it straight to `dest_path`."""
    build_layered_psd(layers, size, layer_names=layer_names).save(dest_path)


def _rewrite_type_layer_text(layer, text: str) -> bool:
    """Replace a live type layer's copy with `text`, in place, keeping it
    editable text rather than turning it into pixels.

    A type layer stores its string TWICE, and Photoshop will disagree
    with itself if only one is updated: once as the `Txt ` value in the
    layer's type-tool descriptor (what psd-tools' own `.text` reads back)
    and again inside the EngineData blob at `Editor/Text` (what Photoshop
    actually lays out on open). Both are set here.

    The style and paragraph run arrays carry per-run character counts
    that have to keep summing to the new string's length -- a stale count
    is what makes Photoshop reject a file as damaged. Any extra runs are
    dropped and the survivor is stretched over the whole string, which
    means a replacement inherits the styling of the original's first run:
    mixed styling within one layer collapses to its first style. That is
    a deliberate trade for text that stays editable.

    The trailing NUL matters -- Photoshop's own strings carry it, and the
    run lengths count it -- hence the +1 and the "\\x00" on both writes.

    Returns True when the copy was actually rewritten. Best-effort by
    design, like set_type_layer_colors(): an unexpectedly shaped engine
    dict returns False and leaves the layer untouched rather than
    raising, because this only ever decorates a download that is already
    correct.
    """
    payload = f"{text}\x00"
    try:
        layer._data.text_data[b"Txt "].value = payload
        engine = layer.engine_dict
        engine["Editor"]["Text"].value = payload
    except Exception:
        return False
    length = len(payload)
    for key in ("StyleRun", "ParagraphRun"):
        try:
            lengths = engine[key]["RunLengthArray"]
            runs = engine[key]["RunArray"]
        except Exception:
            continue
        while len(lengths) > 1:
            lengths.pop()
            if len(runs) > 1:
                runs.pop()
        if lengths:
            lengths[0].value = length
    return True


def save_layered_psd_preserving_type(
    layers: List[Tuple[str, Image.Image]],
    size: Tuple[int, int],
    dest_path,
    *,
    template_path,
    preserve_text: Optional[dict] = None,
    layer_names: Optional[dict] = None,
) -> List[str]:
    """save_layered_psd(), except any layer named in `preserve_text` that
    is a live type layer in `template_path` is carried over as live type
    instead of being written as rendered pixels.

    `preserve_text` maps a lowercased layer name to the copy that layer
    should end up saying -- or to None to keep the template's own words.
    A name that isn't a type layer in the template is ignored, so asking
    to preserve "cta" against a template whose CTA is already flattened
    art costs nothing and changes nothing.

    Why it works this way: psd-tools can only ever CREATE pixel layers
    (PSDImage exposes create_pixel_layer and create_group, and nothing
    that authors a type layer), so live text can only be inherited, never
    generated. The document therefore starts as the template -- the one
    file in play that has real type layers -- gets emptied of its
    original artwork, and is refilled with this render's pixel layers,
    with the preserved type layer re-inserted at the same point in the
    stack it occupied in `layers`. Its z-order relative to everything
    else is preserved; what it loses is any font-size or alignment
    override, since those live in the render, not in the type layer.

    Returns the names of the layers actually kept live -- empty when
    nothing was preserved, in which case the file written is exactly what
    save_layered_psd() would have written. Falls back to that same plain
    export on any failure: a PSD download that is layered-but-rasterized
    is a far better outcome than no PSD at all.
    """
    wanted = {
        name.strip().lower(): text for name, text in (preserve_text or {}).items()
    }
    if not wanted:
        save_layered_psd(layers, size, dest_path, layer_names=layer_names)
        return []

    try:
        psd = PSDImage.open(template_path)
    except Exception:
        save_layered_psd(layers, size, dest_path, layer_names=layer_names)
        return []

    kept = {}
    for layer in list(psd):
        key = layer.name.strip().lower()
        if key in wanted and getattr(layer, "kind", None) == "type":
            kept[key] = layer
    if not kept:
        save_layered_psd(layers, size, dest_path, layer_names=layer_names)
        return []

    if layer_names is None:
        layer_names = REUPLOAD_LAYER_NAMES

    try:
        # Empty the template of its own artwork, keeping the detached
        # type layer objects alive in `kept` so they can go back in at
        # the right height in the stack below.
        for layer in list(psd):
            psd.remove(layer)

        preserved: List[str] = []
        for name, layer_img in layers:
            key = name.strip().lower()
            if key in kept:
                layer = kept[key]
                replacement = wanted[key]
                if replacement:
                    _rewrite_type_layer_text(layer, replacement)
                psd.insert(len(list(psd)), layer)
                preserved.append(layer.name)
                continue
            rgba = layer_img if layer_img.mode == "RGBA" else layer_img.convert("RGBA")
            cropped, left, top = _tight_bbox_crop(rgba)
            psd.create_pixel_layer(
                cropped, name=layer_names.get(name, name), top=top, left=left
            )
        psd.save(dest_path)
        return preserved
    except Exception:
        save_layered_psd(layers, size, dest_path, layer_names=layer_names)
        return []
