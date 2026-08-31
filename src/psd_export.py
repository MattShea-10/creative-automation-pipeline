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
