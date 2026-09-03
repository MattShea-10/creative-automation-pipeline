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


def _named_type_layers(node) -> dict:
    """Map every name a caller could reasonably use for a live type layer
    in `node` to that layer, groups included.

    Two things make a flat `for layer in psd` scan miss text the rest of
    the app can see. Type layers nest: a designer's CTA is a group -- a
    vector rectangle with its label on top -- so the layer holding the
    words is a child called "Click", not the "cta" the form and the
    render both address it by. And the app names layers by their ROLE
    (header, description, legal, cta), which for a group is the group's
    name, not the label's.

    So both keys point at the same layer: its own name, and -- for the
    first type layer inside a group -- the group's name. A layer's own
    name wins if the two ever collide, since that is the more specific
    of the two.

    Returns {lowercased name: layer}. Order of preference aside, every
    lookup here is best-effort: a node psd-tools can't walk contributes
    nothing rather than raising.
    """
    found = {}

    def visit(container):
        for layer in container:
            name = (layer.name or "").strip().lower()
            if getattr(layer, "kind", None) == "type":
                found.setdefault(name, layer)
                continue
            if getattr(layer, "is_group", None) and layer.is_group():
                # The group's own name, claimed by the first type layer
                # inside it -- registered before descending so a nested
                # group's label can't take the outer group's name.
                for child in _first_type_layer(layer):
                    if name:
                        found.setdefault(name, child)
                    break
                visit(layer)

    try:
        visit(node)
    except Exception:  # noqa: BLE001
        pass
    return found


def _first_type_layer(group):
    """Yield the type layers inside `group`, outermost first."""
    try:
        children = list(group)
    except Exception:  # noqa: BLE001
        return
    for layer in children:
        if getattr(layer, "kind", None) == "type":
            yield layer
    for layer in children:
        if getattr(layer, "is_group", None) and layer.is_group():
            for nested in _first_type_layer(layer):
                yield nested


def refresh_flattened_preview(psd_path) -> bool:
    """Rebuild the PSD's stored flattened composite from its own layers.

    For a file this app has just edited layer by layer -- a saved
    template with new words typed into it -- the merged snapshot
    Photoshop wrote is now a picture of the OLD document. Everything
    that reads a PSD the quick way reads that snapshot: Finder, Preview,
    and Pillow's Image.open(), which is how this app itself loads a
    template to render on. So a template that had been correctly
    rewritten still rendered with its old copy underneath, and the new
    words drawn into a box sized for them, on top of placeholder text
    that should have been gone.

    Composited with force=True so the layers are actually drawn rather
    than the stale snapshot handed straight back. Best-effort.
    """
    try:
        from psd_tools import PSDImage
    except ImportError:
        return False
    try:
        psd = PSDImage.open(psd_path)
        # Type layers are left out of the composite and pasted back from
        # their own pictures afterwards. Forced to draw a type layer,
        # psd-tools typesets it itself, in a fallback font at a default
        # size -- a fair attempt, and nothing like what Photoshop laid
        # out or what this app just drew into the layer's raster. The
        # raster is the truth here.
        flat = psd.composite(
            force=True,
            layer_filter=lambda layer: layer.is_visible() and layer.kind != "type",
        ).convert("RGBA")
        for layer in psd.descendants():
            if layer.kind != "type" or not layer.is_visible():
                continue
            try:
                picture = layer.topil()
            except Exception:  # noqa: BLE001
                continue
            if picture is None:
                continue
            flat.alpha_composite(picture.convert("RGBA"), dest=(layer.left, layer.top))
    except Exception:  # noqa: BLE001
        return False
    return set_flattened_preview(psd_path, flat)


def set_flattened_preview(psd_path, image) -> bool:
    """Replace the PSD's stored flattened composite with `image`.

    Every PSD carries a merged snapshot of the document alongside its
    layers -- what Finder, Preview, quick-look and psd-tools show, and
    what "maximize compatibility" writes. Photoshop ignores it and
    redraws from the layers, so it is a picture of the file rather than
    the file itself; but the two disagreeing is what makes a correctly
    edited PSD look untouched everywhere except Photoshop. Retyped text
    and a restyled shape are exactly that case: the layer data is new and
    the cached snapshot is the template's.

    So the snapshot is replaced with this render. The layers stay live
    and editable; what changes is only what a viewer sees before opening
    it properly.

    Returns True when the preview was written. Best-effort, like the rest
    of this module.
    """
    try:
        from psd_tools import PSDImage
    except ImportError:
        return False
    try:
        psd = PSDImage.open(psd_path)
    except Exception:  # noqa: BLE001
        return False
    try:
        header = psd._record.header
        flat = image.convert("RGB")
        if flat.size != (header.width, header.height):
            flat = flat.resize((header.width, header.height), Image.LANCZOS)
        channels = [band.tobytes() for band in flat.split()]
        # A 4-channel document wants an alpha plane too; the composite is
        # opaque, so it is a solid one.
        while len(channels) < header.channels:
            channels.append(
                Image.new("L", (header.width, header.height), 255).tobytes()
            )
        psd._record.image_data.set_data(channels[: header.channels], header)
        psd.save(psd_path)
    except Exception:  # noqa: BLE001
        return False
    return True


def replace_pixel_layers(psd_path, images: dict) -> list:
    """Swap named pixel layers' artwork in the PSD at `psd_path`, in
    place, leaving every other layer -- type layers, groups, the CTA's
    live shape -- exactly as it was.

    `images` maps a lowercased layer name to a full-canvas RGBA image at
    the PSD's own size. A name that isn't a top-level pixel layer in the
    file is skipped.

    This is what turns the source-template download from a copy of the
    template into a copy of THIS creative. The file's whole value is that
    it still has editable text and an editable button, so the artwork
    cannot be baked in by flattening -- each pixel layer is removed and
    rebuilt from the new image at the same height in the stack, which is
    the only way psd-tools can write pixels at all (it can create layers
    and nothing else).

    Returns the names of the layers actually replaced. Best-effort: a
    file that won't open, or a layer that won't rebuild, returns [] and
    leaves the file untouched rather than raising.
    """
    try:
        from psd_tools import PSDImage
    except ImportError:
        return []
    try:
        psd = PSDImage.open(psd_path)
    except Exception:  # noqa: BLE001
        return []

    wanted = {
        name.strip().lower(): image for name, image in (images or {}).items() if image
    }
    if not wanted:
        return []

    canvas = (psd.width, psd.height)
    replaced = []
    try:
        for name in list(wanted):
            index = None
            original = None
            for position, layer in enumerate(psd):
                if (layer.name or "").strip().lower() == name and layer.kind == "pixel":
                    index, original = position, layer
                    break
            if original is None:
                continue
            image = wanted[name]
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            if image.size != canvas:
                image = image.resize(canvas, Image.LANCZOS)
            cropped, left, top = _tight_bbox_crop(image)
            psd.remove(original)
            rebuilt = psd.create_pixel_layer(
                cropped, name=original.name, top=top, left=left
            )
            # create_pixel_layer appends; the layer has to go back where
            # the one it replaces was, or a background lands on top of
            # everything it is supposed to sit under.
            if rebuilt is not None:
                psd.remove(rebuilt)
                psd.insert(index, rebuilt)
            replaced.append(original.name)
    except Exception:  # noqa: BLE001
        return []

    if not replaced:
        return []
    try:
        psd.save(psd_path)
    except Exception:  # noqa: BLE001
        return []
    return replaced


def pair_type_layers_with_pixels(psd_path, images: dict, prefer: str = "text") -> list:
    """Put the rendered words in as pixels, and keep the editable text
    beside them switched off.

    `images` maps a lowercased layer (or group) name to a full-canvas
    RGBA image at the PSD's own size -- the words as the renderer drew
    them, colour, glow and all.

    Both versions of every text layer end up in the file, and `prefer`
    decides which one is switched on:

        "text"    the live type layer keeps the plain name and shows;
                  the drawn words sit beside it as "<name> (rendered)",
                  hidden. Editable on open, at the cost of trusting
                  Photoshop to recompose the layer from its text.
        "pixels"  the drawn words take the plain name and show; the type
                  layer becomes "<name> (editable text)", hidden.
                  Guaranteed to look right, one click from editable.

    Why the choice exists at all: a type layer holds its text, a cached
    picture of that text, and the engine data Photoshop lays out from.
    This module writes all three and psd-tools reads all three back
    correctly -- and a file can still open showing the words it started
    with, because when Photoshop recomposes a type layer is Photoshop's
    decision, not the file's. Pixels have no such argument in them. So
    neither answer is right for everyone, and both are in the file
    either way: whichever is hidden is one click from being the one you
    see.

    Returns the names of the layers paired.
    """
    try:
        from psd_tools import PSDImage
        from psd_tools.api.layers import PixelLayer
    except ImportError:
        return []
    try:
        psd = PSDImage.open(psd_path)
    except Exception:  # noqa: BLE001
        return []

    canvas = (psd.width, psd.height)
    replaced = []
    for name, image in (images or {}).items():
        key = name.strip().lower()
        # Resolved per layer rather than once up front: every insertion
        # renumbers the container, so the map would go stale.
        target = _named_type_layers(psd).get(key)
        if target is None or image is None:
            continue
        try:
            parent = target._parent or psd
            index = list(parent).index(target)

            rgba = image if image.mode == "RGBA" else image.convert("RGBA")
            if rgba.size != canvas:
                rgba = rgba.resize(canvas, Image.LANCZOS)
            cropped, left, top = _tight_bbox_crop(rgba)
            if cropped.width < 1 or cropped.height < 1:
                continue

            drawn_name = target.name
            # One of the pair keeps the plain name and is the one that
            # shows; the other is suffixed and switched off. Renamed
            # BEFORE the new layer goes in, so the two never share a
            # name -- _named_type_layers() would otherwise find the
            # wrong one on the next pass.
            live_wins = prefer != "pixels"
            if live_wins:
                target.visible = True
                pixel_name = f"{drawn_name} (rendered)"
            else:
                target.name = f"{drawn_name} (editable text)"
                target.visible = False
                pixel_name = drawn_name

            pixels = PixelLayer.frompil(
                cropped, psd, name=pixel_name, top=top, left=left
            )
            pixels.visible = not live_wins
            # frompil appends to the document; it belongs directly above
            # the text it stands in for, inside whatever group that is.
            if pixels in list(psd):
                psd.remove(pixels)
            parent.insert(index + 1, pixels)
        except Exception:  # noqa: BLE001
            continue
        replaced.append(drawn_name)

    if not replaced:
        return []
    try:
        psd.save(psd_path)
    except Exception:  # noqa: BLE001
        return []
    return replaced


def set_type_layer_raster(psd_path, images: dict) -> list:
    """Replace the cached picture Photoshop shows for a live type layer,
    without touching the text itself.

    `images` maps a lowercased layer (or group) name to a full-canvas
    RGBA image at the PSD's own size -- normally the words exactly as the
    renderer drew them, glow and all.

    Why a type layer needs this at all: alongside its text, Photoshop
    stores a rasterized copy of how that text last looked, and that is
    what it puts on screen when the file opens. Rewriting the string
    updates what the layer SAYS; the picture beside it still shows the
    old words until something makes Photoshop recompose. So a live-text
    download whose copy had been correctly replaced still opened reading
    the template's placeholder text -- while every pixel layer next to it
    (the logo, the product shot) updated immediately, because a pixel
    layer is nothing but its picture.

    The layer stays a type layer: only its channels and their bounds are
    rebuilt, and every tagged block -- the text engine data, the effects,
    the warp -- is left exactly as it was. Click into it in Photoshop and
    it recomposes from the text, which now matches what it was already
    showing.

    Returns the names of the layers whose raster was replaced.
    """
    try:
        from psd_tools import PSDImage
        from psd_tools.constants import ChannelID, Compression
        from psd_tools.psd.layer_and_mask import ChannelData, ChannelInfo
    except ImportError:
        return []
    try:
        psd = PSDImage.open(psd_path)
    except Exception:  # noqa: BLE001
        return []

    layers = _named_type_layers(psd)
    canvas = (psd.width, psd.height)
    replaced = []
    for name, image in (images or {}).items():
        layer = layers.get(name.strip().lower())
        if layer is None or image is None:
            continue
        try:
            rgba = image if image.mode == "RGBA" else image.convert("RGBA")
            if rgba.size != canvas:
                rgba = rgba.resize(canvas, Image.LANCZOS)
            # The picture is cut to the layer's EXISTING box, not to the
            # new words' tight extent. That box is the designer's text
            # box, and it is what every later run fits its copy into --
            # shrink it to a short line's outline and the next run
            # squeezes into that, and the one after into less again,
            # until the description is a whisker. Words that fall outside
            # it were never going to fit the design anyway.
            box = tuple(int(v) for v in layer.bbox)
            _tight, tight_left, tight_top = _tight_bbox_crop(rgba)
            tight_box = (
                tight_left, tight_top,
                tight_left + _tight.width, tight_top + _tight.height,
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                box = tight_box
            elif _tight.width and _tight.height:
                # The union: never smaller than the designed box, never
                # clipping words that reach past it.
                box = (
                    min(box[0], tight_box[0]), min(box[1], tight_box[1]),
                    max(box[2], tight_box[2]), max(box[3], tight_box[3]),
                )
            cropped = rgba.crop(box)
            left, top = box[0], box[1]
            if cropped.width < 1 or cropped.height < 1:
                continue

            record = layer._record
            record.channel_info = []
            # ChannelDataList is a list subclass without .clear() in
            # psd-tools 1.18, so it is emptied by slice assignment.
            channels = layer._channels
            channels[:] = []

            width, height = cropped.width, cropped.height
            version = psd._record.header.version

            alpha = ChannelData(Compression.RLE)
            alpha.set_data(cropped.getchannel("A").tobytes(), width, height, 8, version)
            record.channel_info.append(
                ChannelInfo(ChannelID.TRANSPARENCY_MASK, len(alpha.data) + 2)
            )
            channels.append(alpha)

            for index, band in enumerate(("R", "G", "B")):
                data = ChannelData(Compression.RLE)
                data.set_data(
                    cropped.getchannel(band).tobytes(), width, height, 8, version
                )
                record.channel_info.append(ChannelInfo(ChannelID(index), len(data.data) + 2))
                channels.append(data)

            record.top, record.left = top, left
            record.bottom, record.right = top + height, left + width
        except Exception:  # noqa: BLE001
            continue
        replaced.append(layer.name)

    if not replaced:
        return []
    try:
        psd.save(psd_path)
    except Exception:  # noqa: BLE001
        return []
    return replaced


def _named_shape_layers(node) -> dict:
    """_named_type_layers()'s counterpart for vector shape layers.

    Same reason it exists: a designer's CTA is a group, and the shape
    holding the button is a child called something like "Rectangle 1",
    not the "cta" the form addresses it by. Both names point at it.
    """
    found = {}

    def visit(container):
        for layer in container:
            name = (layer.name or "").strip().lower()
            if getattr(layer, "kind", None) == "shape":
                found.setdefault(name, layer)
                continue
            if getattr(layer, "is_group", None) and layer.is_group():
                for child in _first_shape_layer(layer):
                    if name:
                        found.setdefault(name, child)
                    break
                visit(layer)

    try:
        visit(node)
    except Exception:  # noqa: BLE001
        pass
    return found


def _first_shape_layer(group):
    """Yield the vector shape layers inside `group`, outermost first."""
    try:
        children = list(group)
    except Exception:  # noqa: BLE001
        return
    for layer in children:
        if getattr(layer, "kind", None) == "shape":
            yield layer
    for layer in children:
        if getattr(layer, "is_group", None) and layer.is_group():
            for nested in _first_shape_layer(layer):
                yield nested


def _shape_block(layer, tag_name):
    """The parsed data of one of `layer`'s tagged blocks, or None.

    Looked up by the tag's *name* rather than by importing the Tag enum,
    so a psd-tools that spells one differently costs a skipped property
    rather than an import error at module load.
    """
    try:
        for key, block in layer.tagged_blocks.items():
            if str(key).rsplit(".", 1)[-1] == tag_name:
                return block.data
    except Exception:  # noqa: BLE001
        return None
    return None


def _write_descriptor_rgb(colour_descriptor, rgb) -> bool:
    """Set an RGB colour descriptor's channels in place.

    The components are psd-tools Double objects and are mutated rather
    than replaced: assigning plain Python floats parses back fine but
    blows up on save, since the writer expects objects that know how to
    serialize themselves. Same trap as set_type_layer_colors().
    """
    red, green, blue = rgb
    try:
        for key, value in ((b"Rd  ", red), (b"Grn ", green), (b"Bl  ", blue)):
            colour_descriptor[key].value = float(value)
    except Exception:  # noqa: BLE001
        return False
    return True


# Bezier's circle constant: the handle length, as a fraction of the
# radius, that makes four cubic curves indistinguishable from a circle.
_KAPPA = 0.5522847498307936


def _round_rectangle_path(closed_path, canvas, radius_px: float) -> bool:
    """Rewrite a 4-knot rectangular path in place as a rounded rectangle.

    Photoshop draws a shape layer from this path. The "live shape" radii
    that the Properties panel reads sit in a different block entirely and
    changing only those rounds the number in the panel, not the corners
    on screen -- so the corners have to be built here, as real bezier
    knots, and the panel told about them separately.

    Knot coordinates are (y, x) as fractions of the document's height and
    width. Each corner becomes two anchors joined by one cubic curve,
    with the straight edges keeping their control points on the anchor.

    Returns False and leaves the path untouched unless it is exactly the
    four-cornered rectangle this knows how to round.
    """
    from psd_tools.psd.vector import ClosedKnotLinked

    if len(closed_path) != 4:
        return False
    ys = sorted({round(k.anchor[0], 6) for k in closed_path})
    xs = sorted({round(k.anchor[1], 6) for k in closed_path})
    if len(ys) != 2 or len(xs) != 2:
        return False

    (y0, y1), (x0, x1) = ys, xs
    height, width = canvas
    # Never more than half the shorter side, or the corners cross over.
    span_y, span_x = (y1 - y0) * height, (x1 - x0) * width
    radius = max(0.0, min(float(radius_px), span_y / 2.0, span_x / 2.0))
    if radius <= 0.5:
        return False
    fy, fx = radius / height, radius / width
    ky, kx = fy * _KAPPA, fx * _KAPPA

    def knot(anchor, preceding=None, leaving=None):
        return ClosedKnotLinked(
            preceding=preceding or anchor, anchor=anchor, leaving=leaving or anchor
        )

    # Clockwise from the top edge's left end. A corner's two anchors sit
    # one radius along each edge; their handles point back into the
    # corner they cut off.
    knots = [
        knot((y0, x0 + fx), preceding=(y0, x0 + fx - kx)),
        knot((y0, x1 - fx), leaving=(y0, x1 - fx + kx)),
        knot((y0 + fy, x1), preceding=(y0 + fy - ky, x1)),
        knot((y1 - fy, x1), leaving=(y1 - fy + ky, x1)),
        knot((y1, x1 - fx), preceding=(y1, x1 - fx + kx)),
        knot((y1, x0 + fx), leaving=(y1, x0 + fx - kx)),
        knot((y1 - fy, x0), preceding=(y1 - fy + ky, x0)),
        knot((y0 + fy, x0), leaving=(y0 + fy - ky, x0)),
    ]
    closed_path[:] = knots
    return True


def _tell_the_shape_panel_its_radii(layer, radius_px: float) -> None:
    """Update the live-shape record so Photoshop's Properties panel
    agrees with the path just written.

    Left alone it would still describe a square-cornered rectangle, and a
    live shape whose recorded geometry contradicts its own path is how a
    file starts opening wrong -- Photoshop is entitled to redraw the path
    from the record. keyOriginType 2 is "rounded rectangle"; the radii
    are in document pixels, the same units as keyOriginShapeBBox
    alongside them.
    """
    from psd_tools.psd.descriptor import Descriptor, Double, Integer

    data = _shape_block(layer, "VECTOR_ORIGINATION_DATA")
    try:
        entries = data[b"keyDescriptorList"]
    except Exception:  # noqa: BLE001
        return
    for entry in entries:
        try:
            entry[b"keyOriginType"] = Integer(2)
            radii = Descriptor(classID=b"radii")
            radii[b"unitValueQuadVersion"] = Integer(1)
            for key in (
                b"topRight",
                b"topLeft",
                b"bottomLeft",
                b"bottomRight",
            ):
                radii[key] = Double(float(radius_px))
            entry[b"keyOriginRRectRadii"] = radii
        except Exception:  # noqa: BLE001
            continue


def set_shape_layer_style(psd_path, styles: dict) -> list:
    """Restyle live vector shape layers in the PSD at `psd_path`, in
    place, keeping them editable shapes.

    `styles` maps a lowercased layer (or group) name to any of:

        fill              (r, g, b) -- the shape's fill colour
        stroke_color      (r, g, b) -- the stroke's colour
        stroke_width_pct  percentage of the shape's own height, matching
                          how the renderer sizes a CTA border, so one
                          setting reads the same at 160x600 and 1080x1080
        corner_radius_pct percentage of half the shape's height, so 0 is
                          square, 100 a full pill -- the renderer's own
                          scale

    A stroke_width_pct of 0 switches the stroke off; anything above it
    turns the stroke on, since asking for a width is asking to see one.

    These are the properties Photoshop's own shape toolbar edits, stored
    as descriptors on the layer (the fill in the vector stroke *content*
    block, the rest in the vector stroke block), so what comes back is a
    button whose colour and outline can still be changed by clicking it
    -- not a picture of one.

    The corner radius rewrites the path's bezier knots, because that is
    what Photoshop actually draws from -- the live-shape radii beside it
    are only what the Properties panel reads back, and setting those
    alone rounds the number in the panel and nothing on screen. Both are
    written, so the two agree; only a plain four-cornered rectangle can
    be rounded this way, and anything else is left as it is.

    Returns the names of the layers actually restyled. Best-effort, like
    everything else here: an unreadable file, a name that isn't a shape,
    or an unexpectedly shaped descriptor returns [] rather than raising.

    The same caveat as set_type_layer_colors() applies: this rewrites the
    shape's styling, not the rasterized preview Photoshop caches beside
    it. Photoshop redraws the shape on open, so it is right there; a
    viewer that only reads the cached composite may still show the old
    button.
    """
    try:
        from psd_tools import PSDImage
    except ImportError:
        return []
    try:
        psd = PSDImage.open(psd_path)
    except Exception:  # noqa: BLE001
        return []

    layers = _named_shape_layers(psd)
    restyled = []
    for name, spec in (styles or {}).items():
        layer = layers.get(name.strip().lower())
        if layer is None or not spec:
            continue
        touched = False

        fill = spec.get("fill")
        if fill is not None:
            content = _shape_block(layer, "VECTOR_STROKE_CONTENT_DATA")
            try:
                colour = content[b"Clr "]
            except Exception:  # noqa: BLE001
                colour = None
            if colour is not None and _write_descriptor_rgb(colour, fill):
                touched = True

        radius_pct = spec.get("corner_radius_pct")
        if radius_pct is not None:
            try:
                top, bottom = layer.bbox[1], layer.bbox[3]
                height = max(0, bottom - top)
                # The renderer's rule: a percentage of half the height,
                # so 100 is a pill and 0 square corners.
                radius_px = (height / 2.0) * (max(0, min(100, radius_pct)) / 100.0)
                vector = _shape_block(layer, "VECTOR_MASK_SETTING2")
                closed = None
                for record in (vector.path if vector is not None else []):
                    if hasattr(record, "is_closed") and record.is_closed():
                        closed = record
                        break
                if closed is not None and _round_rectangle_path(
                    closed, (psd.height, psd.width), radius_px
                ):
                    _tell_the_shape_panel_its_radii(layer, radius_px)
                    touched = True
            except Exception:  # noqa: BLE001
                pass

        stroke = _shape_block(layer, "VECTOR_STROKE_DATA")
        if stroke is not None:
            stroke_colour = spec.get("stroke_color")
            if stroke_colour is not None:
                try:
                    colour = stroke[b"strokeStyleContent"][b"Clr "]
                except Exception:  # noqa: BLE001
                    colour = None
                if colour is not None and _write_descriptor_rgb(colour, stroke_colour):
                    touched = True

            pct = spec.get("stroke_width_pct")
            if pct is not None:
                try:
                    top, bottom = layer.bbox[1], layer.bbox[3]
                    height = max(0, bottom - top)
                    # The renderer's own rule (apply_layer_cta_override):
                    # a percentage of the button's height, floored at one
                    # pixel so a small percentage is still visible.
                    width_px = 0.0
                    if pct > 0 and height:
                        width_px = float(max(1, round(height * (pct / 100.0))))
                    stroke[b"strokeStyleLineWidth"].value = width_px
                    stroke[b"strokeEnabled"].value = width_px > 0
                    touched = True
                except Exception:  # noqa: BLE001
                    pass

        if touched:
            restyled.append(layer.name)

    if not restyled:
        return []
    try:
        psd.save(psd_path)
    except Exception:  # noqa: BLE001
        return []
    return restyled


def _rgb_descriptor(rgb):
    """An RGBC colour descriptor, the shape Photoshop stores colours in
    inside a layer effect."""
    from psd_tools.psd.descriptor import Descriptor, Double

    colour = Descriptor(classID=b"RGBC")
    red, green, blue = rgb
    colour[b"Rd  "] = Double(float(red))
    colour[b"Grn "] = Double(float(green))
    colour[b"Bl  "] = Double(float(blue))
    return colour


def _outer_glow_descriptor(color, radius_px: float, opacity: int):
    """Photoshop's Outer Glow, as its `OrGl` descriptor.

    Screen blend, soft technique -- the same defaults the fx dialog
    starts from, which is what makes the result recognisable as "a glow"
    rather than something odd that happens to have the right colour.
    """
    from psd_tools.psd.descriptor import Bool, Descriptor, Enumerated, UnitFloat
    from psd_tools.terminology import Unit

    glow = Descriptor(classID=b"OrGl")
    glow[b"enab"] = Bool(True)
    glow[b"present"] = Bool(True)
    glow[b"showInDialog"] = Bool(True)
    glow[b"Md  "] = Enumerated(b"BlnM", b"Scrn")
    glow[b"Clr "] = _rgb_descriptor(color)
    glow[b"Opct"] = UnitFloat(float(max(0, min(100, opacity))), Unit.Percent)
    glow[b"GlwT"] = Enumerated(b"BETE", b"SfBL")
    glow[b"Ckmt"] = UnitFloat(0.0, Unit.Pixels)
    glow[b"blur"] = UnitFloat(float(max(1.0, radius_px)), Unit.Pixels)
    glow[b"Nose"] = UnitFloat(0.0, Unit.Percent)
    glow[b"ShdN"] = UnitFloat(0.0, Unit.Percent)
    glow[b"AntA"] = Bool(False)
    glow[b"Inpr"] = UnitFloat(50.0, Unit.Percent)
    return glow


def _stroke_descriptor(color, size_px: float):
    """Photoshop's Stroke effect, as its `FrFX` descriptor -- outside the
    letterforms, solid colour, which is what the renderer draws."""
    from psd_tools.psd.descriptor import Bool, Descriptor, Enumerated, UnitFloat
    from psd_tools.terminology import Unit

    stroke = Descriptor(classID=b"FrFX")
    stroke[b"enab"] = Bool(True)
    stroke[b"present"] = Bool(True)
    stroke[b"showInDialog"] = Bool(True)
    stroke[b"Styl"] = Enumerated(b"FStl", b"OutF")
    stroke[b"PntT"] = Enumerated(b"FrFl", b"SClr")
    stroke[b"Md  "] = Enumerated(b"BlnM", b"Nrml")
    stroke[b"Opct"] = UnitFloat(100.0, Unit.Percent)
    stroke[b"Sz  "] = UnitFloat(float(max(1.0, size_px)), Unit.Pixels)
    stroke[b"Clr "] = _rgb_descriptor(color)
    return stroke


def set_type_layer_effects(psd_path, effects: dict) -> list:
    """Give live type layers a real Photoshop glow and/or stroke.

    `effects` maps a lowercased layer (or group) name to a dict with any
    of:

        glow    {"color": (r, g, b), "radius": px, "opacity": 0-100}
        stroke  {"color": (r, g, b), "size": px}

    Both sizes are in document pixels, already resolved by the caller
    against the font size the renderer actually laid the words out at.
    They cannot be worked out here: a type layer keeps its point size in
    the document's resource defaults scaled by the layer's own transform,
    so there is nothing on the layer to take the form's percentage of,
    and estimating it from the text box overshoots badly whenever the
    designer drew a box taller than the words in it.

    A layer named with neither is left alone; passing an empty dict for a
    layer clears nothing, it simply does nothing.

    Why this exists: colour, size and weight live inside the type layer's
    own text engine data and can be set there, but a glow is not text
    styling at all in Photoshop -- it is a layer *effect*, a separate
    structure (`lfx2`) hanging off the layer. So live text that the
    renderer had drawn with a green glow arrived in the editable download
    as flat green words, and the file no longer looked like the creative
    it came from.

    Written only into the live-text download, never the layered one: the
    layered file is pixels throughout and already correct, so if a
    Photoshop version disagrees with anything authored here, the file
    that always opens right is untouched.

    Returns the names of the layers actually given effects. Best-effort
    like the rest of this module.
    """
    try:
        from psd_tools import PSDImage
        from psd_tools.constants import Tag
        from psd_tools.psd.descriptor import Bool, DescriptorBlock2, UnitFloat
        from psd_tools.terminology import Unit
    except ImportError:
        return []
    try:
        psd = PSDImage.open(psd_path)
    except Exception:  # noqa: BLE001
        return []

    layers = _named_type_layers(psd)
    styled = []
    for name, spec in (effects or {}).items():
        layer = layers.get(name.strip().lower())
        if layer is None or not spec:
            continue
        glow = spec.get("glow")
        stroke = spec.get("stroke")
        if not glow and not stroke:
            continue
        try:
            block = DescriptorBlock2(classID=b"null", version=0, data_version=16)
            block[b"masterFXSwitch"] = Bool(True)
            # The effects' own scale. 100% means "the sizes below are in
            # the document's pixels", which is what the renderer measured
            # them in.
            block[b"Scl "] = UnitFloat(100.0, Unit.Percent)
            if glow:
                block[b"OrGl"] = _outer_glow_descriptor(
                    glow.get("color", (255, 255, 255)),
                    glow.get("radius", 8),
                    glow.get("opacity", 100),
                )
            if stroke:
                block[b"FrFX"] = _stroke_descriptor(
                    stroke.get("color", (0, 0, 0)), stroke.get("size", 1)
                )
            layer.tagged_blocks.set_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, block)
        except Exception:  # noqa: BLE001
            continue
        styled.append(layer.name)

    if not styled:
        return []
    try:
        psd.save(psd_path)
    except Exception:  # noqa: BLE001
        return []
    return styled


def set_type_layer_text(psd_path, texts: dict) -> list:
    """Rewrite live Photoshop type layers' copy in the PSD at `psd_path`,
    in place, keeping them editable text.

    `texts` maps a lowercased layer name to the words that layer should
    end up saying; a None or empty value leaves that layer alone, so a
    form field nobody filled in keeps the template's own copy rather than
    emptying it.

    This is the counterpart to set_type_layer_colors() for the source
    template PSD that ships beside every render -- the one file in the
    download that still has the CTA as a live group and every text layer
    editable. It was going out as a straight copy of the template, so it
    read back the template's placeholder copy no matter what had been
    typed into the form: the exact file someone opens to edit the words
    was the one file that did not have them.

    Returns the names of the layers actually rewritten -- best-effort,
    like everything else here: an unreadable file or an unexpectedly
    shaped type layer returns [] instead of raising.
    """
    try:
        from psd_tools import PSDImage
    except ImportError:
        return []
    try:
        psd = PSDImage.open(psd_path)
    except Exception:  # noqa: BLE001
        return []

    layers = _named_type_layers(psd)
    rewritten = []
    for name, text in (texts or {}).items():
        if not text:
            continue
        layer = layers.get(name.strip().lower())
        if layer is None:
            continue
        if _rewrite_type_layer_text(layer, text):
            rewritten.append(layer.name)

    if not rewritten:
        return []
    try:
        psd.save(psd_path)
    except Exception:  # noqa: BLE001
        return []
    return rewritten


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
    # Groups included -- see _named_type_layers(). A CTA's label is a
    # child of the group the form calls "cta", and a flat scan of the
    # document's top level never reaches it.
    layers = _named_type_layers(psd)
    recoloured = []
    for name, rgb in wanted.items():
        layer = layers.get(name)
        if layer is None:
            continue
        red, green, blue = rgb
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
