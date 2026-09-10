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

import copy

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
            if original is None and name != "background":
                continue
            image = wanted[name]
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            if image.size != canvas:
                image = image.resize(canvas, Image.LANCZOS)
            cropped, left, top = _tight_bbox_crop(image)
            if original is None:
                # No background layer in the file (deleted rather than
                # emptied): the hero goes in as one, at the bottom of
                # the stack, under everything.
                rebuilt = psd.create_pixel_layer(cropped, name="background", top=top, left=left)
                if rebuilt is not None:
                    psd.remove(rebuilt)
                    psd.insert(0, rebuilt)
                    replaced.append("background")
                continue
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

            # An earlier export's companion of the same name goes: a
            # template that is re-uploaded run after run was collecting
            # a "(rendered)" layer per run.
            for stale in [l for l in list(parent) if l is not target and l.name == pixel_name]:
                try:
                    parent.remove(stale)
                except Exception:  # noqa: BLE001
                    pass
            index = list(parent).index(target)
            pixels = PixelLayer.frompil(
                cropped, psd, name=pixel_name, top=top, left=left
            )
            pixels.visible = not live_wins
            # The flag every Photoshop layer carries (see
            # set_type_layer_effects); a layer without it shows no
            # Effects in Photoshop, whatever it is later given.
            try:
                pixels._record.flags.undocumented_1 = True
            except Exception:  # noqa: BLE001
                pass
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


def set_layer_visibility(psd_path, visibility: dict) -> list:
    """Switch named top-level layers on or off in place: `visibility`
    maps a lowercased layer name to True/False. Returns the names
    changed. A layer hidden on the form is left out of the preview, so
    the editable file has to open the same way -- with that layer's eye
    off, not gone."""
    try:
        psd = PSDImage.open(psd_path)
    except Exception:  # noqa: BLE001
        return []
    wanted = {k.strip().lower(): bool(v) for k, v in (visibility or {}).items()}
    changed = []
    for layer in psd:
        key = (layer.name or "").strip().lower()
        if key in wanted and layer.visible != wanted[key]:
            layer.visible = wanted[key]
            changed.append(key)
    if changed:
        try:
            psd.save(psd_path)
        except Exception:  # noqa: BLE001
            return []
    return changed


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
            radii = _ps_descriptor(b"radii")
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


# A complete, empty layer-effects block exactly as Photoshop writes one:
# every effect present in the structure and switched off (drop shadow,
# inner shadow, outer glow, colour/gradient overlays, stroke, inner
# glow, bevel, satin), version 0, the null class names, `Scl ` 100%.
# Lifted from a layer Photoshop saved, then disabled. A block built from
# scratch with just the one effect in it -- `Scl `, the switch, `OrGl`
# -- was silently ignored by Photoshop: the logo came in with no
# Effects at all, while a layer whose block came from Photoshop kept
# them. The base is decoded on demand and the form's effects set into it.
_PHOTOSHOP_EMPTY_LFX2_B64 = (
    "AAAAAAAAABAAAAABAAAAAAAAbnVsbAAAAAwAAAAAU2NsIFVudEYjUHJjQFkAAAAAAAAAAAAObWFzdGVyRlhTd2l0Y2hib29sAQAA"
    "AA9kcm9wU2hhZG93TXVsdGlWbExzAAAAAU9iamMAAAABAAAAAAAARHJTaAAAAA8AAAAAZW5hYmJvb2wAAAAAB3ByZXNlbnRib29s"
    "AAAAAAxzaG93SW5EaWFsb2dib29sAQAAAABNZCAgZW51bQAAAABCbG5NAAAACG11bHRpcGx5AAAAAENsciBPYmpjAAAAAQAAAAAA"
    "AFJHQkMAAAADAAAAAFJkICBkb3ViAAAAAAAAAAAAAAAAR3JuIGRvdWIAAAAAAAAAAAAAAABCbCAgZG91YgAAAAAAAAAAAAAAAE9w"
    "Y3RVbnRGI1ByY0BBgAAAAAAAAAAAAHVnbGdib29sAQAAAABsYWdsVW50RiNBbmdAVoAAAAAAAAAAAABEc3RuVW50RiNQeGxACAAA"
    "AAAAAAAAAABDa210VW50RiNQeGwAAAAAAAAAAAAAAABibHVyVW50RiNQeGxAMAAAAAAAAAAAAABOb3NlVW50RiNQcmMAAAAAAAAA"
    "AAAAAABBbnRBYm9vbAAAAAAAVHJuU09iamMAAAABAAAAAAAAU2hwQwAAAAIAAAAATm0gIFRFWFQAAAAHAEwAaQBuAGUAYQByAAAA"
    "AAAAQ3J2IFZsTHMAAAACT2JqYwAAAAEAAAAAAABDclB0AAAAAgAAAABIcnpuZG91YgAAAAAAAAAAAAAAAFZydGNkb3ViAAAAAAAA"
    "AABPYmpjAAAAAQAAAAAAAENyUHQAAAACAAAAAEhyem5kb3ViQG/gAAAAAAAAAAAAVnJ0Y2RvdWJAb+AAAAAAAAAAAA1sYXllckNv"
    "bmNlYWxzYm9vbAEAAAAQaW5uZXJTaGFkb3dNdWx0aVZsTHMAAAABT2JqYwAAAAEAAAAAAABJclNoAAAADgAAAABlbmFiYm9vbAAA"
    "AAAHcHJlc2VudGJvb2wAAAAADHNob3dJbkRpYWxvZ2Jvb2wBAAAAAE1kICBlbnVtAAAAAEJsbk0AAAAIbXVsdGlwbHkAAAAAQ2xy"
    "IE9iamMAAAABAAAAAAAAUkdCQwAAAAMAAAAAUmQgIGRvdWIAAAAAAAAAAAAAAABHcm4gZG91YgAAAAAAAAAAAAAAAEJsICBkb3Vi"
    "AAAAAAAAAAAAAAAAT3BjdFVudEYjUHJjQEGAAAAAAAAAAAAAdWdsZ2Jvb2wBAAAAAGxhZ2xVbnRGI0FuZ0BWgAAAAAAAAAAAAERz"
    "dG5VbnRGI1B4bEAIAAAAAAAAAAAAAENrbXRVbnRGI1B4bAAAAAAAAAAAAAAAAGJsdXJVbnRGI1B4bEAcAAAAAAAAAAAAAE5vc2VV"
    "bnRGI1ByYwAAAAAAAAAAAAAAAEFudEFib29sAAAAAABUcm5TT2JqYwAAAAEAAAAAAABTaHBDAAAAAgAAAABObSAgVEVYVAAAAAcA"
    "TABpAG4AZQBhAHIAAAAAAABDcnYgVmxMcwAAAAJPYmpjAAAAAQAAAAAAAENyUHQAAAACAAAAAEhyem5kb3ViAAAAAAAAAAAAAAAA"
    "VnJ0Y2RvdWIAAAAAAAAAAE9iamMAAAABAAAAAAAAQ3JQdAAAAAIAAAAASHJ6bmRvdWJAb+AAAAAAAAAAAABWcnRjZG91YkBv4AAA"
    "AAAAAAAAAE9yR2xPYmpjAAAAAQAAAAAAAE9yR2wAAAAOAAAAAGVuYWJib29sAAAAAAdwcmVzZW50Ym9vbAAAAAAMc2hvd0luRGlh"
    "bG9nYm9vbAEAAAAATWQgIGVudW0AAAAAQmxuTQAAAAZub3JtYWwAAAAAQ2xyIE9iamMAAAABAAAAAAAAUkdCQwAAAAMAAAAAUmQg"
    "IGRvdWJAb+AAAAAAAAAAAABHcm4gZG91YkBv4AAAAAAAAAAAAEJsICBkb3ViQG/gAAAAAAAAAAAAT3BjdFVudEYjUHJjQFCAAAAA"
    "AAAAAAAAR2x3VGVudW0AAAAAQkVURQAAAABTZkJMAAAAAENrbXRVbnRGI1B4bEBRAAAAAAAAAAAAAGJsdXJVbnRGI1B4bEAAAAAA"
    "AAAAAAAAAE5vc2VVbnRGI1ByYwAAAAAAAAAAAAAAAFNoZE5VbnRGI1ByYwAAAAAAAAAAAAAAAEFudEFib29sAAAAAABUcm5TT2Jq"
    "YwAAAAEAAAAAAABTaHBDAAAAAgAAAABObSAgVEVYVAAAAAcATABpAG4AZQBhAHIAAAAAAABDcnYgVmxMcwAAAAJPYmpjAAAAAQAA"
    "AAAAAENyUHQAAAACAAAAAEhyem5kb3ViAAAAAAAAAAAAAAAAVnJ0Y2RvdWIAAAAAAAAAAE9iamMAAAABAAAAAAAAQ3JQdAAAAAIA"
    "AAAASHJ6bmRvdWJAb+AAAAAAAAAAAABWcnRjZG91YkBv4AAAAAAAAAAAAElucHJVbnRGI1ByYz/wAAAAAAAAAAAADnNvbGlkRmls"
    "bE11bHRpVmxMcwAAAAFPYmpjAAAAAQAAAAAAAFNvRmkAAAAGAAAAAGVuYWJib29sAAAAAAdwcmVzZW50Ym9vbAAAAAAMc2hvd0lu"
    "RGlhbG9nYm9vbAEAAAAATWQgIGVudW0AAAAAQmxuTQAAAAZub3JtYWwAAAAAQ2xyIE9iamMAAAABAAAAAAAAUkdCQwAAAAMAAAAA"
    "UmQgIGRvdWJAYCAPwAAAAAAAAABHcm4gZG91YkBgIA/AAAAAAAAAAEJsICBkb3ViQGAgD8AAAAAAAAAAT3BjdFVudEYjUHJjQFkA"
    "AAAAAAAAAAARZ3JhZGllbnRGaWxsTXVsdGlWbExzAAAAAU9iamMAAAABAAAAAAAAR3JGbAAAAA4AAAAAZW5hYmJvb2wAAAAAB3By"
    "ZXNlbnRib29sAAAAAAxzaG93SW5EaWFsb2dib29sAQAAAABNZCAgZW51bQAAAABCbG5NAAAABm5vcm1hbAAAAABPcGN0VW50RiNQ"
    "cmNAWQAAAAAAAAAAAABHcmFkT2JqYwAAAAkARwByAGEAZABpAGUAbgB0AAAAAAAAR3JkbgAAAAUAAAAATm0gIFRFWFQAAAAMAEcA"
    "cgBhAHkALAAgAFcAaABpAHQAZQAAAAAAAEdyZEZlbnVtAAAAAEdyZEYAAAAAQ3N0UwAAAABJbnRyZG91YkCwAAAAAAAAAAAAAENs"
    "cnNWbExzAAAAAk9iamMAAAABAAAAAAAAQ2xydAAAAAQAAAAAQ2xyIE9iamMAAAABAAAAAAAAUkdCQwAAAAMAAAAAUmQgIGRvdWJA"
    "auAFAAAAAAAAAABHcm4gZG91YkBq4AUAAAAAAAAAAEJsICBkb3ViQGrgBQAAAAAAAAAAVHlwZWVudW0AAAAAQ2xyeQAAAABVc3JT"
    "AAAAAExjdG5sb25nAAAAAAAAAABNZHBubG9uZwAAADJPYmpjAAAAAQAAAAAAAENscnQAAAAEAAAAAENsciBPYmpjAAAAAQAAAAAA"
    "AFJHQkMAAAADAAAAAFJkICBkb3ViQG/gAAAAAAAAAAAAR3JuIGRvdWJAb+AAAAAAAAAAAABCbCAgZG91YkBv4AAAAAAAAAAAAFR5"
    "cGVlbnVtAAAAAENscnkAAAAAVXNyUwAAAABMY3RubG9uZwAAEAAAAAAATWRwbmxvbmcAAAAyAAAAAFRybnNWbExzAAAAAk9iamMA"
    "AAABAAAAAAAAVHJuUwAAAAMAAAAAT3BjdFVudEYjUHJjQFkAAAAAAAAAAAAATGN0bmxvbmcAAAAAAAAAAE1kcG5sb25nAAAAMk9i"
    "amMAAAABAAAAAAAAVHJuUwAAAAMAAAAAT3BjdFVudEYjUHJjQFkAAAAAAAAAAAAATGN0bmxvbmcAABAAAAAAAE1kcG5sb25nAAAA"
    "MgAAAABBbmdsVW50RiNBbmdAVoAAAAAAAAAAAABUeXBlZW51bQAAAABHcmRUAAAAAExuciAAAAAAUnZyc2Jvb2wAAAAAAER0aHJi"
    "b29sAAAAAABnczk5ZW51bQAAAB9ncmFkaWVudEludGVycG9sYXRpb25NZXRob2RUeXBlAAAAAFNtb28AAAAAQWxnbmJvb2wBAAAA"
    "AFNjbCBVbnRGI1ByY0BZAAAAAAAAAAAAAE9mc3RPYmpjAAAAAQAAAAAAAFBudCAAAAACAAAAAEhyem5VbnRGI1ByYwAAAAAAAAAA"
    "AAAAAFZydGNVbnRGI1ByYwAAAAAAAAAAAAAADGZyYW1lRlhNdWx0aVZsTHMAAAABT2JqYwAAAAEAAAAAAABGckZYAAAACgAAAABl"
    "bmFiYm9vbAAAAAAHcHJlc2VudGJvb2wAAAAADHNob3dJbkRpYWxvZ2Jvb2wBAAAAAFN0eWxlbnVtAAAAAEZTdGwAAAAASW5zRgAA"
    "AABQbnRUZW51bQAAAABGckZsAAAAAFNDbHIAAAAATWQgIGVudW0AAAAAQmxuTQAAAAZub3JtYWwAAAAAT3BjdFVudEYjUHJjQFkA"
    "AAAAAAAAAAAAU3ogIFVudEYjUHhsP/AAAAAAAAAAAAAAQ2xyIE9iamMAAAABAAAAAAAAUkdCQwAAAAMAAAAAUmQgIGRvdWIAAAAA"
    "AAAAAAAAAABHcm4gZG91YgAAAAAAAAAAAAAAAEJsICBkb3ViAAAAAAAAAAAAAAAJb3ZlcnByaW50Ym9vbAAAAAAASXJHbE9iamMA"
    "AAABAAAAAAAASXJHbAAAAA8AAAAAZW5hYmJvb2wAAAAAB3ByZXNlbnRib29sAAAAAAxzaG93SW5EaWFsb2dib29sAQAAAABNZCAg"
    "ZW51bQAAAABCbG5NAAAABnNjcmVlbgAAAABDbHIgT2JqYwAAAAEAAAAAAABSR0JDAAAAAwAAAABSZCAgZG91YkBv4AAAAAAAAAAA"
    "AEdybiBkb3ViQG/gAAAAAAAAAAAAQmwgIGRvdWJAb+AAAAAAAAAAAABPcGN0VW50RiNQcmNAQYAAAAAAAAAAAABHbHdUZW51bQAA"
    "AABCRVRFAAAAAFNmQkwAAAAAQ2ttdFVudEYjUHhsAAAAAAAAAAAAAAAAYmx1clVudEYjUHhsQBwAAAAAAAAAAAAATm9zZVVudEYj"
    "UHJjAAAAAAAAAAAAAAAAU2hkTlVudEYjUHJjAAAAAAAAAAAAAAAAQW50QWJvb2wAAAAAAFRyblNPYmpjAAAAAQAAAAAAAFNocEMA"
    "AAACAAAAAE5tICBURVhUAAAABwBMAGkAbgBlAGEAcgAAAAAAAENydiBWbExzAAAAAk9iamMAAAABAAAAAAAAQ3JQdAAAAAIAAAAA"
    "SHJ6bmRvdWIAAAAAAAAAAAAAAABWcnRjZG91YgAAAAAAAAAAT2JqYwAAAAEAAAAAAABDclB0AAAAAgAAAABIcnpuZG91YkBv4AAA"
    "AAAAAAAAAFZydGNkb3ViQG/gAAAAAAAAAAAASW5wclVudEYjUHJjQEkAAAAAAAAAAAAAZ2x3U2VudW0AAAAASUdTcgAAAABTcmNF"
    "AAAAAGViYmxPYmpjAAAAAQAAAAAAAGViYmwAAAAWAAAAAGVuYWJib29sAAAAAAdwcmVzZW50Ym9vbAAAAAAMc2hvd0luRGlhbG9n"
    "Ym9vbAEAAAAAaGdsTWVudW0AAAAAQmxuTQAAAAZzY3JlZW4AAAAAaGdsQ09iamMAAAABAAAAAAAAUkdCQwAAAAMAAAAAUmQgIGRv"
    "dWJAb+AAAAAAAAAAAABHcm4gZG91YkBv4AAAAAAAAAAAAEJsICBkb3ViQG/gAAAAAAAAAAAAaGdsT1VudEYjUHJjQEkAAAAAAAAA"
    "AAAAc2R3TWVudW0AAAAAQmxuTQAAAAhtdWx0aXBseQAAAABzZHdDT2JqYwAAAAEAAAAAAABSR0JDAAAAAwAAAABSZCAgZG91YgAA"
    "AAAAAAAAAAAAAEdybiBkb3ViAAAAAAAAAAAAAAAAQmwgIGRvdWIAAAAAAAAAAAAAAABzZHdPVW50RiNQcmNASQAAAAAAAAAAAABi"
    "dmxUZW51bQAAAABidmxUAAAAAFNmQkwAAAAAYnZsU2VudW0AAAAAQkVTbAAAAABJbnJCAAAAAHVnbGdib29sAQAAAABsYWdsVW50"
    "RiNBbmdAVoAAAAAAAAAAAABMYWxkVW50RiNBbmdAPgAAAAAAAAAAAABzcmdSVW50RiNQcmNAWQAAAAAAAAAAAABibHVyVW50RiNQ"
    "eGxAHAAAAAAAAAAAAABidmxEZW51bQAAAABCRVNzAAAAAEluICAAAAAAVHJuU09iamMAAAABAAAAAAAAU2hwQwAAAAIAAAAATm0g"
    "IFRFWFQAAAAHAEwAaQBuAGUAYQByAAAAAAAAQ3J2IFZsTHMAAAACT2JqYwAAAAEAAAAAAABDclB0AAAAAgAAAABIcnpuZG91YgAA"
    "AAAAAAAAAAAAAFZydGNkb3ViAAAAAAAAAABPYmpjAAAAAQAAAAAAAENyUHQAAAACAAAAAEhyem5kb3ViQG/gAAAAAAAAAAAAVnJ0"
    "Y2RvdWJAb+AAAAAAAAAAAA5hbnRpYWxpYXNHbG9zc2Jvb2wAAAAAAFNmdG5VbnRGI1B4bAAAAAAAAAAAAAAACHVzZVNoYXBlYm9v"
    "bAAAAAAKdXNlVGV4dHVyZWJvb2wAAAAAAENoRlhPYmpjAAAAAQAAAAAAAENoRlgAAAAMAAAAAGVuYWJib29sAAAAAAdwcmVzZW50"
    "Ym9vbAAAAAAMc2hvd0luRGlhbG9nYm9vbAEAAAAATWQgIGVudW0AAAAAQmxuTQAAAAhtdWx0aXBseQAAAABDbHIgT2JqYwAAAAEA"
    "AAAAAABSR0JDAAAAAwAAAABSZCAgZG91YgAAAAAAAAAAAAAAAEdybiBkb3ViAAAAAAAAAAAAAAAAQmwgIGRvdWIAAAAAAAAAAAAA"
    "AABBbnRBYm9vbAEAAAAASW52cmJvb2wBAAAAAE9wY3RVbnRGI1ByY0BJAAAAAAAAAAAAAGxhZ2xVbnRGI0FuZ0BWgAAAAAAAAAAA"
    "AERzdG5VbnRGI1B4bEBJAAAAAAAAAAAAAGJsdXJVbnRGI1B4bEBUAAAAAAAAAAAAAE1wZ1NPYmpjAAAAAQAAAAAAAFNocEMAAAAC"
    "AAAAAE5tICBURVhUAAAABwBMAGkAbgBlAGEAcgAAAAAAAENydiBWbExzAAAAAk9iamMAAAABAAAAAAAAQ3JQdAAAAAIAAAAASHJ6"
    "bmRvdWIAAAAAAAAAAAAAAABWcnRjZG91YgAAAAAAAAAAT2JqYwAAAAEAAAAAAABDclB0AAAAAgAAAABIcnpuZG91YkBv4AAAAAAA"
    "AAAAAFZydGNkb3ViQG/gAAAAAAAAAAAObnVtTW9kaWZ5aW5nRlhsb25nAAAAAAAAAA=="
)


def _empty_effects_block():
    """A fresh lfx2 block for a layer that has none -- see the constant."""
    import base64
    import io

    from psd_tools.psd.descriptor import DescriptorBlock2

    return DescriptorBlock2.read(io.BytesIO(base64.b64decode("".join(_PHOTOSHOP_EMPTY_LFX2_B64))))


# The legacy effects block (`lrFX`) Photoshop writes next to every
# lfx2 -- the same six effects in the pre-CS format, kept in sync with
# the descriptor block. Photoshop still writes it for every layer with
# effects, and a layer carrying lfx2 alone was ignored (the logo opened
# with no Effects while the header, which had both, kept its shadow).
# Lifted from a Photoshop-saved layer with everything off; the drop
# shadow and outer glow in it are set to match what goes into lfx2.
_PHOTOSHOP_EMPTY_LRFX_B64 = (
    "AAAABzhCSU1jbW5TAAAABwAAAAABAAA4QklNZHNkdwAAADMAAAACABAAAAAAAAAAWgAAAAMAAAAAAAAAAAAAAAA4QklNbXVsIAAB"
    "WQAAAAAAAAAAAAA4QklNaXNkdwAAADMAAAACAAcAAAAAAAAAWgAAAAMAAAAAAAAAAAAAAAA4QklNbXVsIAABWQAAAAAAAAAAAAA4"
    "QklNb2dsdwAAACoAAAACAAIAAAAAAAAAAP///////wAAOEJJTW5vcm0AqAAA////////AAA4QklNaWdsdwAAACsAAAACAAcAAAAA"
    "AAAAAP///////wAAOEJJTXNjcm4AWQEAAP///////wAAOEJJTWJldmwAAABOAAAAAgBaAAAABwAAAAcAADhCSU1zY3JuOEJJTW11"
    "bCAAAP///////wAAAAAAAAAAAAAAAAKAgAABAAAA////////AAAAAAAAAAAAAAAAOEJJTXNvZmkAAAAiAAAAAjhCSU1ub3JtAACB"
    "gYGBgYEAAP8AAACBgYGBgYEAAAAA"
)


def _empty_legacy_effects_block():
    import base64
    import io

    from psd_tools.psd.effects_layer import EffectsLayer

    return EffectsLayer.read(io.BytesIO(base64.b64decode("".join(_PHOTOSHOP_EMPTY_LRFX_B64))))


def _sync_legacy_effects(block, shadow, glow):
    """Set lrFX's drop shadow and outer glow from the form's specs.
    Sizes are 16.16 fixed point, colours 16-bit per channel, opacity
    0-255 -- the legacy encoding."""
    from psd_tools.constants import BlendMode
    from psd_tools.psd.color import Color
    from psd_tools.psd.effects_layer import EffectOSType

    def fixed(px):
        return int(round(float(px) * 65536))

    def colour16(rgb):
        r, g, b = rgb
        return Color(values=[int(r) * 257, int(g) * 257, int(b) * 257, 0])

    if shadow:
        item = block.get(EffectOSType.DROP_SHADOW)
        if item is not None:
            item.enabled = 1
            item.blur = fixed(max(0.0, float(shadow.get("size", 5))))
            item.distance = fixed(max(0.0, float(shadow.get("distance", 5))))
            item.angle = fixed(float(shadow.get("angle", 120)))
            item.use_global_angle = 0
            item.opacity = int(round(max(0, min(100, float(shadow.get("opacity", 75)))) * 2.55))
            item.blend_mode = BlendMode.MULTIPLY
            item.color = colour16(shadow.get("color", (0, 0, 0)))
            item.native_color = colour16(shadow.get("color", (0, 0, 0)))
    if glow:
        item = block.get(EffectOSType.OUTER_GLOW)
        if item is not None:
            item.enabled = 1
            item.blur = fixed(max(1.0, float(glow.get("radius", 8))))
            item.opacity = int(round(max(0, min(100, float(glow.get("opacity", 100)))) * 2.55))
            item.blend_mode = BlendMode.SCREEN
            item.color = colour16(glow.get("color", (255, 255, 255)))
            item.native_color = colour16(glow.get("color", (255, 255, 255)))
    return block


def _count_enabled_effects(block) -> int:
    """How many effects in an lfx2 block are switched on -- what
    Photoshop keeps in `numModifyingFX`."""
    count = 0
    for key, value in block.items():
        if key in (b"Scl ", b"masterFXSwitch", b"numModifyingFX"):
            continue
        items = list(value) if type(value).__name__ == "List" else [value]
        for item in items:
            try:
                if item[b"enab"].value:
                    count += 1
            except (KeyError, TypeError, AttributeError):
                continue
    return count


def _ps_descriptor(class_id: bytes):
    """A descriptor the way Photoshop writes one. Its class *name* is
    the null character, not the empty string: Photoshop stores the name
    as a null-terminated Unicode string, so "no name" is one character
    long (\\x00) -- and reading a zero-length name back is one of the
    things that makes it refuse the layer ("not compatible with this
    version of Photoshop"). Every descriptor in a file Photoshop saved
    has name == "\\x00"; every one built here does too."""
    from psd_tools.psd.descriptor import Descriptor

    return Descriptor(name="\x00", classID=class_id)


def _rgb_descriptor(rgb):
    """An RGBC colour descriptor, the shape Photoshop stores colours in
    inside a layer effect."""
    from psd_tools.psd.descriptor import Descriptor, Double

    colour = _ps_descriptor(b"RGBC")
    red, green, blue = rgb
    colour[b"Rd  "] = Double(float(red))
    colour[b"Grn "] = Double(float(green))
    colour[b"Bl  "] = Double(float(blue))
    return colour


def _linear_contour_descriptor():
    """The straight-line contour every effect in a Photoshop-written file
    carries as `TrnS` -- a `ShpC` shape with two curve points. Photoshop
    writes it on every shadow, glow and stroke it saves; a descriptor
    without it is not the shape it expects and it gives up on the layer
    ("not compatible with this version of Photoshop")."""
    from psd_tools.psd.descriptor import Descriptor, Double, List, String

    def point(x, y):
        pt = _ps_descriptor(b"CrPt")
        pt[b"Hrzn"] = Double(float(x))
        pt[b"Vrtc"] = Double(float(y))
        return pt

    contour = _ps_descriptor(b"ShpC")
    contour[b"Nm  "] = String("Linear\x00")
    contour[b"Crv "] = List([point(0, 0), point(255, 255)])
    return contour


def _outer_glow_descriptor(color, radius_px: float, opacity: int, spread_pct: float = 0.0):
    """Photoshop's Outer Glow, as its `OrGl` descriptor.

    Screen blend, soft technique -- the same defaults the fx dialog
    starts from, which is what makes the result recognisable as "a glow"
    rather than something odd that happens to have the right colour.
    `radius_px` is the dialog's Size and `spread_pct` its Spread: the
    solid share of that size before the soft fall-off. The caller maps
    the renderer's halo onto the pair (see webapp's live_text_effects);
    written with no spread, the same size came out as a faint haze next
    to the preview's.
    """
    from psd_tools.psd.descriptor import Bool, Descriptor, Enumerated, UnitFloat
    from psd_tools.terminology import Unit

    glow = _ps_descriptor(b"OrGl")
    glow[b"enab"] = Bool(True)
    glow[b"present"] = Bool(True)
    glow[b"showInDialog"] = Bool(True)
    # The long names, not the four-letter codes: that is what Photoshop
    # itself writes inside lfx2 (b"screen", b"multiply", b"normal"), and
    # the codes are what made it refuse the layer.
    glow[b"Md  "] = Enumerated(b"BlnM", b"screen")
    glow[b"Clr "] = _rgb_descriptor(color)
    glow[b"Opct"] = UnitFloat(float(max(0, min(100, opacity))), Unit.Percent)
    glow[b"GlwT"] = Enumerated(b"BETE", b"SfBL")
    # A percentage, tagged as pixels -- Photoshop's own habit (see the
    # drop shadow's Ckmt).
    glow[b"Ckmt"] = UnitFloat(float(max(0.0, min(100.0, spread_pct))), Unit.Pixels)
    glow[b"blur"] = UnitFloat(float(max(1.0, radius_px)), Unit.Pixels)
    glow[b"Nose"] = UnitFloat(0.0, Unit.Percent)
    glow[b"ShdN"] = UnitFloat(0.0, Unit.Percent)
    glow[b"AntA"] = Bool(False)
    glow[b"TrnS"] = _linear_contour_descriptor()
    glow[b"Inpr"] = UnitFloat(50.0, Unit.Percent)
    return glow


def _stroke_descriptor(color, size_px: float):
    """Photoshop's Stroke effect, as its `FrFX` descriptor -- outside the
    letterforms, solid colour, which is what the renderer draws."""
    from psd_tools.psd.descriptor import Bool, Descriptor, Enumerated, UnitFloat
    from psd_tools.terminology import Unit

    stroke = _ps_descriptor(b"FrFX")
    stroke[b"enab"] = Bool(True)
    stroke[b"present"] = Bool(True)
    stroke[b"showInDialog"] = Bool(True)
    stroke[b"Styl"] = Enumerated(b"FStl", b"OutF")
    stroke[b"PntT"] = Enumerated(b"FrFl", b"SClr")
    stroke[b"Md  "] = Enumerated(b"BlnM", b"normal")
    stroke[b"Opct"] = UnitFloat(100.0, Unit.Percent)
    stroke[b"Sz  "] = UnitFloat(float(max(1.0, size_px)), Unit.Pixels)
    stroke[b"Clr "] = _rgb_descriptor(color)
    stroke[b"overprint"] = Bool(False)
    return stroke


def _drop_shadow_descriptor(color, opacity, angle, distance, spread, size):
    """Photoshop's Drop Shadow, as its `DrSh` descriptor -- multiply
    blend, the dialog's defaults, the numbers as typed on the form."""
    from psd_tools.psd.descriptor import Bool, Descriptor, Enumerated, UnitFloat
    from psd_tools.terminology import Unit

    shadow = _ps_descriptor(b"DrSh")
    shadow[b"enab"] = Bool(True)
    shadow[b"present"] = Bool(True)
    shadow[b"showInDialog"] = Bool(True)
    shadow[b"Md  "] = Enumerated(b"BlnM", b"multiply")
    shadow[b"Clr "] = _rgb_descriptor(color)
    shadow[b"Opct"] = UnitFloat(float(max(0, min(100, opacity))), Unit.Percent)
    shadow[b"uglg"] = Bool(False)
    shadow[b"lagl"] = UnitFloat(float(angle), Unit.Angle)
    shadow[b"Dstn"] = UnitFloat(float(max(0.0, distance)), Unit.Pixels)
    # Spread is a percentage in the dialog, but Photoshop stores it with
    # the pixel unit tag like the other sizes -- a percent-tagged value
    # here is another thing it refuses.
    shadow[b"Ckmt"] = UnitFloat(float(max(0.0, min(100.0, spread))), Unit.Pixels)
    shadow[b"blur"] = UnitFloat(float(max(0.0, size)), Unit.Pixels)
    shadow[b"Nose"] = UnitFloat(0.0, Unit.Percent)
    shadow[b"AntA"] = Bool(False)
    shadow[b"TrnS"] = _linear_contour_descriptor()
    shadow[b"layerConceals"] = Bool(True)
    return shadow


def set_type_layer_effects(psd_path, effects: dict) -> list:
    """Give live type layers a real Photoshop glow and/or stroke.

    `effects` maps a lowercased layer (or group) name to a dict with any
    of:

        glow    {"color": (r, g, b), "radius": px, "opacity": 0-100, "spread": 0-100}
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
        from collections import OrderedDict

        from psd_tools.psd.descriptor import Bool, Integer, List, UnitFloat
        from psd_tools.psd.tagged_blocks import TaggedBlock
        from psd_tools.terminology import Unit
    except ImportError:
        return []
    try:
        psd = PSDImage.open(psd_path)
    except Exception:  # noqa: BLE001
        return []

    layers = _named_type_layers(psd)
    # Picture layers too: a glow or a shadow on the logo or the product
    # is a layer effect like any other.
    for layer in psd:
        key = (layer.name or "").strip().lower()
        if key and key not in layers and layer.kind == "pixel":
            layers[key] = layer
    styled = []
    for name, spec in (effects or {}).items():
        layer = layers.get(name.strip().lower())
        if layer is None or not spec:
            continue
        glow = spec.get("glow")
        stroke = spec.get("stroke")
        shadow = spec.get("shadow")
        if not glow and not stroke and not shadow:
            continue
        try:
            # The layer's existing effects stay unless replaced: a glow
            # from the form must not strip the drop shadow the designer
            # gave the header.
            existing = layer.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)
            block = existing if existing is not None else _empty_effects_block()
            if not block.name:
                # See _ps_descriptor: Photoshop needs the null name.
                block.name = "\x00"
            # lfx2's "object effects version" is 0 in every file Photoshop
            # writes. psd-tools' DescriptorBlock2 defaults to 1 -- and its
            # tagged_blocks.set_data() rebuilds the block through that
            # default (DescriptorBlock2(items)), which also drops the null
            # name. Photoshop reads version 1 as "a newer format than I
            # know" and refuses the layer. So: version 0, and the block
            # goes in as-is rather than through set_data.
            block.version = 0
            block[b"masterFXSwitch"] = Bool(True)
            # The effects' own scale. 100% means "the sizes below are in
            # the document's pixels", which is what the renderer measured
            # them in.
            block[b"Scl "] = UnitFloat(100.0, Unit.Percent)
            # Photoshop 2021+ keeps effects that can have several
            # instances -- drop shadow, stroke -- in a list
            # (`dropShadowMulti`, `frameFXMulti`) and reads that in
            # preference to the older single key. A block that already
            # has the list gets the effect there; one without (a
            # header Photoshop saved with a single shadow) keeps the
            # single key. Never both: the list would win and the
            # single one would be a shadow nobody sees.
            def put(single_key, multi_key, descriptor):
                if multi_key in block:
                    block[multi_key] = List([descriptor])
                    block.pop(single_key, None)
                else:
                    block[single_key] = descriptor

            if shadow:
                put(b"DrSh", b"dropShadowMulti", _drop_shadow_descriptor(
                    shadow.get("color", (0, 0, 0)),
                    shadow.get("opacity", 75),
                    shadow.get("angle", 120),
                    shadow.get("distance", 5),
                    shadow.get("spread", 0),
                    shadow.get("size", 5),
                ))
            if glow:
                block[b"OrGl"] = _outer_glow_descriptor(
                    glow.get("color", (255, 255, 255)),
                    glow.get("radius", 8),
                    glow.get("opacity", 100),
                    glow.get("spread", 0),
                )
            if stroke:
                put(b"FrFX", b"frameFXMulti", _stroke_descriptor(
                    stroke.get("color", (0, 0, 0)), stroke.get("size", 1)
                ))
            # numModifyingFX is Photoshop's count of the effects that are
            # switched on -- 0 on a layer with none, 1 on the header with
            # its one shadow, and read before the effects themselves: a
            # block saying 0 is a block Photoshop doesn't look inside.
            # The logo's fresh block came from a layer with nothing on,
            # so it said 0 with a glow switched on inside it, and the
            # logo opened with no Effects. Counted from what is actually
            # enabled now.
            block[b"numModifyingFX"] = Integer(_count_enabled_effects(block))
            layer.tagged_blocks[Tag.OBJECT_BASED_EFFECTS_LAYER_INFO] = TaggedBlock(
                key=Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, data=block
            )
            # The layer flag every Photoshop-written layer carries (bit 5
            # of the record's flags byte, undocumented in Adobe's spec).
            # A layer psd-tools builds from a picture has it off, and
            # Photoshop shows no Effects on such a layer no matter what
            # its effects blocks say -- proven by bisecting a download:
            # the same file with only this bit flipped on the logo
            # opened with the logo's glow listed.
            try:
                layer._record.flags.undocumented_1 = True
            except Exception:  # noqa: BLE001
                pass
            # The legacy twin, kept in step (see _PHOTOSHOP_EMPTY_LRFX_B64).
            legacy = layer.tagged_blocks.get_data(Tag.EFFECTS_LAYER)
            if legacy is None:
                legacy = _empty_legacy_effects_block()
            _sync_legacy_effects(legacy, shadow, glow)
            layer.tagged_blocks[Tag.EFFECTS_LAYER] = TaggedBlock(key=Tag.EFFECTS_LAYER, data=legacy)
            # And in Photoshop's order: the two effects blocks lead the
            # layer's extra data, ahead of the name, id and the rest.
            ordered = OrderedDict()
            for key in (Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, Tag.EFFECTS_LAYER):
                ordered[key] = layer.tagged_blocks[key]
            for key, value in layer.tagged_blocks.items():
                if key not in ordered:
                    ordered[key] = value
            layer.tagged_blocks._items = ordered
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


def _forget_document_text_engine(psd) -> bool:
    """Drop the document-level text engine block (Txt2) so Photoshop
    lays every type layer out from the layer's own engine data.

    Photoshop 2015.5+ keeps a second, whole-document copy of every type
    layer's engine state in that block, and prefers it when the file
    opens: a type layer rewritten here showed the new words (the raster
    is ours) until the layer was clicked, at which point Photoshop
    re-laid it out from Txt2 -- the template's English -- and the
    edit looked undone. Without the block Photoshop reads the per-layer
    data, which is what this module writes, and rebuilds Txt2 itself
    on save.
    """
    try:
        from psd_tools.constants import Tag

        blocks = psd.tagged_blocks
        if blocks is not None and Tag.TEXT_ENGINE_DATA in blocks:
            del blocks[Tag.TEXT_ENGINE_DATA]
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


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
    _forget_document_text_engine(psd)
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
    _forget_document_text_engine(psd)
    try:
        psd.save(psd_path)
    except Exception:
        return []
    return recoloured


def set_type_layer_font_size(psd_path, sizes: dict) -> list:
    """Resize live Photoshop type layers in place, keeping them text.

    `sizes` maps a lowercased layer name to the size in the PSD's own
    pixels the words should render at. Photoshop renders a type layer
    at FontSize x the layer's transform scale (a header scaled with
    Free Transform keeps its FontSize and carries a matrix), so the
    stored FontSize is the wanted pixels divided by that scale. Every
    style run gets the size -- a header set in two sizes becomes one
    size, which is what typing one size on the form means.

    Returns the names of the layers resized; [] when nothing could be.
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
    resized = []
    for name, px in (sizes or {}).items():
        layer = layers.get(name.strip().lower())
        if layer is None or not px or px <= 0:
            continue
        try:
            transform = layer.transform
            scale = float(transform[3]) if transform and len(transform) >= 4 and transform[3] else 1.0
            runs = layer.engine_dict["StyleRun"]["RunArray"]
        except Exception:  # noqa: BLE001
            continue
        stored = float(px) / (scale or 1.0)
        touched = False
        for run in runs:
            try:
                data = run["StyleSheet"]["StyleSheetData"]
                if "FontSize" in data:
                    data["FontSize"].value = stored
                    touched = True
            except Exception:  # noqa: BLE001
                continue
        if touched:
            resized.append(layer.name)
    if not resized:
        return []
    _forget_document_text_engine(psd)
    try:
        psd.save(psd_path)
    except Exception:  # noqa: BLE001
        return []
    return resized


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
    hidden: Optional[set] = None,
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

    `hidden` names (lowercased, before renaming) the layers to write
    switched off. A template layer someone turned off in Photoshop stays
    out of the flattened preview, and it has to stay off in the layered
    file too -- or the download opens showing a header the preview
    never had, and the motion clip (which reads this file) animates it.
    Written rather than dropped so it is one click away in Photoshop.

    Raises ValueError if `layers` is empty -- there's always at least a
    "Background" layer for a real creative, so an empty list almost
    certainly means the caller passed the wrong thing.
    """
    if not layers:
        raise ValueError("layers must contain at least one (name, image) entry")
    if layer_names is None:
        layer_names = REUPLOAD_LAYER_NAMES
    hidden = {h.strip().lower() for h in (hidden or ())}

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
        layer = psd.create_pixel_layer(cropped, name=psd_layer_name, top=top, left=left)
        if name.strip().lower() in hidden:
            layer.visible = False
    return psd


def save_layered_psd(
    layers: List[Tuple[str, Image.Image]],
    size: Tuple[int, int],
    dest_path,
    *,
    layer_names: Optional[dict] = None,
    hidden: Optional[set] = None,
) -> None:
    """build_layered_psd() and write it straight to `dest_path`."""
    build_layered_psd(layers, size, layer_names=layer_names, hidden=hidden).save(dest_path)


def hidden_layer_names(psd_path) -> set:
    """Lowercased names of the top-level layers switched off in
    `psd_path`; empty when the file can't be read."""
    try:
        psd = PSDImage.open(psd_path)
    except Exception:
        return set()
    from src.image_ops import layers_under_background

    # A layer under the background is hidden by the stack even with its
    # eye on -- it stays out of the layered PSD's visible set and out
    # of the motion clip, like one switched off.
    return {
        (layer.name or "").strip().lower()
        for layer in psd
        if (layer.name or "").strip() and not layer.visible
    } | layers_under_background(psd)


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
    # Photoshop's own files: the descriptor string ends in a NUL, the
    # engine string in a paragraph return, and the run lengths count
    # that return. Line breaks in `text` are paragraph returns too.
    lines = [line for line in text.replace("\r\n", "\n").replace("\n", "\r").split("\r")]
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines:
        return False
    engine_text = "\r".join(lines) + "\r"
    try:
        engine = layer.engine_dict
        old_text = str(engine["Editor"]["Text"].value)
        layer._data.text_data[b"Txt "].value = "\r".join(lines) + "\x00"
        engine["Editor"]["Text"].value = engine_text
    except Exception:
        return False
    old_paragraphs = old_text.split("\r")
    if old_paragraphs and old_paragraphs[-1] == "":
        old_paragraphs = old_paragraphs[:-1]
    new_lengths = [len(line) + 1 for line in lines]
    for key in ("StyleRun", "ParagraphRun"):
        try:
            lengths = engine[key]["RunLengthArray"]
            runs = engine[key]["RunArray"]
        except Exception:
            continue
        # Same number of lines as before: each line keeps the run its
        # first real character had -- so a header whose lines were set
        # at different sizes keeps every size. Otherwise the first run
        # is stretched over everything, as before.
        keep = None
        if len(old_paragraphs) == len(lines) and len(lengths) == len(runs) and len(runs) >= 1:
            old_lengths = [int(n) for n in lengths]
            keep = []
            cursor = 0
            for paragraph in old_paragraphs:
                first = cursor
                for offset, ch in enumerate(paragraph):
                    if not ch.isspace():
                        first = cursor + offset
                        break
                pos = 0
                chosen = len(runs) - 1
                for index, length in enumerate(old_lengths):
                    if first < pos + length:
                        chosen = index
                        break
                    pos += length
                keep.append(chosen)
                cursor += len(paragraph) + 1
        if keep is not None:
            kept_runs = [copy.deepcopy(runs[i]) for i in keep]
            while len(runs) > 0:
                runs.pop()
            while len(lengths) > 1:
                lengths.pop()
            for run in kept_runs:
                runs.append(run)
            template_length = lengths[0]
            lengths[0].value = new_lengths[0]
            for value in new_lengths[1:]:
                item = copy.deepcopy(template_length)
                item.value = value
                lengths.append(item)
        else:
            while len(lengths) > 1:
                lengths.pop()
                if len(runs) > 1:
                    runs.pop()
            if lengths:
                lengths[0].value = len(engine_text)
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


def write_whole_ad_psd(
    ad_image: Image.Image,
    split_layers: List[Tuple[str, Image.Image]],
    dest_path,
    *,
    template_path=None,
    copy: Optional[dict] = None,
) -> list:
    """The PSD for a whole-ad generation: the reconstructed layers of the
    model's picture, and -- when this size has a saved template -- the
    template's own real elements hidden beneath them for retouching.

    `split_layers` is AdSplit.layers(): ("background", ...) first, then
    the painted subject and text. With a template, the file IS the
    template: its background pixels become the reconstructed background,
    the painted layers go directly above it, and every other layer (the
    real logo and product, the live header / description / CTA type,
    retyped to `copy`) is switched off -- present, editable, one click
    from visible. Without a template it is just the reconstructed stack.

    Returns the top-level layer names in the file, bottom to top, or []
    when nothing could be written.
    """
    import shutil

    try:
        from psd_tools import PSDImage
    except ImportError:
        return []

    if template_path is None:
        try:
            build_layered_psd(split_layers, ad_image.size, layer_names={}).save(dest_path)
            set_flattened_preview(dest_path, ad_image)
            return [name for name, _ in split_layers]
        except Exception:  # noqa: BLE001
            return []

    try:
        shutil.copy(template_path, dest_path)
        background = dict(split_layers).get("background")
        if background is not None:
            if replace_pixel_layers(dest_path, {"background": background}) == []:
                # No pixel layer called background to swap: put one in
                # at the bottom instead.
                psd = PSDImage.open(dest_path)
                rgba = background if background.mode == "RGBA" else background.convert("RGBA")
                if rgba.size != (psd.width, psd.height):
                    rgba = rgba.resize((psd.width, psd.height), Image.LANCZOS)
                cropped, left, top = _tight_bbox_crop(rgba)
                made = psd.create_pixel_layer(cropped, name="background", top=top, left=left)
                psd.remove(made)
                psd.insert(0, made)
                psd.save(dest_path)
        if copy:
            set_type_layer_text(dest_path, {k: v for k, v in copy.items() if v})

        psd = PSDImage.open(dest_path)
        canvas = (psd.width, psd.height)
        for layer in psd:
            layer.visible = (layer.name or "").strip().lower() == "background"
        index = next(
            (i for i, layer in enumerate(psd) if (layer.name or "").strip().lower() == "background"),
            -1,
        )
        for name, rgba in split_layers:
            if name == "background":
                continue
            if rgba.mode != "RGBA":
                rgba = rgba.convert("RGBA")
            if rgba.size != canvas:
                rgba = rgba.resize(canvas, Image.LANCZOS)
            cropped, left, top = _tight_bbox_crop(rgba)
            made = psd.create_pixel_layer(cropped, name=name, top=top, left=left)
            psd.remove(made)
            index += 1
            psd.insert(index, made)
        psd.save(dest_path)
        set_flattened_preview(dest_path, ad_image)
        return [layer.name for layer in PSDImage.open(dest_path)]
    except Exception:  # noqa: BLE001
        return []
