"""Offline, zero-dependency image 'generator'.

Used as (a) the default when no network/API is available, and (b) an
automatic fallback if a live provider errors out mid-run -- so a demo
recording never fails just because a free API had a bad moment.

It deterministically renders a stylized placeholder (gradient + product
name) seeded from the prompt text, so re-runs of the same brief are
reproducible.
"""

from __future__ import annotations

import hashlib

from PIL import Image, ImageDraw

from ..image_ops import fit_text_block
from .base import ImageProvider


def _seed_color(text: str, offset: int = 0) -> tuple:
    h = hashlib.sha256((text + str(offset)).encode("utf-8")).hexdigest()
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


class MockImageProvider(ImageProvider):
    name = "mock"

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        negative_prompt: str = None,
    ) -> Image.Image:
        # Accepted and ignored: this draws a labelled placeholder, so
        # there is nothing to steer. Present only so callers never have
        # to ask which provider they're holding.
        top = _seed_color(prompt, 0)
        bottom = _seed_color(prompt, 1)

        # Build a 1px-wide vertical gradient and stretch it -- avoids a
        # per-pixel Python loop, which would be far too slow at 1920x1080.
        gradient = Image.new("RGB", (1, height))
        for y in range(height):
            t = y / max(height - 1, 1)
            gradient.putpixel((0, y), tuple(int(top[c] * (1 - t) + bottom[c] * t) for c in range(3)))
        img = gradient.resize((width, height))

        draw = ImageDraw.Draw(img)

        # This hero image is generated once (as a square, by default) and
        # then reused across every requested output size. Two later steps
        # can each clip a naively-placed label:
        #  1. center_crop_to_ratio() trims the sides for a narrow target
        #     (e.g. 9:16, keeping the center ~56% of width) or the top/
        #     bottom for a wide one (e.g. 16:9, keeping the center ~56% of
        #     height).
        #  2. add_message_banner() then overlays an opaque banner across up
        #     to the bottom 40% of the *final* cropped/resized image.
        # Confining the label to a horizontally-centered box (50% of width,
        # comfortably inside the ~56% kept by a 9:16 crop) and a vertical
        # band from ~22% to ~52% of height (inside the ~56% kept by a 16:9
        # crop, and mapped back through the worst-case crop+resize to stay
        # clear of even a maximal-height banner) keeps the label visible
        # and unclipped across all three default sizes.
        safe_w = int(width * 0.5)
        band_top = height * 0.22
        band_bottom = height * 0.52
        safe_h = int(band_bottom - band_top)
        initial_font_size = max(min(width, height) // 26, 16)

        # This placeholder label is deliberately capped at initial_font_size
        # rather than allowed to grow into the autofit box like a real
        # header/message banner would -- it's a fixed-style debug label, not
        # a creative element that should expand to fill space.
        font, lines, line_height = fit_text_block(
            draw,
            f"[MOCK IMAGE] {prompt}",
            safe_w,
            safe_h,
            min_font_size=10,
            max_font_size=initial_font_size,
        )

        text_block_height = line_height * len(lines)
        x = (width - safe_w) // 2
        y = band_top + (safe_h - text_block_height) / 2
        for line in lines:
            draw.text((x, y), line, fill=(255, 255, 255), font=font)
            y += line_height
        return img
