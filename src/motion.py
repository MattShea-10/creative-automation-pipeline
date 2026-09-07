"""Motion versions of a finished creative: a short looping MP4 per size,
built from the layers the app already separated.

Not generative video -- no model, no spend, no waiting on an API. The
backdrop drifts, the product settles into place, the type and the CTA
arrive in order and hold, and in the last half-second the overlays fade
so the clip loops back to its first frame without a jump. It is the
"animated banner" that display networks and social autoplay take, made
deterministically from the size's own layered PSD.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

FPS = 30
DEFAULT_DURATION = 8.0

# When each kind of layer arrives, in seconds, as (fade start, fade end,
# rise as a fraction of the canvas height). Anything not named here is
# treated like text.
ENTRANCES = {
    "background": None,
    "logo": (0.2, 0.8, 0.0),
    "product": (0.3, 1.1, 0.03),
    "subject (painted)": (0.3, 1.1, 0.02),
    "header": (0.9, 1.6, 0.02),
    "description": (1.3, 2.0, 0.02),
    "text (painted)": (1.0, 1.7, 0.02),
    "legal": (1.6, 2.2, 0.0),
    "cta": (1.9, 2.5, 0.015),
}
DEFAULT_ENTRANCE = (1.0, 1.7, 0.02)
# Overlays fade out over this long at the end so the loop closes.
LOOP_FADE_OUT = 0.5
# How much the backdrop pushes in over the clip (1.0 = none).
BACKDROP_ZOOM = 1.06


def _ease(t: float) -> float:
    """Smooth 0..1 -> 0..1 (ease in-out)."""
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def ffmpeg_path() -> Optional[str]:
    """The bundled ffmpeg (imageio-ffmpeg) if present, else one on PATH."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return shutil.which("ffmpeg")


def layers_from_psd(psd_path) -> Tuple[List[Tuple[str, Image.Image]], Tuple[int, int]]:
    """The visible top-level pixel layers of a PSD as (name, full-canvas
    RGBA) pairs, bottom to top -- the layered PSD the app saves per size
    is exactly this stack."""
    from psd_tools import PSDImage

    psd = PSDImage.open(psd_path)
    canvas = (psd.width, psd.height)
    out = []
    for layer in psd:
        if not layer.visible or layer.kind not in ("pixel", "group", "shape", "type"):
            continue
        try:
            image = layer.composite() if layer.kind == "group" else layer.topil()
        except Exception:  # noqa: BLE001
            continue
        if image is None:
            continue
        full = Image.new("RGBA", canvas, (0, 0, 0, 0))
        full.paste(image.convert("RGBA"), (layer.left, layer.top))
        out.append(((layer.name or "").strip().lower(), full))
    return out, canvas


def _backdrop_frame(background: Image.Image, canvas: Tuple[int, int], phase: float) -> Image.Image:
    """The backdrop pushed in by `phase` (0..1) of BACKDROP_ZOOM, cropped
    back to the canvas from the centre."""
    w, h = canvas
    zoom = 1.0 + (BACKDROP_ZOOM - 1.0) * phase
    zw, zh = max(w, int(round(w * zoom))), max(h, int(round(h * zoom)))
    big = background.resize((zw, zh), Image.BILINEAR)
    left, top = (zw - w) // 2, (zh - h) // 2
    return big.crop((left, top, left + w, top + h))


def _with_alpha(layer: Image.Image, alpha: float) -> Image.Image:
    if alpha >= 1.0:
        return layer
    if alpha <= 0.0:
        return None
    a = layer.getchannel("A").point(lambda v: int(v * alpha))
    out = layer.copy()
    out.putalpha(a)
    return out


def _entrance(name: str):
    if name in ENTRANCES:
        return ENTRANCES[name]
    for key, value in ENTRANCES.items():
        if key != "background" and key in name:
            return value
    return DEFAULT_ENTRANCE


def render_frames(layers, canvas, duration: float = DEFAULT_DURATION, fps: int = FPS, loop: bool = True):
    """Yield RGB frames of the motion version."""
    w, h = canvas
    background = next((img for name, img in layers if name == "background"), None)
    if background is None:
        # No named backdrop: the bottom layer plays the part.
        background = layers[0][1] if layers else Image.new("RGBA", canvas, (0, 0, 0, 255))
        overlays = layers[1:]
    else:
        overlays = [(n, i) for n, i in layers if n != "background"]
    background = background.convert("RGBA")
    total = max(1, int(round(duration * fps)))
    for index in range(total):
        t = index / fps
        # Backdrop: out and back over the whole clip when looping, so
        # the last frame's zoom equals the first's.
        phase = _ease(t / duration)
        if loop:
            phase = _ease(2 * t / duration if t < duration / 2 else 2 - 2 * t / duration)
        frame = _backdrop_frame(background, canvas, phase)
        # Overlays: arrive, hold, and (when looping) leave together.
        tail = 1.0
        if loop and t > duration - LOOP_FADE_OUT:
            tail = max(0.0, (duration - t) / LOOP_FADE_OUT)
        for name, layer in overlays:
            start, end, rise = _entrance(name)
            progress = _ease((t - start) / max(end - start, 1e-6))
            alpha = progress * tail
            if alpha <= 0.0:
                continue
            shown = layer
            if rise and progress < 1.0:
                dy = int(round(h * rise * (1.0 - progress)))
                shown = Image.new("RGBA", canvas, (0, 0, 0, 0))
                shown.paste(layer, (0, dy), layer)
            shown = _with_alpha(shown, alpha)
            if shown is not None:
                frame.alpha_composite(shown)
        yield frame.convert("RGB")


def render_motion_clip(
    psd_path,
    out_path,
    *,
    duration: float = DEFAULT_DURATION,
    fps: int = FPS,
    loop: bool = True,
    fallback_image=None,
) -> dict:
    """Write the MP4 for one size. Returns {"path", "seconds", "frames",
    "layers"}; raises RuntimeError with a plain reason when it can't."""
    exe = ffmpeg_path()
    if not exe:
        raise RuntimeError("ffmpeg isn't available (pip install imageio-ffmpeg, or install ffmpeg).")
    layers, canvas = ([], None)
    if psd_path is not None and Path(psd_path).is_file():
        try:
            layers, canvas = layers_from_psd(psd_path)
        except Exception:  # noqa: BLE001
            layers, canvas = [], None
    if not layers:
        if fallback_image is None:
            raise RuntimeError("no layers to animate and no flat image to fall back on")
        flat = Image.open(fallback_image).convert("RGBA")
        layers, canvas = [("background", flat)], flat.size
    w, h = canvas
    # H.264 needs even dimensions; pad a pixel rather than refuse a size.
    ew, eh = w + (w % 2), h + (h % 2)
    cmd = [
        exe, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ew}x{eh}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frames = 0
    try:
        for frame in render_frames(layers, canvas, duration=duration, fps=fps, loop=loop):
            if (ew, eh) != (w, h):
                padded = Image.new("RGB", (ew, eh), (0, 0, 0))
                padded.paste(frame, (0, 0))
                frame = padded
            proc.stdin.write(frame.tobytes())
            frames += 1
    finally:
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed: {err.strip()[:300]}")
    return {"path": str(out_path), "seconds": duration, "frames": frames, "layers": [n for n, _ in layers]}
