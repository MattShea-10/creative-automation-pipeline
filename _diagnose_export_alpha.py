"""Does a soft alpha survive save_layered_psd()?

Takes the real template's header layer WITH its drop shadow drawn in
(soft gradient alpha), writes it through the same export the download
uses, reads it back, and reports what happened to the gradient.

Touches nothing in the app. Writes one PSD into _diagnose/.
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")

from PIL import Image  # noqa: E402
from psd_tools import PSDImage  # noqa: E402

from src.image_ops import get_psd_layer_stack  # noqa: E402
from src.psd_export import save_layered_psd  # noqa: E402

sys.path.insert(0, "tests")
from template_fixture import template  # noqa: E402

source = template("tester-1080x1080.psd")
if source is None:
    raise SystemExit("no shipped template to read")

def describe(name, image):
    alpha = image.convert("RGBA").getchannel("A")
    hist = alpha.histogram()
    total = sum(hist)
    clear = hist[0]
    solid = sum(hist[251:])
    mid = total - clear - solid
    print(f"  {name:22s} clear={clear:8d} partial={mid:8d} solid={solid:8d}"
          f"  ({100.0 * mid / max(total - clear, 1):.1f}% of drawn pixels are partial)")

stack = {n.strip().lower(): im for n, im in (get_psd_layer_stack(source, with_effects=True) or [])}
header = stack.get("header")
if header is None:
    raise SystemExit("no header layer in the stack")

print("BEFORE the export (what the renderer produced):")
describe("header", header)

out = Path("_diagnose"); out.mkdir(exist_ok=True)
dest = out / "alpha_roundtrip.psd"
save_layered_psd([(n, im) for n, im in stack.items()], header.size, dest, layer_names={})

psd = PSDImage.open(dest)
print("AFTER the export (read back out of the PSD):")
for layer in psd:
    if (layer.name or "").strip().lower() != "header":
        continue
    print(f"  layer kind={layer.kind} has_mask={layer.mask is not None}")
    pixels = layer.topil()
    if pixels is not None:
        describe("header topil", pixels)
    if layer.mask is not None:
        m = layer.mask.topil()
        if m is not None:
            describe("header mask", m.convert("RGBA").getchannel("R").convert("RGBA"))
    comp = psd.composite(force=True)
    if comp is not None:
        comp.convert("RGB").crop((0, 0, header.size[0], 260)).save(out / "alpha_roundtrip.png")
        print("  wrote _diagnose/alpha_roundtrip.png")
    break
