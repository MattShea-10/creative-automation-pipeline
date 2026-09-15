"""Why does a run after a template write-back differ from the one before?

Saves both renders and their difference, and reports what the write-back
did to the template's layer styles. Changes nothing.
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "tests")
sys.path.insert(0, ".")
os.environ["CREATIVE_PIPELINE_OFFLINE"] = "1"

from PIL import Image, ImageChops, ImageEnhance  # noqa: E402
from psd_tools import PSDImage  # noqa: E402

import webapp  # noqa: E402
from test_webapp import LayerOverrideIntegrationTest as Case  # noqa: E402


def styles(path, label):
    psd = PSDImage.open(path)
    print(f"  {label}:")
    for layer in psd:
        name = (layer.name or "").strip()
        if not name:
            continue
        try:
            effects = [type(e).__name__ for e in (layer.effects or []) if getattr(e, "enabled", True)]
        except Exception as exc:  # noqa: BLE001
            effects = [f"<{exc}>"]
        if effects:
            print(f"     {name:14s} kind={layer.kind:6s} effects={effects}")


case = Case("test_a_run_after_a_template_write_back_starts_clean")
case.setUp()
try:
    (w, h), staged = case._stage_real_template()
    print(f"staged {staged.name} {w}x{h}")
    styles(staged, "template BEFORE any run")

    form = {
        "product_name": "HydroBoost", "campaign_message": "Drive the summer",
        "upload_custom_hero_enabled": "1",
        "layer_header_text": "Drive the summer",
        "header": "", "description": "",
    }

    first = case.client.post("/generate", data=dict(form, update_saved_templates="1"),
                             content_type="multipart/form-data")
    job1 = webapp.JOBS_DIR / re.search(rb"/download/([0-9a-f]+)", first.data).group(1).decode()
    styles(staged, "template AFTER the write-back")

    second = case.client.post("/generate", data=dict(form), content_type="multipart/form-data")
    job2 = webapp.JOBS_DIR / re.search(rb"/download/([0-9a-f]+)", second.data).group(1).decode()

    a = Image.open(next(job1.glob(f"*_{w}x{h}.png"))).convert("RGB")
    b = Image.open(next(job2.glob(f"*_{w}x{h}.png"))).convert("RGB")
    out = Path("_diagnose"); out.mkdir(exist_ok=True)
    a.save(out / "run1.png"); b.save(out / "run2.png")
    ImageEnhance.Brightness(ImageChops.difference(a, b)).enhance(6).save(out / "run_difference.png")
    total = sum(abs(x - y) for pa, pb in zip(a.getdata(), b.getdata()) for x, y in zip(pa, pb))
    print(f"mean difference between the two runs: {total / (a.width * a.height * 3):.2f} (test allows 2)")
    print("wrote _diagnose/run1.png, run2.png, run_difference.png")
finally:
    case.tearDown()
