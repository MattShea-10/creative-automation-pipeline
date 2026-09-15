"""Writes the preview and the layered-PSD composite side by side, plus a
difference map, for the one test that says they don't match.

Not a test. A way to look at the disagreement instead of arguing about
whether a number is too big.
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

case = Case("test_the_layered_psd_download_looks_like_the_preview")
case.setUp()
try:
    (w, h), staged = case._stage_real_template()
    print(f"staged template: {staged.name} at {w}x{h}")
    response = case.client.post("/generate", data={
        "product_name": "HydroBoost", "market": "UK", "audience": "runners",
        "campaign_message": "Drive the summer",
        "upload_custom_hero_enabled": "1",
        "layer_description_text": "Drive the summer",
        "layer_description_use_custom_color": "1",
        "layer_description_text_color": "#33ff66",
        "layer_cta_text": "claim my prize",
        "layer_cta_button_color": "#f2760c",
        "header": "", "description": "",
    }, content_type="multipart/form-data")
    job_id = re.search(rb"/download/([0-9a-f]+)", response.data).group(1).decode()
    job = webapp.JOBS_DIR / job_id

    rendered = Image.open(job / f"HydroBoost_campaign1_{w}x{h}.png").convert("RGB")
    psd = PSDImage.open(job / f"HydroBoost_campaign1_{w}x{h}.psd")
    layered = psd.composite(force=True).convert("RGB").resize(rendered.size)

    out = Path("_diagnose")
    out.mkdir(exist_ok=True)
    rendered.save(out / "preview.png")
    layered.save(out / "layered_psd.png")
    diff = ImageChops.difference(rendered, layered)
    ImageEnhance.Brightness(diff).enhance(6).save(out / "difference_boosted.png")

    total = sum(abs(a - b) for pa, pb in zip(rendered.getdata(), layered.getdata())
                for a, b in zip(pa, pb))
    print(f"mean per-channel difference: {total / (rendered.width * rendered.height * 3):.2f} (test allows 8)")
    print("layers in the download:", [(l.name, l.kind, l.visible) for l in psd])
    print("wrote _diagnose/preview.png, layered_psd.png, difference_boosted.png")
finally:
    case.tearDown()
