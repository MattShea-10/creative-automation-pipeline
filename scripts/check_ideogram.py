#!/usr/bin/env python3
"""Probe the Ideogram API directly and print exactly what it says.

Run this from a terminal on the machine that runs the app:

    python3 scripts/check_ideogram.py

It reads IDEOGRAM_API_KEY from .env (or the environment), makes one
small request, and prints the HTTP status and response body for each
request shape it tries. That status is the whole diagnosis:

    401  the key is wrong, or billing/credits aren't active yet
    402  the account is out of credits
    404  IDEOGRAM_MODEL is not a valid model path segment
    400/415/422  the request shape is wrong -- the body says which field
    200  the API works; the problem is elsewhere in the app
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

from src.providers.ideogram_provider import API_BASE, DEFAULT_MODEL

key = os.environ.get("IDEOGRAM_API_KEY")
if not key:
    sys.exit("IDEOGRAM_API_KEY is not set (checked .env and the environment).")

model = os.environ.get("IDEOGRAM_MODEL") or DEFAULT_MODEL
url = f"{API_BASE}/v1/{model}/generate"
fields = {
    "prompt": "a red apple on a white table",
    "aspect_ratio": "1x1",
    "rendering_speed": "TURBO",
    "num_images": 1,
}

# Never print the key itself -- just enough to tell two keys apart.
print(f"key      : {key[:4]}...{key[-4:]} ({len(key)} chars)")
print(f"model    : {model}")
print(f"url      : {url}")
print()

for label, kwargs in (
    ("JSON body", {"json": fields}),
    ("multipart", {"files": {k: (None, str(v)) for k, v in fields.items()}}),
):
    try:
        resp = requests.post(url, headers={"Api-Key": key}, timeout=120, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"{label}: request failed before a reply: {exc}")
        continue
    print(f"{label}: HTTP {resp.status_code}")
    print(f"  {resp.text[:600]}")
    print()
    if resp.status_code == 200:
        print("This shape works.")
        break
