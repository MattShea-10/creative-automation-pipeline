# Creative Automation Pipeline (Proof of Concept)

A local, command-line proof-of-concept that turns a campaign brief into
ready-to-post social ad creatives: it reuses or generates a hero image per
product via a GenAI image API, renders it into multiple aspect ratios,
overlays the (optionally localized) campaign message, and runs a couple of
lightweight brand/legal compliance checks -- all logged to a JSON run report.

Built for the "Creative Automation for Scalable Social Ad Campaigns" take-home.

## Why not Adobe Firefly

The brief asks for "best-fit APIs available" rather than naming a specific
vendor, and Firefly's API requires an Adobe enterprise account that isn't
available here. Instead, the image-generation step sits behind a small
provider interface (`src/providers/`) so any vendor can be plugged in.
Three providers ship out of the box:

| Provider | Cost | Setup | Notes |
|---|---|---|---|
| `pollinations` (default) | Free | None -- no signup, no API key | Plain HTTP GET to image.pollinations.ai. Lowest friction for reviewing this project, but no uptime/quality SLA. |
| `huggingface` | Free tier | Free HF account + token | Hits the HF Inference API (defaults to Stable Diffusion XL). A reasonable stand-in for a "real" hosted diffusion model. |
| `mock` | Free | None | Fully offline. Renders a deterministic placeholder (gradient + label) instead of calling any network API. |

**Automatic fallback:** if the selected live provider fails for any reason
(network blocked, rate limited, model cold-starting, etc.), the pipeline
automatically falls back to the offline `mock` provider for that image and
keeps going, logging a warning and recording it in the run report. This
means the pipeline -- and the demo recording -- never hard-fails just
because a free API had a bad moment. Pass `--no-fallback` to disable this
and fail loudly instead.

Swapping in Adobe Firefly, OpenAI's Images API, Stability AI, etc. later is
just writing one more small class in `src/providers/` that implements
`ImageProvider.generate(prompt, width, height) -> PIL.Image` and registering
it in `src/providers/__init__.py`. Nothing else in the pipeline changes.

## How to run it

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Run with the free, keyless default provider:
python -m src.main --brief briefs/sample_campaign.yaml

# Force fully-offline mode (no network calls at all):
python -m src.main --brief briefs/sample_campaign.yaml --provider mock

# Use Hugging Face instead (needs a free token in .env, see .env.example):
cp .env.example .env   # then fill in HUGGINGFACE_API_TOKEN
python -m src.main --brief briefs/sample_campaign.yaml --provider huggingface
```

Useful flags: `--sizes` (comma-separated `WIDTHxHEIGHT` pixel sizes, e.g.
`--sizes 1080x1080,1080x1920,1920x1080` -- overrides the brief and the
built-in default), `--fit {crop,contain}` (how to fit the hero image into
each size -- see "Bringing in your own designed creative" below),
`--no-header` (hide the header/title band -- see "Header, message, and
logo overlays" below), `--assets-dir` (default `assets/`), `--output-dir`
(default `outputs/`), `--report` (default `<output-dir>/run_report.json`),
`-v` for debug logging, `--no-fallback` to disable the mock fallback.

Output sizes default to three pixel presets (1080x1080, 1080x1920,
1920x1080) covering the brief's 1:1 / 9:16 / 16:9 requirement, but they're
fully configurable rather than hardcoded to those ratios:

- Per-brief, via an `output_sizes` list (strings like `"1080x1080"`, or
  `{width, height}` objects) -- see `briefs/sample_campaign.yaml` for a
  commented example.
- Per-run, via `--sizes` on the command line, which takes precedence over
  the brief.

Filenames and the run report label each creative by its actual pixel
dimensions (e.g. `hydroboost_1080x1080.png`) plus the aspect ratio derived
from those pixels (e.g. `1:1`) -- so pointing the pipeline at a custom size
like `1200x628` still gets a sensible label instead of a hardcoded name.

**A campaign isn't limited to 3 sizes.** `--sizes` (and `output_sizes` in
the brief) both accept a comma-separated list where each item is a preset
name and/or an explicit size, freely mixed -- so one run can cover more
than one size family at once:

```bash
# All 9 web ad sizes plus all 3 broadcast resolutions in one run:
python -m src.main --brief briefs/sample_campaign.yaml --sizes "web-top7,broadcast"

# The 3 default social sizes plus one extra custom size:
python -m src.main --brief briefs/sample_campaign.yaml --sizes "default,1200x628"
```

Any pixel size that appears in more than one requested preset (e.g.
1920x1080 is in both `default` and `broadcast`) is only rendered once --
sizes are de-duplicated, not doubled up.

Two example briefs are included: `briefs/sample_campaign.yaml` (2 products,
YAML) and `briefs/sample_campaign.json` (3 products, JSON) -- both formats
are accepted interchangeably.

## Quick-generate web UI

The CLI above is built around a full campaign brief -- a region, an
audience, 2+ products, GenAI generation, localization, compliance checks.
That's the right tool for an actual campaign, but it's more setup than you
want when you just have one hero image (or a short product video), a
header, and a description, and want to see every ad size immediately. For
that, there's a small local web app that satisfies the brief's "simple app"
option directly in the browser:

```bash
pip install -r requirements.txt   # includes Flask
python webapp.py
```

Then open `http://127.0.0.1:5000`. Upload a hero image or video, type a
header and description, optionally expand "Advanced options" to pick sizes
(the same `default` / `web-top7` / `broadcast` presets and custom
`WIDTHxHEIGHT` sizes the CLI's `--sizes` takes), a fit-mode toggle, a text
color and background toggle for the header and, separately, for the
description, and an optional logo upload. Click "Generate creatives" and
every size renders as a thumbnail on the results page, downloadable
individually or all together as a zip.

Each of the two text overlays -- "Header text style" and "Description text
style" -- has its own color picker and its own "No background
(transparent)" checkbox, set independently. Checking "No background"
removes the semi-transparent black plate from behind that banner's text
entirely, so it reads as plain colored text sitting on the image -- useful
when the plate feels too heavy for a given design, or clashes with a
light/branded hero image. Since removing the plate also removes its
guaranteed contrast, an outline is added around that banner's text
automatically in that case (black outline for light text, white outline
for dark text, chosen from the text color you picked) so it stays legible
regardless of what's behind it, whichever banner it's applied to.

Each banner also has a "Glow" checkbox and its own glow color picker, as a
stylistic alternative to that automatic outline -- a soft colored halo
rendered behind the text (a blurred, colorized copy of the text, composited
under the crisp letterforms) instead of a thin contrast line. Turning glow
on for a banner skips its automatic outline, so the two effects don't
visually compete; header and message glow are independent, same as their
colors and background toggles.

Text size in both banners is chosen automatically to be the *largest* font
that still fits the available space, rather than a fixed guess that only
ever shrinks. In practice this means a short headline on a wide or
squarish frame renders large enough to actually use the width instead of
looking small and lost, and a narrow-but-tall format like a 160x600
skyscraper gets a legible size driven by its generous height rather than
its cramped width -- while long copy still wraps and shrinks down to a
readable floor instead of overflowing the frame, exactly as before. Each
banner also has its own "Font size (px)" field to override that -- leave it
on "Auto" for the automatic behavior above, or type an exact pixel size to
pin it; a pinned size that needs more room than the banner would normally
take grows the banner/plate to fit it (up to the full frame as a hard
safety limit) rather than clipping your choice.

Each banner also has its own Left/Center/Right alignment control. The
header defaults to centered and the message banner to left-aligned,
matching their original fixed look, but either can be changed
independently -- e.g. a right-aligned header to balance a logo on the
opposite corner, or a centered message banner for a more poster-like
layout.

The "Brand logo" upload also has its own "Logo placement" and "Logo size &
opacity" controls, defaulting to the original top-right, ~16%-of-frame,
fully-opaque watermark look. A top-left or top-right logo shares the
header row, so the header automatically reserves space on that side (the
same collision-avoidance the original top-right-only logo always had, now
mirrored for top-left too) and the logo is composited right after the
header so it sits on top of that band. Bottom-left, bottom-right, and
center have no header to share, so the logo is instead composited *last*
-- after the message banner and badge image -- so it isn't covered by
either; as with the badge and CTA button, that means a bottom or center
logo can visually overlap message-banner text that reaches the same
region, so pick a placement that clears your layout.

There's also a "Badge image" upload, independent of the brand logo -- a
second image for anything the logo isn't meant for: a "Sale" sticker,
seasonal seal, award badge, or a full-frame tint/texture/decorative frame.
"Badge placement" picks where it goes: a corner or center sizes it (via
"Size (% of frame)") and draws it *last*, on top of everything else, like
a badge sitting on the finished creative; "Full frame" stretches it to
cover the whole canvas and draws it right after the hero image is fit into
the frame, *before* the header/message text, so it reads as a backdrop the
text sits on top of rather than a decal covering the text. "Opacity (%)"
blends the badge in at any strength either way, so a fully opaque source
image can still be used as a subtle tint without needing a pre-baked
semi-transparent asset.

The hero image upload is optional -- there's also a separate
"Size-specific PSD templates" section (right below it, not buried in
"Advanced options") where you can upload up to four `.psd` files, each
paired with an exact target size (e.g. `728x480`). Each one becomes the
background for *that* size only, using Pillow's native flattened/composite
preview of the PSD (`Image.open(path).convert("RGB")` -- the same path any
other image takes); there's no layer extraction or role recognition. A
size added this way is automatically folded into the batch even if it
isn't separately checked or typed under "Output sizes". The hero image is
only required for sizes that aren't covered by a matching PSD template --
if some requested size has neither, generation is rejected with a message
naming exactly which size(s) are missing one.

Templates you want to reuse across campaigns don't have to be re-uploaded
each time -- drop a `.psd` file into the project's `default_templates/`
folder with its target size in the filename (e.g. `970x90.psd` or
`tester-160x600.psd`, matched anywhere in the name) and it's picked up
automatically on every future generate request, including auto-adding
that size to the batch the same way an uploaded template does. A
per-request upload for the same size (via the form's upload rows) always
overrides the saved default; a bad/corrupt file in that folder is simply
skipped rather than breaking generation. See `default_templates/README.txt`.

Any size resolved from a PSD template -- whether uploaded this request or
picked up from `default_templates/` -- is used exactly as-is: fit to the
frame (no cropping, matching `resize_to_contain`) but with **no**
header/message/logo/badge/CTA drawn on top. The whole point of a template
is that it's already a finished design; those overlay fields only apply
to sizes filled in from the general hero image.

For a "just one upload" workflow, there's also a **728x480 content PSD**
field at the very top of the form. Upload a single `.psd` there and the
exported batch becomes exactly that size plus whatever's saved in
`default_templates/` -- the "Output sizes"/"Custom sizes" choices and the
general hero image are all disregarded in that mode.

Saved `default_templates/` files only come into play on a request where
you've uploaded at least one PSD yourself (a per-request row, or the
content PSD field) -- upload no PSD at all and they're skipped entirely,
so a plain hero-image request never silently gains extra sizes or has one
swapped out for a saved template.

The PSD section also has an **"Update layers across all template
sizes"** block: a description-text field and logo/CTA-image/product-image
uploads. Fill any of these in and, on every template-covered size whose
PSD has a matching named layer (`description`, `logo`, `cta`, `product`),
the new content is composited into that exact layer's bounding box for
that size -- a different position/scale per size, since each size's
template has its own layout. This doesn't edit the PSD itself (Pillow
can't reliably decode or rewrite an arbitrary PSD layer's own pixels,
especially a live text layer) -- image layers are pasted in, contain-fit
and centered, on top of the flattened template; the description is
painted over with a sampled fill and redrawn with your text. A size whose
PSD doesn't have a given named layer just skips that one update rather
than erroring.

Under the hood this calls the exact same `render_creative()` function
(`src/creative_render.py`) the CLI's pipeline calls, so a creative made in
the browser looks and behaves identically to one made by the full
pipeline at the same size/fit-mode/header/logo/message -- video frame
extraction, the crop-vs-contain fit logic, and the
header/logo-collision handling are all shared, not reimplemented. What the
web UI intentionally does *not* do -- by design, to stay a quick single-image
tool rather than a second copy of the campaign pipeline -- is GenAI
generation, multi-product briefs, localization/translation, or the
brand/legal compliance checks; those are what the full CLI brief flow is
for. Generated files land in `outputs/web/<job-id>/` (gitignored, same as
the CLI's `outputs/`).

## Web ad & broadcast video size presets

Two named presets are built in, on top of the `default` (1080x1080,
1080x1920, 1920x1080) set:

```bash
# The top 7 standard IAB/Google display (web) ad sizes:
python -m src.main --brief briefs/sample_campaign.yaml --sizes web-top7

# If these creatives were extended to video, the pixel sizes a broadcast/TV
# delivery would target:
python -m src.main --brief briefs/sample_campaign.yaml --sizes broadcast
```

**`web-top7`** -- common web/display ad sizes (desktop placements, one
mobile placement, plus commonly-used additional sizes):

| Size | Name |
|---|---|
| 728x90 | Leaderboard |
| 300x250 | Medium Rectangle |
| 336x280 | Large Rectangle |
| 160x600 | Skyscraper |
| 320x50 | Mobile Leaderboard |
| 250x250 | Square |
| 200x200 | Small Square |
| 468x60 | Banner |
| 970x90 | Large Leaderboard |
| 728x480 | Wide Rectangle |

(Kept under the `web-top7` preset name for continuity even though the list
has grown to 10 entries. "Wide Rectangle" isn't an official IAB name --
there isn't a standard one for 728x480 -- it's just a readable label for a
size that comes up often as a flattened-creative delivery size (e.g. a
Photoshop export). Two other sizes sometimes grouped with these -- 300x600
"Half Page Ad" and 970x250 "Billboard" -- aren't in this preset but are
still recognized with a friendly name if you request them explicitly, e.g.
`--sizes 300x600`.)

**`broadcast`** -- if these same creatives were extended to video for TV,
broadcast (unlike web) standardizes on a single 16:9 frame rather than a
variety of aspect ratios, so this is a short list of resolution tiers:

| Size | Name |
|---|---|
| 1920x1080 | Full HD / 1080i / 1080p -- the primary US broadcast delivery standard |
| 1280x720 | 720p -- used by some networks (e.g. Fox/Disney-owned) instead of 1080i |
| 3840x2160 | 4K UHD -- increasingly requested for premium/streaming-adjacent delivery |

Both presets work with `--sizes <name>` on the command line or
`output_sizes: <name>` in a brief, exactly like an explicit `WIDTHxHEIGHT`
list (see `src/image_ops.py`'s `SIZE_PRESETS`). A requested size that
matches one of the tables above is also labeled with its friendly name
(e.g. "Leaderboard") in the console output and `run_report.json`, instead
of just an aspect ratio.

**Mobile vs. desktop folders.** Any recognized display ad size (the
`web-top7` table above, whether requested via the preset or individually)
is additionally sorted into a `mobile/` or `desktop/` subfolder under its
product folder, based on published IAB placement guidance -- only 320x50
(Mobile Leaderboard) is mobile-specific; the rest are desktop/web
placements:

```
outputs/
  hydroboost/
    mobile/
      hydroboost_320x50.png
    desktop/
      hydroboost_728x90.png
      hydroboost_300x250.png
      hydroboost_336x280.png
      hydroboost_160x600.png
      hydroboost_250x250.png
      hydroboost_200x200.png
      hydroboost_468x60.png
      hydroboost_970x90.png
```

The social defaults (1:1/9:16/16:9) and `broadcast` resolutions aren't
classic "display ad" units in that sense (a 9:16 story runs on mobile and
a 16:9 broadcast frame has no device at all), so those stay directly under
the product folder, unchanged from before -- the mobile/desktop split only
applies where it's a meaningful, well-established distinction.

Note this pipeline still only produces still images -- `broadcast` gives
you the correct frame dimensions to design/export video into, not an
actual video file. See "Assumptions & limitations" for what changes (and
what doesn't) about the creative template at these very wide/short web
ad ratios.

## Bringing in your own designed creative (e.g. a video)

The pipeline isn't limited to product photos it generates or that you drop
in as plain source photography -- you can also point it at an already
laid-out, finished design (a flattened image export), or even a video
file, and have a frame from it resized/fit into every configured output
size.

**Getting the file in:**

- Drop it in following the naming convention -- `assets/<slug>.png` (or
  `.jpg`/`.webp`/`.tif`/`.tiff`/`.mp4`/`.mov`/`.m4v`/`.avi`/`.mkv`/`.webm`)
  -- and it's picked up automatically, same as any other pre-made asset.
- Or point a specific product at it explicitly via `asset_path` in the
  brief (see the commented example in `briefs/sample_campaign.yaml`), which
  is handier when the file doesn't already follow the slug-based naming
  convention or lives outside `assets/`:
  ```yaml
  products:
    - name: "HydroBoost Sports Drink"
      slug: "hydroboost"
      asset_path: "assets/renders/hydroboost_728x480.png"
      # or: asset_path: "assets/video/hydroboost_demo.mp4"
  ```
  `asset_path` takes priority over both the naming convention and GenAI
  generation for that product.

**Video support**: point `asset_path` (or the naming convention) at a
`.mp4`/`.mov`/`.m4v`/`.avi`/`.mkv`/`.webm` file and the pipeline extracts a
single frame from it and treats that frame exactly like any other hero
image from that point on -- run through `crop`/`contain`, headline, logo,
message overlay, all the same as a photo. Requires `opencv-python-headless`
(listed in `requirements.txt`, so `pip install -r requirements.txt` covers
it). By default it grabs the frame from the **middle** of the video, since
a video's first frame is often a black frame, fade-in, or title card that
wouldn't make a good hero shot. To pick a specific moment instead, set
`video_frame_seconds` alongside `asset_path`:
```yaml
products:
  - name: "HydroBoost Sports Drink"
    asset_path: "assets/video/hydroboost_demo.mp4"
    video_frame_seconds: 4.5   # grab the frame at 4.5s instead of the middle
```
(`video_frame_seconds` only has an effect when `asset_path` is set and
points at a video -- a video picked up via the plain naming convention
always uses the middle-frame default, since there's no per-asset brief
entry to attach a timestamp override to.) If the file won't open, the
error message suggests re-exporting it as H.264 MP4, which OpenCV reads
reliably across platforms.

**Fitting it into every output size -- `crop` vs. `contain`:** this is the
part that matters most for a finished design, and it's controlled by
`--fit` on the CLI or `fit_mode` in the brief (CLI wins if both are set):

| Mode | Behavior | Best for |
|---|---|---|
| `crop` (default) | Scales up to fill the target frame completely, cropping whatever doesn't fit -- the same "fill" behavior used for generic product photos. | Generic photography where losing a bit of the edges is fine. |
| `contain` | Scales the whole image down to fit *entirely* inside the target frame with **no cropping**, and fills the leftover space with a softly blurred, stretched version of the same image as a backdrop rather than plain black bars. | A finished, already-composed design -- logo, CTA, and layout are all fixed in place, so cropping would cut pieces off. |

This matters a lot in practice: a 728x480 finished design (logo
top-left, CTA bottom-right) run through `crop` into a 970x90 banner keeps
only a thin center sliver -- the logo and CTA are both gone. The same file
through `contain` keeps the entire design intact, just shrunk down and
centered on a blurred backdrop of itself. `crop` is still the right
default for a product photo the pipeline is free to reframe; `contain` is
the right choice whenever the source image is itself the finished ad:

```bash
python -m src.main --brief briefs/sample_campaign.yaml --fit contain
```

or in the brief:

```yaml
fit_mode: "contain"
```

## Header, message, and logo overlays

Every creative can carry up to five overlays. Header, logo, and message
are positioned so they don't collide with each other by default; the
badge image and CTA button are free-floating on top and can be
positioned to avoid the others (see the CTA note below):

| Overlay | Position | Text/image source | Purpose |
|---|---|---|---|
| Header / title band | Top, centered by default | `headline` on the product, else `headline` on the brief, else the product's own name | A short title -- reads like an ad's headline. |
| Brand logo | Configurable -- a corner or center, top-right by default | `brand.logo` image in the brief | Brand mark, also feeds the compliance check. |
| Message banner | Bottom, left-aligned by default | `message` in the brief (localized) | The main campaign message/CTA. |
| Badge image | Configurable -- a corner, center, or the full frame | any image, independent of the brand logo | A sticker/seal, or a full-frame tint/texture. Web UI only for now -- see `add_badge_image()` in `src/image_ops.py`. |
| CTA button | Configurable -- six positions including a "sticky" bottom-center | short text (e.g. "Shop Now") | A filled, pill-shaped call-to-action button. Web UI only for now -- see `add_cta_button()` in `src/image_ops.py`. |

The header is on by default -- if you don't set anything, each product's
own name becomes its title, so every creative gets *some* header without
any extra brief configuration. To customize it:

```yaml
headline: "Summer Sale -- Up To 30% Off"   # campaign-wide, applies to every product
products:
  - name: "HydroBoost Sports Drink"
    headline: "New: HydroBoost Zero Sugar"  # overrides the campaign-wide headline for just this product
```

Precedence is product's own `headline` > brief's `headline` > the product's
name. A custom `headline` (either level) is translated the same way the
campaign message is; a headline that's just the product's name falls back
to English untranslated, since product names are proper nouns.

To turn the header off entirely: `--no-header` on the CLI (always wins), or
`show_header: false` in the brief. Both the header and the logo are sized
and positioned so a long headline won't run underneath the logo -- the
header's text area automatically shrinks to leave room for it when a brand
logo is configured.

Like the message banner, the header's font size is chosen automatically to
be the largest that fits the wrapped text into the available space, and
its text wraps rather than overflowing the frame. That autofit -- an
explicit binary search for "the biggest font that still fits," rather than
a fixed guess that only ever shrinks -- is what makes a short headline on a
wide/squarish frame grow to actually use the width, and what makes a
narrow-but-tall format like a 160x600 skyscraper get a legible size driven
by its abundant height instead of its cramped width. At the shortest
web-ad banners (e.g. 320x50, 970x90) a header, logo, and message banner all
sharing one very short frame still gets visually tight -- see "Assumptions
& limitations" for the same tradeoff already documented for those extreme
ratios; `--no-header` is the quickest way to reclaim space there if you
don't need a title on those particular sizes.

Both the header and message banners always render in white text, centered
and left-aligned respectively, on a semi-transparent black plate when
driven from the CLI/brief -- none of that is yet exposed as a brief field.
The quick-generate web UI (see above) does expose a text color picker, a
toggle to drop the plate entirely, a glow (soft colored halo) option, an
exact font-size override, and a left/center/right alignment control,
independently for the header and the description; the same
`text_color`/`show_background`/`glow`/`glow_color`/`align`/`font_size`
parameters already exist on `add_header_banner()`/`add_message_banner()` in
`src/image_ops.py` (and on `render_creative()`, as `header_*`/`message_*`)
if you want to wire brief-level control through later.

### Call-to-action button

The web UI's "Call-to-action" field renders a short label (e.g. "Shop Now",
"Learn More") as a filled, pill-shaped button -- `button_color` (the pill's
background fill) and `text_color` pickers, a `font_family` choice (sans-serif,
serif, monospace, or condensed), an optional exact `font_size` (auto-sized
to the frame if left blank), an optional `glow`/`glow_color` for a soft
colored halo around the whole button shape (a "this button is lit up"
emphasis effect, off by default), and six positions (the four corners, dead
center, or "bottom-center," the common sticky-action-bar placement). It's
composited last of everything -- after the header, message, logo, and
badge image -- so the button is always the topmost, clickable-looking
element on the finished creative, regardless of what else is on the frame.

The button auto-shrinks its font (and, as a last resort, truncates the
label with a trailing "…") so it always stays within the canvas even on a
narrow frame like a 160x600 skyscraper with a long CTA string -- it will
never spill past the edges. That shrink-then-truncate behavior always
targets the button's own label text and is independent of `font_family`/
`glow`.

Because the CTA button floats independently of the header/message banners
rather than reserving space from them, a bottom-center or corner CTA can
visually overlap bottom-message or header text if that text runs long
enough to reach the same area (e.g. a two-line bottom message combined
with a bottom-center button). Pick a CTA position that doesn't compete with
your header/message layout -- e.g. a bottom-right or bottom-left button
tends to clear a left-aligned message, and a top corner clears a bottom
message entirely -- or leave the CTA text blank if the message banner
already reads as the call to action.

## Example input

`briefs/sample_campaign.yaml`:

```yaml
campaign:
  name: "Summer Refresh 2026"
  target_region: "Mexico"
  target_audience: "Young adults 18-25 interested in fitness and outdoor activity"
  message: "Feel the Fresh. Own Your Summer."
  brand:
    logo: "assets/brand/logo.png"
    colors: ["#0057B8", "#FFD100"]
  products:
    - name: "HydroBoost Sports Drink"
      slug: "hydroboost"
      prompt_hint: "Professional studio product photo of a chilled blue sports drink bottle..."
    - name: "FreshGlow Body Wash"
      slug: "freshglow"
      prompt_hint: "Professional studio product photo of a green body wash bottle..."
```

## Example output

Running the command above produces:

```
outputs/
  hydroboost/
    hydroboost_1080x1080.png
    hydroboost_1080x1920.png
    hydroboost_1920x1080.png
  freshglow/
    freshglow_1080x1080.png
    freshglow_1080x1920.png
    freshglow_1920x1080.png
  run_report.json
```

Console summary:

```
=== Creative Automation Run Summary ===
Campaign:     Summer Refresh 2026
Language:     es (translated: False)
Fit mode:     crop
Header:       shown
Creatives:    6
Duration:     1.2s
Report file:  outputs/run_report.json

  [OK] HydroBoost Sports Drink 1080x1080  1:1                        generated                -> outputs/hydroboost/hydroboost_1080x1080.png
  ...
```

`run_report.json` is a machine-readable log of every creative produced:
source (user-provided / cache / generated / mock fallback), aspect ratio,
the header text actually rendered (if any), output path, and the result of
each compliance check -- useful for downstream reporting/analytics per the
brief's "actionable insights" goal.

## Design decisions

- **Language & structure**: Python 3.11, stdlib + Pillow/requests/PyYAML.
  Chosen for fast iteration and because Pillow makes aspect-ratio
  cropping and text compositing straightforward without a heavy
  dependency tree.
- **Provider abstraction** (`src/providers/`): decouples the pipeline from
  any single GenAI vendor -- see "Why not Adobe Firefly" above.
- **Storage abstraction** (`src/storage.py`): the brief allows Azure/AWS/
  Dropbox for storage. `LocalAssetStore` implements the same narrow
  `get_hero_image` / `put_hero_image` interface a cloud-backed
  implementation would need, backed by the local filesystem, so pointing
  this at real cloud storage later means writing one adapter class rather
  than reworking the pipeline.
- **Asset reuse**: before generating anything, the pipeline checks a
  product's explicit `asset_path` (if set in the brief), then looks for a
  pre-made asset at `assets/<slug>.png` or `assets/<slug>/hero.png`
  (`.png`/`.jpg`/`.webp`/`.tif`/`.tiff` all accepted;
  `.mp4`/`.mov`/`.m4v`/`.avi`/`.mkv`/`.webm` via OpenCV, which extracts a
  single frame -- see "Video support" above -- and hands it off as if it
  were any other hero image). If found,
  it's reused as-is (logged as `user-provided` or `user-provided (explicit
  asset_path)`, with `, extracted video frame` appended for a video
  source). Otherwise it checks a local generation cache
  (`assets/generated_cache/`, logged as `cache`) before finally calling the
  GenAI provider (logged as `generated`). See "Bringing in your own
  designed creative" above.
- **Output sizes**: explicit pixel dimensions (not ratio labels) are the
  source of truth (`src/image_ops.py`'s `DEFAULT_SIZES`), with an aspect
  ratio label (e.g. `16:9`) derived from the pixels via `gcd` purely for
  display. Precedence is `--sizes` CLI flag > brief's `output_sizes` >
  built-in default (1080x1080, 1080x1920, 1920x1080). Rendering defaults to
  center-crop-then-resize (the "fill" strategy) because social platforms
  expect the frame fully filled, not letterboxed -- but a `contain`
  (fit-without-cropping, blurred-backdrop letterbox) mode is also available
  for finished/composed creatives; see "Bringing in your own designed
  creative" above. Precedence there is the same shape: `--fit` CLI flag >
  brief's `fit_mode` > default (`crop`).
- **Message overlay**: a semi-transparent bottom banner that grows to fit
  the wrapped text (and shrinks the font automatically for longer or
  localized messages) rather than a fixed-height banner, so text always
  wraps within the frame instead of being clipped.
- **Header/title overlay**: a second, independent banner across the top,
  built on the same auto-fit/auto-wrap logic as the message banner (they
  share one implementation, `_add_edge_banner()`, parameterized by which
  edge and whether text is centered). It defaults to each product's name so
  every creative has a header with zero brief configuration, and can be
  overridden per-product or campaign-wide. Its available width shrinks
  automatically when a brand logo is configured in the header row (the
  default top-right, or top-left via the web UI's "Logo placement"), so a
  long headline is guaranteed not to render underneath it -- see "Header,
  message, and logo overlays" above.
- **Mock provider text placement**: the offline placeholder image is
  generated once per product (typically as a square) and then reused
  across every output size, so its debug label is confined to a safe
  central band that survives both the later aspect-ratio crop and the
  message banner overlay, rather than being positioned relative to the
  square canvas alone.
- **Compliance checks are heuristics, not production-grade review**:
  - *Brand*: logo presence is tracked deterministically (the pipeline
    knows whether it composited the logo), and a simple average-color
    distance check flags creatives whose dominant color strays far from
    the declared brand palette.
  - *Legal*: a configurable prohibited-word list is scanned against the
    campaign message (e.g. "guaranteed", "clinically proven"). This is a
    keyword check, not real legal review.
  - Both are wired as a pass/fail gate per creative and surfaced as
    warnings in the console output and `run_report.json`, demonstrating
    the hook a production system would use to block or route creatives
    for human review.
- **Logging/reporting**: standard `logging` module to console
  (`-v` for debug) plus a structured `run_report.json` per run.
- **Shared rendering, two front ends**: the "fit hero image into frame,
  then stack header/logo/message overlays" logic used to live inline in
  `CreativePipeline.run()`'s loop. It's now `render_creative()` in
  `src/creative_render.py`, called by both the CLI pipeline and
  `webapp.py`, so the quick-generate web UI can't silently drift from what
  the full brief pipeline produces -- one implementation, two ways to
  drive it (a YAML/JSON brief, or a browser form).

## Assumptions & limitations

- This is a proof-of-concept scoped to ~2-3 hours of work, not a
  production system -- error handling, retries, and compliance checks are
  intentionally simple and clearly labeled as heuristics above.
- `webapp.py` is a local single-user development tool (Flask's built-in
  dev server, `debug=False` by default, no authentication) -- fine for
  running on your own machine, not intended to be exposed on a network or
  deployed as-is. It also doesn't run the GenAI providers, localization,
  or compliance checks the CLI's brief pipeline does (see "Quick-generate
  web UI" above) -- it's deliberately the "one image in, sizes out" half of
  the tool, not a second campaign pipeline.
- Localization translates only the campaign message text (via
  `deep-translator`, which is free and keyless but scrapes a public Google
  Translate endpoint -- not an official, rate-limited, or SLA-backed API).
  If translation fails (no network, endpoint blocked, etc.) the pipeline
  silently falls back to the original English message and records that in
  the report rather than failing the run. Region-to-language mapping is a
  small hardcoded table (`src/localization.py`) covering the example
  briefs' regions plus a few common ones; unmapped regions default to
  English.
- The default `pollinations` provider and the translation fallback both
  require outbound internet access on whatever machine runs this. If
  that's unavailable (e.g. a locked-down sandbox), the pipeline still
  completes successfully using the offline `mock` image provider and
  untranslated text -- this was in fact how it was validated during
  development, since the build environment's network is restricted.
- Brand color compliance compares one average image color against the
  declared palette -- a real implementation would use palette extraction/
  clustering (e.g. k-means) rather than a single average.
- Automated coverage is a set of smoke tests (`tests/test_pipeline_smoke.py`,
  run via `python -m unittest discover tests`), not a comprehensive suite --
  they check that the default sizes, a custom size override, the
  `web-top7`/`broadcast` presets, `fit_mode="contain"` (including a
  synthetic "designed creative" comparison between `crop` and `contain` at
  several output sizes), an explicit `asset_path` override, and a
  missing/invalid `asset_path` falling back gracefully, all render
  end-to-end without error.
- Video support adds `opencv-python-headless` as a real dependency
  (`requirements.txt`) rather than an optional extra -- it's a heavier
  install than the rest of the stack, but keeps video ingestion working
  out of the box via `pip install -r requirements.txt` rather than needing
  a separate manual step. It's validated with synthetic MP4 fixtures built
  and read back with OpenCV directly in the test suite (asserting the
  middle-frame default and an explicit `video_frame_seconds` both land on
  the right part of the clip), not against a real-world video export --
  container/codec combinations OpenCV can't decode on a given machine are
  the most likely failure mode; re-exporting as H.264 MP4 is the fallback
  the pipeline's error message suggests.
- Generated/cached images and run outputs are gitignored; `assets/brand/logo.png`
  is a placeholder sample logo generated for this repo, not a real brand asset.
- The pipeline generates one hero image per product (at a fixed default
  resolution) and reuses it across every requested size via center-crop --
  a real photo shoot for a single hero shot reused across formats. This
  works well for the social ratios (1:1, 9:16, 16:9) and broadcast (all
  16:9), but the `web-top7` sizes include much more extreme ratios (e.g.
  970x90 is ~10.8:1, 468x60 is ~7.8:1, 320x50 is ~6.4:1). Cropping any
  single photo that aggressively will only show a sliver of it -- true of
  a real hero photo, not just the mock placeholder -- and the
  header-plus-message-banner-plus-logo template gets visually cramped at
  very short heights, now more so with both a top header band and a bottom
  message band competing for the same handful of pixels. A production
  system would either art-direct/re-crop a source image per format or use a
  dedicated, simpler layout (logo + short text on a solid/blurred
  background) for compact standard banner sizes rather than reusing the
  full photographic template. The pipeline still renders something for
  every `web-top7` size without erroring (fonts, the logo, and both banners
  shrink to fit), it's just not a polished result at the most extreme
  ratios -- `--no-header` is a quick way to free up space there if a title
  isn't needed on those particular sizes.
- The web UI's profanity check (`src/compliance.py`) uses the
  `better-profanity` library and is a real `requirements.txt` dependency --
  it blocks generation outright when flagged text is found in any of the
  campaign brief, header/title, description, or CTA fields. The companion
  trademark check is a lighter-weight bonus: it OCRs uploaded images (hero,
  logo, badge, and any layer-override images) and flags well-known brand
  names it finds as literal text, as a warning rather than a block --
  text-only, so it can't recognize an actual logo mark. It needs the
  `tesseract` OCR binary plus `pip install pytesseract`, neither of which
  is in `requirements.txt` (a heavier, optional install); without them it
  silently finds nothing rather than failing.

## Project structure

```
webapp.py             # quick-generate web UI entry point (python webapp.py)
templates/             # HTML templates for the web UI (index + results pages)
src/
  main.py             # CLI entry point
  brief_loader.py     # parses YAML/JSON campaign briefs
  models.py           # CampaignBrief / Product / BrandGuidelines dataclasses
  pipeline.py         # orchestrates the full-brief run + builds the report
  creative_render.py  # render_creative(): fit + header + logo + message,
                       # shared by both the CLI pipeline and the web UI
  storage.py          # asset storage abstraction (local folder today)
  image_ops.py        # sizing, crop/contain fit, text overlay, logo watermark,
                       # video frame reading
  localization.py     # region -> language + translation w/ fallback
  compliance.py       # brand color / logo / legal keyword checks
  providers/
    base.py
    mock_provider.py
    pollinations_provider.py
    huggingface_provider.py
briefs/               # example campaign briefs (YAML + JSON)
assets/
  brand/logo.png      # sample brand logo
  generated_cache/    # generated hero images get cached here for reuse
outputs/              # generated creatives + run_report.json land here
  web/                # quick-generate UI job output (one folder per job)
```

## Sources

`web-top7`'s sizes/names were specified directly. The mobile-vs-desktop
placement classification and the `broadcast` preset's resolution/frame-rate
standards were confirmed against:

- [Display Ad Sizes For Desktop And Mobile](https://martech.zone/standard-ad-sizes/) -- Martech Zone
- [1080i](https://en.wikipedia.org/wiki/1080i) -- Wikipedia
