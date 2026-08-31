Default size templates
=======================

Any .psd file placed directly in this folder (no subfolders) whose
filename contains a WIDTHxHEIGHT pattern -- e.g. "970x90.psd",
"tester-160x600.psd", "hero_728x480_v2.psd" -- is automatically used as
the background template for that exact output size on every future
"generate" request in the web app. That size is automatically included
in the output batch even if it wasn't checked or typed into the
"Output sizes" / "Custom sizes" fields for that request.

This is separate from the "Size-specific PSD templates" upload rows on
the generate form: a per-request upload for the same size always takes
priority over a file saved here. Use this folder for layouts you want
saved permanently (e.g. a leaderboard or skyscraper template you reuse
every campaign); use the form's upload rows for a template that's fresh
for just this one request.

Only the flattened/composite preview of the PSD is used (the same way
Pillow renders any other image) -- no layer extraction or role
recognition. If a file here fails to open (corrupt, wrong format despite
the .psd extension), it's silently skipped rather than breaking
generation for everyone else.
