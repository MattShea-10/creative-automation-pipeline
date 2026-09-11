"""Very small localization helper.

Design decision / limitation (documented in README): full localization is
out of scope for a 2-3 hour proof-of-concept. This module maps a target
region to a language code via a small lookup table, then uses
deep-translator (free, no API key -- wraps the public Google Translate
endpoint) to translate the campaign message. If the region is unknown or
the translation call fails (no network, library missing, etc.), it falls
back to the original English message and flags that in the run report
rather than silently failing.
"""

from __future__ import annotations

import sys

REGION_TO_LANGUAGE = {
    "mexico": "es",
    "spain": "es",
    "latin america": "es",
    "brazil": "pt",
    "france": "fr",
    "germany": "de",
    "italy": "it",
    "japan": "ja",
    "south korea": "ko",
    "china": "zh-CN",
    "india": "hi",
    "united states": "en",
    "usa": "en",
    "canada": "en",
    "united kingdom": "en",
}


def infer_language(target_region: str, explicit_language: str | None = None) -> str:
    if explicit_language:
        return explicit_language
    key = (target_region or "").strip().lower()
    return REGION_TO_LANGUAGE.get(key, "en")


def localize_message(message: str, language: str) -> tuple[str, bool]:
    """Returns (localized_text, was_translated). Falls back to English on any failure."""
    if language == "en":
        return message, False

    try:
        from deep_translator import GoogleTranslator

        # source="auto" rather than "en": what arrives here is not always
        # English. A PSD exported from an earlier Spanish run carries
        # Spanish in its type layers, and asking Google to read Spanish as
        # though it were English hands back something barely touched --
        # "BEBIDA REFRESCANTE" came back "REFRESCANTE BEBIDA" -- which the
        # caller cannot tell from a real translation, so it gets drawn onto
        # the creative and cached as the French for that phrase. Letting
        # Google detect the language actually translates it.
        translated = GoogleTranslator(source="auto", target=language).translate(message)
        if translated:
            return translated, True
    except Exception as exc:
        # Still swallowed -- the caller falls back to English and warns on
        # the results page -- but no longer without trace: the reason a run
        # came out in English (no network, rate limit, the endpoint moving)
        # used to be lost entirely. This lands in the console window the
        # packaged app runs in, and in the CI smoke test's exe.log.
        print("[localization] %s: %s: %s" % (language, type(exc).__name__, exc), file=sys.stderr)

    return message, False
