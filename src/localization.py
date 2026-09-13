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


# Google's error pages come back through the translator as ordinary
# strings, not exceptions -- so a 500 reads as a successful translation.
# Three campaigns shipped with the headline "ERROR 500 (SERVER ERROR)"
# and the body "That's an error. There was an error. Please try again
# later. That's all we know." painted into the creative by the model,
# and the same text cached against the source phrase so every later run
# reused it without calling the translator at all.
_ERROR_PAGE_MARKERS = (
    "that\u2019s an error",
    "that's an error",
    "that\u2019s all we know",
    "that's all we know",
    "server error",
    "error 500",
    "error 502",
    "error 503",
    "<!doctype",
    "<html",
)


def _looks_like_an_error_page(text: str) -> bool:
    """Is this an error page wearing a translation's clothes?

    Deliberately a fixed list of markers rather than a guess at what a
    translation "should" look like: a real translation can be any
    length, any script, and can legitimately contain the word "error".
    Two markers together, or one of the unmistakable ones, is the bar.
    """
    lowered = (text or "").lower()
    if "<!doctype" in lowered or "<html" in lowered:
        return True
    hits = sum(1 for marker in _ERROR_PAGE_MARKERS if marker in lowered)
    return hits >= 2



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
        if translated and not _looks_like_an_error_page(translated):
            return translated, True
        if translated:
            print(
                "[localization] %s: the translator returned an error page, not a "
                "translation: %r" % (language, translated[:120]),
                file=sys.stderr,
            )
    except Exception as exc:
        # Still swallowed -- the caller falls back to English and warns on
        # the results page -- but no longer without trace: the reason a run
        # came out in English (no network, rate limit, the endpoint moving)
        # used to be lost entirely. This lands in the console window the
        # packaged app runs in, and in the CI smoke test's exe.log.
        print("[localization] %s: %s: %s" % (language, type(exc).__name__, exc), file=sys.stderr)

    return message, False
