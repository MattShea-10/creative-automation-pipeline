"""Turning campaign copy into the market's language.

A region maps to a language code here; the translating itself belongs to
src/translation_providers.py, the same way image generation belongs to
src/providers/. This module decides *what* to ask for and what to do
when nobody can answer; the providers decide *how*.

Nothing here ever raises. A run whose copy could not be translated draws
the English and says so on the results page -- shipping the wrong
language silently is worse than shipping English loudly.
"""

from __future__ import annotations

import os
import sys

from .translation_providers import TranslationError, active_providers

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


# Why the most recent call failed, and who answered the most recent one
# that worked. Read through take_last_failure() / take_last_provider(),
# which clear as they read: a reason left lying around would be reported
# against some later, unrelated run.
_last_failure = ""
_last_provider = ""


def _network_is_off_limits() -> bool:
    """Whether this process may call a translator at all.

    The suite drives the real pipeline, and the real pipeline translates
    -- so a full run fired hundreds of requests at Google's free
    endpoint. That is what exhausted the daily quota, and the symptom was
    not a failing test: it was every real run afterwards coming back in
    English.

    A frozen build is never held back, whatever it happens to import:
    the packaged app translating for real is the whole point.
    """
    if getattr(sys, "frozen", False):
        return False
    if os.environ.get("CREATIVE_PIPELINE_OFFLINE"):
        return True
    return "unittest" in sys.modules


def take_last_failure() -> str:
    """Why the last translation failed -- see REASONS in
    translation_providers -- or "" if it worked. Cleared as it is read."""
    global _last_failure
    reason, _last_failure = _last_failure, ""
    return reason


def take_last_provider() -> str:
    """Which provider produced the last translation ("deepl",
    "google-free"), or "". Cleared as it is read. The results page says
    so, because "which service wrote this Spanish" is a fair question
    about copy that is going to be published."""
    global _last_provider
    name, _last_provider = _last_provider, ""
    return name


def localize_message(message: str, language: str) -> tuple[str, bool]:
    """(localized text, was_translated). Falls back to English on any
    failure and records why in take_last_failure().

    Each configured provider is tried in turn, so a DeepL key that is out
    of characters -- or that simply has no Hindi -- falls through to the
    free endpoint instead of failing the run.
    """
    global _last_failure, _last_provider
    if language == "en":
        return message, False
    if _network_is_off_limits():
        _last_failure = "offline"
        return message, False

    reason = "unavailable"
    for provider in active_providers():
        try:
            translated = provider.translate(message, language)
        except TranslationError as exc:
            reason = exc.reason
            print("[localization] %s: %s: %s" % (provider.name, language, exc), file=sys.stderr)
            continue
        except Exception as exc:  # noqa: BLE001
            # A provider raising something unexpected must not take the
            # run down with it -- there is another one behind it.
            reason = "unavailable"
            print("[localization] %s: %s: %s: %s" % (provider.name, language, type(exc).__name__, exc), file=sys.stderr)
            continue
        _last_failure = ""
        _last_provider = provider.name
        return translated, True

    _last_failure = reason
    return message, False
