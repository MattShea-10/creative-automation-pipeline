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

import os
import sys
import time

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



# The free endpoint allows roughly five calls a second, and a run does
# not make them one at a time: the header, description, legal, CTA and
# AI headline go out back to back, then every line of text found in the
# templates. A dozen calls inside one second is answered with
# TooManyRequests, the copy comes out in English, and the warning blamed
# the network -- so a rate limit we caused ourselves read as an outage.
_MIN_SECONDS_BETWEEN_CALLS = 0.25

# How long to wait after being rate limited, per retry. Rising, because
# a fixed delay against a per-second limit just arrives late together.
_RATE_LIMIT_BACKOFF = (1.0, 2.0, 4.0)

_last_call_at = 0.0

# Why the most recent call failed, for the warning the results page
# shows. Read through take_last_failure(), which clears it: a reason
# left lying around would be attached to some later, unrelated run.
_last_failure = ""


def _network_is_off_limits() -> bool:
    """Whether this process is allowed to call the translator at all.

    The suite drives the real pipeline, and the real pipeline
    translates -- so a full run fired hundreds of requests at Google's
    free endpoint. That is what exhausted the daily quota, and the
    symptom was not a failing test: it was every real run afterwards
    coming back in English. Tests get the English fallback and never
    reach the network.

    A frozen build is never held back, whatever else it happens to
    import: the packaged app translating for real is the whole point.
    """
    if getattr(sys, "frozen", False):
        return False
    if os.environ.get("CREATIVE_PIPELINE_OFFLINE"):
        return True
    return "unittest" in sys.modules


def _pause(seconds: float) -> None:
    """Every wait in this module goes through here, so tests can hold
    still instead of sleeping through a backoff."""
    if seconds > 0:
        time.sleep(seconds)


def _wait_our_turn() -> None:
    """Keep consecutive calls at least _MIN_SECONDS_BETWEEN_CALLS apart."""
    _pause(_MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_call_at))


def _is_a_rate_limit(exc: Exception) -> bool:
    """deep-translator raises TooManyRequests, but importing that class
    to catch it would make this module depend on the library being
    installed -- which is exactly the case it has to survive."""
    return (
        type(exc).__name__ == "TooManyRequests"
        or "too many requests" in str(exc).lower()
    )


def take_last_failure() -> str:
    """Why the last translator call failed -- "rate limit", "error page",
    "unavailable" -- or "" if the last one worked or nothing has called
    yet. Cleared as it is read, so a stale reason can never be reported
    against a run that didn't have it.
    """
    global _last_failure
    reason, _last_failure = _last_failure, ""
    return reason


def localize_message(message: str, language: str) -> tuple[str, bool]:
    """Returns (localized_text, was_translated). Falls back to English on
    any failure, and records why in take_last_failure().

    Calls are spaced and a rate limit is waited out rather than given up
    on: the caller translates a whole form's worth of copy in a loop, and
    the endpoint counts requests per second, so the first version of this
    reliably tripped the limit on its own copy and then reported the
    result as "the translator didn't answer".
    """
    global _last_call_at, _last_failure
    if language == "en":
        return message, False
    if _network_is_off_limits():
        _last_failure = "offline"
        return message, False

    for attempt in range(1 + len(_RATE_LIMIT_BACKOFF)):
        _wait_our_turn()
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
        except Exception as exc:
            _last_call_at = time.monotonic()
            # Still swallowed -- the caller falls back to English and warns on
            # the results page -- but no longer without trace: the reason a run
            # came out in English (no network, rate limit, the endpoint moving)
            # used to be lost entirely. This lands in the console window the
            # packaged app runs in, and in the CI smoke test's exe.log.
            print("[localization] %s: %s: %s" % (language, type(exc).__name__, exc), file=sys.stderr)
            if _is_a_rate_limit(exc) and attempt < len(_RATE_LIMIT_BACKOFF):
                _last_failure = "rate limit"
                _pause(_RATE_LIMIT_BACKOFF[attempt])
                continue
            _last_failure = "rate limit" if _is_a_rate_limit(exc) else "unavailable"
            break

        _last_call_at = time.monotonic()
        if translated and not _looks_like_an_error_page(translated):
            _last_failure = ""
            return translated, True
        if translated:
            print(
                "[localization] %s: the translator returned an error page, not a "
                "translation: %r" % (language, translated[:120]),
                file=sys.stderr,
            )
            _last_failure = "error page"
        else:
            # An empty answer is not a translation, and asking again gets
            # the same nothing -- the endpoint is up and refusing.
            _last_failure = "unavailable"
        break

    return message, False
