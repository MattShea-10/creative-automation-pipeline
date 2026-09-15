"""Where translated copy comes from.

Same shape as src/providers/base.py does for image generation, and for
the same reason: the pipeline should not know which vendor is answering.
It asks for Spanish and gets Spanish, or gets told why not.

Two providers ship:

  DeepL          -- authenticated, a real monthly quota, better Spanish
                    and French than the alternative. Needs a key.
  Google (free)  -- the unauthenticated endpoint deep-translator wraps.
                    No key, no quota you can check, and no promises.

Google alone was the whole localization story for a while, and it failed
in the least useful way possible: a run of the test suite spent the day's
invisible quota, and hours later a Spanish campaign came out in English
with a warning blaming the network. Hence a provider that can be asked
how much is left, with the free endpoint kept behind it as a fallback
rather than as the plan.
"""

from __future__ import annotations

import os
import sys
import time
from abc import ABC, abstractmethod


class TranslationError(Exception):
    """A provider could not translate. `reason` is one of the strings in
    REASONS -- what the results page tells the person, in their terms."""

    def __init__(self, message: str, reason: str = "unavailable"):
        super().__init__(message)
        self.reason = reason


# Every way a translation can fail to arrive, in the words the results
# page uses. "rate limit" is being told to slow down and is worth waiting
# out; "quota" is the month's allowance gone and is not.
REASONS = ("rate limit", "quota", "error page", "unsupported language", "no key", "unavailable")


def _pause(seconds: float) -> None:
    """Every wait in this module goes through here, so tests can hold
    still instead of sleeping through a backoff."""
    if seconds > 0:
        time.sleep(seconds)


class TranslationProvider(ABC):
    name: str = "base"

    # Whether it needs an API key at all.
    requires_key: bool = False

    # USD per million characters, so the app can say what localization
    # costs the way it already says what an image costs. Zero for the
    # free endpoint -- which is not the same as free of consequence.
    cost_per_million_characters: float = 0.0

    # Minimum spacing between calls. The free endpoint counts requests
    # per second and a run translates a form's worth of copy in a loop.
    min_seconds_between_calls: float = 0.0

    _last_call_at: float = 0.0

    def is_configured(self) -> bool:
        return True

    def wait_our_turn(self) -> None:
        _pause(self.min_seconds_between_calls - (time.monotonic() - self._last_call_at))

    @abstractmethod
    def translate(self, text: str, language: str) -> str:
        """The text in `language` (an app language code: es, fr, ...).

        Raises TranslationError with a reason on any failure. Returning
        the English unchanged is not this layer's decision to make.
        """


class DeepLProvider(TranslationProvider):
    """DeepL's REST API. A free key allows 500,000 characters a month.

    Free and paid keys use different hosts, and a free key sent to the
    paid host answers 403 -- which reads exactly like a bad key. The
    ":fx" suffix DeepL puts on free keys says which host to use, so the
    app works that out rather than asking anyone to.
    """

    name = "deepl"
    requires_key = True
    cost_per_million_characters = 0.0  # within the free allowance

    # https://developers.deepl.com/docs/resources/supported-languages
    # App code -> DeepL target. Anything absent raises "unsupported
    # language" and the caller falls through to the next provider, which
    # is how Hindi still works.
    TARGETS = {
        "es": "ES", "fr": "FR", "de": "DE", "it": "IT", "ja": "JA", "ko": "KO",
        "pt": "PT-BR", "pt-br": "PT-BR", "pt-pt": "PT-PT",
        "zh-cn": "ZH-HANS", "zh": "ZH-HANS", "zh-tw": "ZH-HANT",
        "nl": "NL", "pl": "PL", "ru": "RU", "sv": "SV", "da": "DA", "fi": "FI",
        "nb": "NB", "cs": "CS", "el": "EL", "hu": "HU", "id": "ID", "tr": "TR",
        "uk": "UK", "ro": "RO", "sk": "SK", "sl": "SL", "bg": "BG", "et": "ET",
        "lt": "LT", "lv": "LV", "ar": "AR",
    }

    def __init__(self, api_key: str = None):
        self.api_key = (api_key if api_key is not None else os.environ.get("DEEPL_API_KEY") or "").strip()

    def is_configured(self) -> bool:
        return bool(self.api_key)

    # DeepL runs free and paid accounts on different hosts, and a key
    # sent to the wrong one answers 403 -- indistinguishable from a bad
    # key. The ":fx" suffix says which, but people copy keys out of a
    # dashboard and the suffix gets left behind, so the suffix decides
    # the ORDER rather than the answer: try the likely host, and on a
    # refusal try the other before calling the key bad.
    FREE_HOST = "api-free.deepl.com"
    PAID_HOST = "api.deepl.com"

    def hosts(self) -> list:
        return (
            [self.FREE_HOST, self.PAID_HOST]
            if self.api_key.endswith(":fx")
            else [self.PAID_HOST, self.FREE_HOST]
        )

    @property
    def endpoint(self) -> str:
        return f"https://{self.hosts()[0]}/v2/translate"

    def target_for(self, language: str) -> str:
        target = self.TARGETS.get((language or "").strip().lower())
        if not target:
            raise TranslationError(f"DeepL has no {language}", "unsupported language")
        return target

    def translate(self, text: str, language: str) -> str:
        if not self.is_configured():
            raise TranslationError("no DeepL key set", "no key")
        target = self.target_for(language)
        import requests

        response = None
        for host in self.hosts():
            self.wait_our_turn()
            try:
                response = requests.post(
                    f"https://{host}/v2/translate",
                    headers={"Authorization": f"DeepL-Auth-Key {self.api_key}"},
                    json={"text": [text], "target_lang": target},
                    timeout=20,
                )
            except Exception as exc:  # noqa: BLE001
                raise TranslationError(f"{type(exc).__name__}: {exc}", "unavailable") from exc
            finally:
                type(self)._last_call_at = time.monotonic()
            if response.status_code not in (401, 403):
                break   # this host knows the key; whatever it says is the answer

        if response.status_code == 456:
            raise TranslationError("DeepL says this month's characters are used up", "quota")
        if response.status_code == 429:
            raise TranslationError("DeepL says slow down", "rate limit")
        if response.status_code in (401, 403):
            raise TranslationError(
                "DeepL refused the key (a free key must end in :fx and uses api-free.deepl.com)",
                "no key",
            )
        if response.status_code != 200:
            raise TranslationError(f"DeepL answered {response.status_code}", "unavailable")
        try:
            translated = response.json()["translations"][0]["text"]
        except (ValueError, KeyError, IndexError) as exc:
            raise TranslationError(f"DeepL sent a reply this doesn't understand: {exc}", "unavailable") from exc
        if not translated.strip():
            raise TranslationError("DeepL sent back nothing", "unavailable")
        return translated

    def verify(self) -> tuple[bool, str]:
        """(ok, what to tell the person). Asks DeepL rather than guessing.

        A key box that accepts anything long enough is how an Ideogram
        key ended up in DEEPL_API_KEY, reading as "set" on the page while
        every run quietly came back in English. Shape rules would have
        caught that one and would also reject some future key DeepL
        decides to issue -- so the check is simply to use the key and see
        what DeepL says.

        An unreachable network is not a bad key: that answers ok, with a
        note saying it could not be confirmed.
        """
        if not self.is_configured():
            return False, "no key"
        import requests

        response = None
        for host in self.hosts():
            try:
                response = requests.get(
                    f"https://{host}/v2/usage",
                    headers={"Authorization": f"DeepL-Auth-Key {self.api_key}"},
                    timeout=15,
                )
            except Exception:  # noqa: BLE001
                return True, "couldn't reach DeepL to confirm the key -- saved anyway"
            if response.status_code not in (401, 403):
                break
        if response.status_code in (401, 403):
            return False, (
                "DeepL rejected that key on both its free and paid endpoints. A DeepL key "
                "looks like 0123abcd-4567-89ef-0123-456789abcdef, with \":fx\" on the end "
                "for a free account -- check you copied the DeepL one and not another "
                "service's, and that nothing was left off the end."
            )
        if response.status_code != 200:
            return True, f"DeepL answered {response.status_code} when asked about usage -- saved anyway"
        try:
            body = response.json()
            used, limit = body["character_count"], body["character_limit"]
        except Exception:  # noqa: BLE001
            return True, "key accepted"
        return True, f"{used:,} of {limit:,} characters used this month"

    def usage(self) -> dict | None:
        """{"character_count": n, "character_limit": n} or None.

        The point of paying a vendor -- even nothing -- is being able to
        ask how much is left before a demo rather than discovering the
        answer during one.
        """
        if not self.is_configured():
            return None
        import requests

        try:
            response = requests.get(
                self.endpoint.replace("/v2/translate", "/v2/usage"),
                headers={"Authorization": f"DeepL-Auth-Key {self.api_key}"},
                timeout=20,
            )
            if response.status_code != 200:
                return None
            body = response.json()
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(body, dict) or "character_count" not in body:
            return None
        return body


class GoogleFreeProvider(TranslationProvider):
    """The unauthenticated endpoint deep-translator wraps.

    Kept as the fallback, not the plan. It counts about five requests a
    second and 200,000 a day against whatever IP asks, tells you neither
    number, and answers an exhausted quota with an HTML error page that
    arrives as an ordinary string -- so a failed call reads as a
    successful translation unless someone checks.
    """

    name = "google-free"
    min_seconds_between_calls = 0.25

    # Rising, because a fixed delay against a per-second limit just
    # arrives late together.
    RATE_LIMIT_BACKOFF = (1.0, 2.0, 4.0)

    # Google's error page, as it looks coming back through the translator.
    ERROR_PAGE_MARKERS = (
        "that’s an error", "that's an error",
        "that’s all we know", "that's all we know",
        "server error", "error 500", "error 502", "error 503",
        "<!doctype", "<html",
    )

    @classmethod
    def looks_like_an_error_page(cls, text: str) -> bool:
        """Deliberately a fixed list of markers rather than a guess at
        what a translation "should" look like: a real translation can be
        any length, any script, and can legitimately contain the word
        "error". Two markers together, or one unmistakable one."""
        lowered = (text or "").lower()
        if "<!doctype" in lowered or "<html" in lowered:
            return True
        return sum(1 for marker in cls.ERROR_PAGE_MARKERS if marker in lowered) >= 2

    @staticmethod
    def is_a_rate_limit(exc: Exception) -> bool:
        """deep-translator raises TooManyRequests, but importing that
        class to catch it would make this depend on the library being
        installed -- which is one of the cases it has to survive."""
        return type(exc).__name__ == "TooManyRequests" or "too many requests" in str(exc).lower()

    def translate(self, text: str, language: str) -> str:
        last = TranslationError("the translator never answered", "unavailable")
        for attempt in range(1 + len(self.RATE_LIMIT_BACKOFF)):
            self.wait_our_turn()
            try:
                from deep_translator import GoogleTranslator

                # source="auto" rather than "en": what arrives here is not
                # always English. A PSD exported from an earlier Spanish run
                # carries Spanish in its type layers, and asking Google to
                # read Spanish as though it were English hands back something
                # barely touched -- "BEBIDA REFRESCANTE" came back
                # "REFRESCANTE BEBIDA" -- which the caller cannot tell from a
                # real translation. Letting Google detect it actually works.
                translated = GoogleTranslator(source="auto", target=language).translate(text)
            except Exception as exc:  # noqa: BLE001
                type(self)._last_call_at = time.monotonic()
                print("[localization] %s: %s: %s" % (language, type(exc).__name__, exc), file=sys.stderr)
                if self.is_a_rate_limit(exc):
                    last = TranslationError(str(exc), "rate limit")
                    if attempt < len(self.RATE_LIMIT_BACKOFF):
                        _pause(self.RATE_LIMIT_BACKOFF[attempt])
                        continue
                else:
                    last = TranslationError(str(exc), "unavailable")
                raise last from exc

            type(self)._last_call_at = time.monotonic()
            if translated and not self.looks_like_an_error_page(translated):
                return translated
            if translated:
                print(
                    "[localization] %s: the translator returned an error page, not a "
                    "translation: %r" % (language, translated[:120]),
                    file=sys.stderr,
                )
                raise TranslationError("an error page, not a translation", "error page")
            # An empty answer is not a translation, and asking again gets
            # the same nothing -- the endpoint is up and refusing.
            raise TranslationError("the translator sent back nothing", "unavailable")
        raise last


class MyMemoryProvider(TranslationProvider):
    """MyMemory's public API. No key, no account.

    Here because the keyless path had exactly one member. Google's free
    endpoint counts an invisible daily quota against whoever shares your
    IP, and when it ran out the whole feature went with it -- copy came
    back in English with no way to see why or when it would return. A
    second keyless provider turns that from an outage into a bad day for
    translation quality.
    """

    name = "mymemory"
    min_seconds_between_calls = 0.25

    ENDPOINT = "https://api.mymemory.translated.net/get"

    # MyMemory wants a language, not a locale, but is happier with the
    # common pairings for the ones that have regional variants.
    PAIRS = {"pt": "pt-BR", "zh-cn": "zh-CN", "zh": "zh-CN"}

    # It answers an exhausted allowance with 200 and a message in the
    # translation field, which is the same trap Google's error page set.
    REFUSALS = ("mymemory warning", "you used all available free translations",
                "no query specified", "invalid langpair", "please contact")

    def target_for(self, language: str) -> str:
        code = (language or "").strip().lower()
        return self.PAIRS.get(code, code)

    def translate(self, text: str, language: str) -> str:
        import requests

        self.wait_our_turn()
        try:
            response = requests.get(
                self.ENDPOINT,
                params={"q": text, "langpair": f"en|{self.target_for(language)}"},
                timeout=20,
            )
        except Exception as exc:  # noqa: BLE001
            raise TranslationError(f"{type(exc).__name__}: {exc}", "unavailable") from exc
        finally:
            type(self)._last_call_at = time.monotonic()

        if response.status_code == 429:
            raise TranslationError("MyMemory says slow down", "rate limit")
        if response.status_code != 200:
            raise TranslationError(f"MyMemory answered {response.status_code}", "unavailable")
        try:
            body = response.json()
            translated = (body.get("responseData") or {}).get("translatedText") or ""
        except Exception as exc:  # noqa: BLE001
            raise TranslationError(f"MyMemory sent a reply this doesn't understand: {exc}", "unavailable") from exc

        lowered = translated.lower()
        if any(refusal in lowered for refusal in self.REFUSALS):
            reason = "quota" if "all available free translations" in lowered else "unavailable"
            raise TranslationError(translated[:120], reason)
        if not translated.strip():
            raise TranslationError("MyMemory sent back nothing", "unavailable")
        if translated.strip().lower() == text.strip().lower():
            # Unchanged text is not a translation. Cached as one, it would
            # stand in front of the real thing for every later run.
            raise TranslationError("MyMemory returned the source unchanged", "unavailable")
        return translated


class LingvaProvider(TranslationProvider):
    """Lingva, a keyless front end onto Google's translations.

    Third in the keyless chain. Public instances come and go, so it tries
    more than one host rather than trusting any single volunteer server.
    """

    name = "lingva"
    min_seconds_between_calls = 0.25

    HOSTS = ("lingva.ml", "lingva.lunar.icu", "translate.plausibility.cloud")

    def translate(self, text: str, language: str) -> str:
        import urllib.parse

        import requests

        quoted = urllib.parse.quote(text, safe="")
        target = (language or "").strip().lower()
        last = TranslationError("no Lingva instance answered", "unavailable")
        for host in self.HOSTS:
            self.wait_our_turn()
            try:
                response = requests.get(
                    f"https://{host}/api/v1/en/{target}/{quoted}", timeout=15
                )
            except Exception as exc:  # noqa: BLE001
                last = TranslationError(f"{host}: {type(exc).__name__}", "unavailable")
                continue
            finally:
                type(self)._last_call_at = time.monotonic()
            if response.status_code != 200:
                last = TranslationError(f"{host} answered {response.status_code}", "unavailable")
                continue
            try:
                translated = (response.json() or {}).get("translation") or ""
            except Exception:  # noqa: BLE001
                last = TranslationError(f"{host} sent a reply this doesn't understand", "unavailable")
                continue
            if translated.strip() and translated.strip().lower() != text.strip().lower():
                return translated
            last = TranslationError(f"{host} returned nothing usable", "unavailable")
        raise last


def active_providers() -> list:
    """Who to ask, in order. DeepL first when a key is set, then the free
    endpoint -- so the app works with no key at all and gets better with
    one, rather than breaking without."""
    providers = []
    deepl = DeepLProvider()
    if deepl.is_configured():
        providers.append(deepl)
    # Three keyless providers, not one. The feature is meant to work with
    # nothing configured, and for a long time it did -- until the single
    # free endpoint it relied on hit a daily quota nobody could see, and
    # "choose Spanish" quietly started meaning "get English".
    providers.append(GoogleFreeProvider())
    providers.append(MyMemoryProvider())
    providers.append(LingvaProvider())
    return providers
