"""User breed/coat text helpers. Raw text is always stored unchanged next to these forms."""

import re
import unicodedata

_SEPARATORS = re.compile(r"[^\w+]+", re.UNICODE)


def normalize_text(value: str | None) -> str | None:
    """Search form only: NFKC, casefold, ё->е, punctuation to spaces. Never replaces raw input.

    "+" is kept so that mixed-breed text such as "немецкая овчарка + терьер" stays readable.
    """
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    text = _SEPARATORS.sub(" ", text).replace("_", " ")
    text = " ".join(text.split())
    return text or None
