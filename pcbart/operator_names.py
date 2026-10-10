"""The name to print on the 干员代号 line.

That line is set in the template's own ``Agency FB Negreta`` -- a Latin-only
face. A Chinese name renders as a row of empty boxes, so the line wants the
operator's **English** name: it is both what the design intends and what makes
the template's typeface usable at all.

Names come from :mod:`pcbart.operator_names_en`, a generated table of the
``data-en`` attribute PRTS publishes on its operator grid. Several kinds of
name are not in it, and each has a different honest answer:

* **the name is already Latin** (``Pith``, ``Sharp``, ``Stormeye``, ``Touch``)
  -- print it as-is; that *is* the official name;
* **a Latin run inside a Chinese name** (``领主·Sharp``, ``Mechanist(卫戍协议)``)
  -- print the Latin run, because the game itself spells that part in Latin;
* **nothing Latin at all** (``博士``, ``预备干员-狙击``) -- fall back to the
  Chinese name. Inventing a transliteration would put a name on the card that
  no one can look up.
"""

from __future__ import annotations

import re

__all__ = [
    "NAME_EN",
    "english_name",
    "latin_part",
    "has_latin",
]

try:
    from .operator_names_en import NAME_EN
except ImportError:  # pragma: no cover - the table ships with the package
    NAME_EN: dict[str, str] = {}

#: A run of Latin letters with any internal punctuation, which is what a name
#: like ``Mechanist`` or ``Kal'tsit`` looks like. Two letters minimum: a single
#: one is as likely to be from a Chinese phrase as from a name.
_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z0-9'’.\-]{1,}")


def latin_part(name: str) -> str:
    """The longest Latin run inside ``name``, or ``""``.

    ``Mechanist(卫戍协议)`` gives ``Mechanist``; ``领主·Sharp`` gives ``Sharp``;
    ``预备干员-狙击`` gives ``""`` -- the ``-`` is between two Chinese words, not
    inside a Latin one, which is why the pattern has to *start* with a letter.
    """
    text = str(name or "")
    runs = _LATIN_RUN.findall(text)
    if not runs:
        return ""
    return max(runs, key=len)


def has_latin(name: str) -> bool:
    """Whether any Latin run was found in ``name``."""
    return bool(latin_part(name))


def english_name(name: str) -> str:
    """What to print on the 干员代号 line for ``name``.

    Falls back to the Chinese name rather than a guess -- see the module
    docstring for the three cases.
    """
    text = str(name or "").strip()
    if not text:
        return ""
    published = NAME_EN.get(text, "")
    if published:
        return published
    return latin_part(text) or text
