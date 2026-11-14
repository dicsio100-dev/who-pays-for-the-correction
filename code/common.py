"""Shared paths and text normalisation for the accent study.

Every transcript (the reference and Whisper's output) goes through the same
normalise() before we count errors, so differences in spelling style
("vingt-deux" vs "22", "cœur" vs "coeur", "n' en" vs "n'en") are not counted
as recognition errors.
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

from text_to_num import alpha2digit

DATA = Path(os.environ.get("VERITAS_DATA", Path.home() / "veritas-data"))
RAW = DATA / "raw"
AUDIO = DATA / "audio"          # 16 kHz mono WAV copies we actually send to Whisper
RESULTS = DATA / "results"      # transcription caches (JSONL) and tables (CSV)
MANIFEST = DATA / "manifest.csv"

_APOS = re.compile(r"[’`´ʼ]")
_SPACED_APOS = re.compile(r"(\w)'\s+")
_NON_WORD = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    text = text.replace("œ", "oe").replace("æ", "ae").replace("Œ", "oe").replace("Æ", "ae")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def normalise(text: str) -> str:
    """Lower-case, numbers as digits, no accents, no punctuation, single spaces."""
    text = _APOS.sub("'", text.lower())
    text = _SPACED_APOS.sub(r"\1'", text)            # Yaoundé transcripts write "n' en"
    text = alpha2digit(text, "fr")                    # "vingt-deux" -> "22", keeps "un"/"une"
    text = strip_accents(text)
    text = text.replace("'", " ").replace("-", " ")  # "l'abondance" -> "l abondance" on both sides
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def _self_check() -> None:
    cases = {
        "La bouche parle de l' abondance du coeur.": "la bouche parle de l abondance du coeur",
        "La bouche parle de l'abondance du cœur": "la bouche parle de l abondance du coeur",
        "Il habite au vingt-deux, rue Saint-Joseph.": "il habite au 22 rue saint joseph",
        "Il habite au 22 rue Saint-Joseph": "il habite au 22 rue saint joseph",
        "Épargne de bouche vaut rente de pré": "epargne de bouche vaut rente de pre",
        "Une fuite, un dégât": "une fuite un degat",
    }
    for raw, want in cases.items():
        got = normalise(raw)
        assert got == want, f"normalise({raw!r}) = {got!r}, want {want!r}"


_self_check()
