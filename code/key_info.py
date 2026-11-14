"""Second definition of "misheard", closer to what a business needs: only the KEY words count.

Defined after seeing the first results (disclosed): a phone agent must get names and numbers right,
not every "le" or "de". In Common Voice, key words are
  - names: a capitalised word that does not start the sentence (sentence-initial names cannot be told apart
    from ordinary capitals, so they are left out);
  - numbers: any word containing a digit.
A clip is "key-misheard" if any key word is replaced or missing in Whisper's text. Only clips that contain
at least one key word are used. Yaoundé proverbs contain no names, so the group drops out here.

Reports, per accent (whisper-large-v3, all speakers):
  key-word error rate, with 95 % range and two-sided p of the gap with France;
  the repeat rule re-run with key-misheard as the thing to catch (cross-fitted: line tuned on one half,
  judged on the other, then swapped).
Usage: python key_info.py   ->  results/key_info.csv
"""
from __future__ import annotations

import csv
import json
import re

import jiwer

from analyze import (GROUP_ORDER, bootstrap, bootstrap_gap, by_speaker, decisions, repeat_share, seeded,
                     slip_share)
from common import MANIFEST, RESULTS, normalise

DIGIT = re.compile(r"\d")


def key_marks(sentence: str) -> tuple[list[str], list[bool]]:
    """Reference tokens (normalised word by word) and whether each one belongs to a key word."""
    toks, keys = [], []
    for i, word in enumerate(sentence.split()):
        bare = word.strip("«»\"'()[],.;:!?")
        is_key = bool(DIGIT.search(word)) or (i > 0 and bare[:1].isupper())
        for t in normalise(word).split():
            toks.append(t)
            keys.append(is_key)
    return toks, keys


def key_errors(ref: list[str], keys: list[bool], hyp: list[str]) -> tuple[int, int]:
    """(key tokens wrong, other tokens wrong): a reference token is wrong if replaced or deleted."""
    if not hyp:
        return sum(keys), len(keys) - sum(keys)
    wrong_key = wrong_other = 0
    for c in jiwer.process_words(" ".join(ref), " ".join(hyp)).alignments[0]:
        if c.type in ("substitute", "delete"):
            for i in range(c.ref_start_idx, c.ref_end_idx):
                if keys[i]:
                    wrong_key += 1
                else:
                    wrong_other += 1
    return wrong_key, wrong_other


def load_key(condition: str) -> list[dict]:
    manifest = {r["clip_id"]: r for r in csv.DictReader(open(MANIFEST, encoding="utf-8", newline=""))}
    rows = []
    for line in open(RESULTS / f"groq_whisper-large-v3_{condition}.jsonl", encoding="utf-8"):
        d = json.loads(line)
        m = manifest[d["clip_id"]]
        ref, keys = key_marks(m["reference"])
        if not any(keys):
            continue
        hyp = [t for w in d["text"].split() for t in normalise(w).split()]
        wk, wo = key_errors(ref, keys, hyp)
        rows.append({"clip_id": d["clip_id"], "group": m["group"], "speaker": m["speaker"], "split": m["split"],
                     "conf": d["avg_logprob"], "key_words": sum(keys), "key_wrong": wk,
                     "other_words": len(keys) - sum(keys), "other_wrong": wo, "misheard": wk > 0})
    return rows


def key_rate(rows: list[dict]) -> float:
    n = sum(r["key_words"] for r in rows)
    return sum(r["key_wrong"] for r in rows) / n if n else float("nan")


def other_rate(rows: list[dict]) -> float:
    n = sum(r["other_words"] for r in rows)
    return sum(r["other_wrong"] for r in rows) / n if n else float("nan")


def main() -> None:
    table = []
    for cond in ("clean", "phone"):
        rows = load_key(cond)
        judged, how = decisions(rows, crossfit=True)
        groups = [g for g in GROUP_ORDER if any(r["group"] == g for r in rows)]
        base, base_j = [r for r in rows if r["group"] == "france"], [r for r in judged if r["group"] == "france"]
        print(f"\n== {cond}: clips with a name or number = {len(rows)}; repeat rule {how}")
        print(f"{'group':24s} {'spk':>4s} {'clips':>5s} {'key words wrong':>26s} {'gap vs France':>24s} "
              f"{'other words wrong':>18s} {'repeat':>7s} {'repeat gap':>20s} {'slip':>6s}")
        for g in groups:
            gr, gj = [r for r in rows if r["group"] == g], [r for r in judged if r["group"] == g]
            lo, hi = bootstrap(gr, key_rate, seeded("key", cond, g, "rate"))
            kg = bootstrap_gap(gr, base, key_rate, seeded("key", cond, g, "gap")) if g != "france" else (0.0, 0.0, float("nan"))
            rg = bootstrap_gap(gj, base_j, repeat_share, seeded("key", cond, g, "repeat_gap")) if g != "france" else (0.0, 0.0, float("nan"))
            row = {"condition": cond, "group": g, "speakers": len(by_speaker(gr)), "clips": len(gr),
                   "key_words": sum(r["key_words"] for r in gr), "key_rate": key_rate(gr), "key_lo": lo, "key_hi": hi,
                   "key_gap": key_rate(gr) - key_rate(base), "key_gap_lo": kg[0], "key_gap_hi": kg[1], "key_gap_p": kg[2],
                   "other_rate": other_rate(gr), "repeat": repeat_share(gj), "repeat_gap": repeat_share(gj) - repeat_share(base_j),
                   "repeat_gap_lo": rg[0], "repeat_gap_hi": rg[1], "repeat_gap_p": rg[2], "slip": slip_share(gj)}
            table.append(row)
            print(f"{g:24s} {row['speakers']:4d} {row['clips']:5d} {row['key_rate']:7.1%} [{lo:.1%}-{hi:.1%}] "
                  f"{row['key_gap']:+6.1%} [{kg[0]:+.1%} {kg[1]:+.1%}] p={kg[2]:.3f} {row['other_rate']:9.1%} "
                  f"{row['repeat']:7.1%} {row['repeat_gap']:+6.1%} p={rg[2]:.3f} {row['slip']:6.1%}")
    with open(RESULTS / "key_info.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)


def _self_check() -> None:
    toks, keys = key_marks("Le district de Mugla a 2 lacs.")
    assert toks == ["le", "district", "de", "mugla", "a", "2", "lacs"] and keys == [0, 0, 0, 1, 0, 1, 0]
    assert key_errors(toks, keys, ["le", "district", "de", "mongla", "a", "2", "lac"]) == (1, 1)
    assert key_errors(toks, keys, []) == (2, 5)


if __name__ == "__main__":
    _self_check()
    main()
