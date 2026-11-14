"""Robustness checks: could something other than accent explain the gap?

  1. recording quality  rough signal-to-noise ratio per clip (loud frames vs quiet frames)
  2. proper nouns       WER again, without sentences that contain a rare name
  3. sentence length    share of misheard clips if every group had France's sentence lengths
  4. noisy-mic subset   WER again, keeping only clips whose SNR is above the overall median

Usage: python checks.py [CONDITION]   (default: clean)
"""
from __future__ import annotations

import csv
import re
import statistics as st
import sys
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf

from analyze import GROUP_ORDER, load, wer
from common import MANIFEST

# A capitalised word that does not start the sentence = a name (Common Voice comes from Wikipedia).
PROPER = re.compile(r"(?<=\s)[A-ZÀÂÇÉÈÊËÎÏÔÛÙÜŸ][\wÀ-ÿ'-]+")


def snr_db(path: str) -> float:
    audio, sr = sf.read(path, dtype="float32")
    frame = int(0.025 * sr)
    n = len(audio) // frame
    if n < 10:
        return float("nan")
    energy = (audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1)
    energy = energy[energy > 1e-9]  # drop digital silence (zero padding) so the floor is the real room noise
    if len(energy) < 10:
        return float("nan")
    return float(10 * np.log10(np.percentile(energy, 95) / np.percentile(energy, 10)))


def length_standardised_misheard(rows: list[dict], ref_dist: Counter) -> float:
    """P(misheard) averaged over France's sentence-length mix (direct standardisation, 3-word bins)."""
    by_bin: dict[int, list[bool]] = defaultdict(list)
    for r in rows:
        by_bin[r["ref_words"] // 3].append(r["misheard"])
    total = sum(ref_dist.values())
    num = den = 0.0
    for b, w in ref_dist.items():
        if by_bin.get(b):
            num += w / total * st.mean(by_bin[b])
            den += w / total
    return num / den if den else float("nan")


def main(condition: str) -> dict[str, dict]:
    manifest = {r["clip_id"]: r for r in csv.DictReader(open(MANIFEST, encoding="utf-8", newline=""))}
    rows = load(condition)
    for r in rows:
        m = manifest[r["clip_id"]]
        r["proper"] = bool(PROPER.search(m["reference"]))
        r["snr"] = snr_db(m[condition])
    median_snr = st.median(r["snr"] for r in rows)
    france_len = Counter(r["ref_words"] // 3 for r in rows if r["group"] == "france")

    print(f"== {condition} (all speakers, tune + test), median SNR {median_snr:.1f} dB")
    print(f"{'group':24s} {'SNR dB':>7s} {'WER':>6s} {'WER no-name':>12s} {'WER good-mic':>13s} "
          f"{'misheard':>9s} {'misheard @FR length':>20s}")
    out = {}
    for g in GROUP_ORDER:
        gr = [r for r in rows if r["group"] == g]
        if not gr:
            continue
        no_name = [r for r in gr if not r["proper"]]
        good = [r for r in gr if r["snr"] >= median_snr]
        out[g] = {"snr_db": st.median(r["snr"] for r in gr), "wer": wer(gr), "wer_no_name": wer(no_name),
                  "clips_no_name": len(no_name), "wer_good_mic": wer(good), "clips_good_mic": len(good),
                  "misheard": st.mean(r["misheard"] for r in gr),
                  "misheard_fr_length": length_standardised_misheard(gr, france_len),
                  "mean_words": st.mean(r["ref_words"] for r in gr)}
        print(f"{g:24s} {out[g]['snr_db']:7.1f} {wer(gr):6.1%} "
              f"{wer(no_name):7.1%} ({len(no_name):3d}) {wer(good):8.1%} ({len(good):3d}) "
              f"{out[g]['misheard']:9.1%} {out[g]['misheard_fr_length']:20.1%}")
    return out


def _self_check() -> None:
    assert PROPER.search("le district de Mugla a été renommé")
    assert not PROPER.search("Ses œuvres conservées sont rares")
    rows = [{"ref_words": 3, "misheard": True}, {"ref_words": 9, "misheard": False}]
    assert length_standardised_misheard(rows, Counter({1: 1, 3: 1})) == 0.5


if __name__ == "__main__":
    _self_check()
    main(sys.argv[1] if len(sys.argv) > 1 else "clean")
