"""The fix the brief suggests: tell Whisper the language in advance. Does it help, and who does it help most?

Same test clips (phone line), transcribed twice by whisper-large-v3 on Groq:
  baseline   no language given (Whisper guesses it)          results/groq_..._phone_autolang.jsonl
  fix        language = "fr"                                  results/groq_..._phone.jsonl
Reports per group: how often Whisper guessed a language other than French, WER without and with the hint,
and the paired gain (speakers resampled, same clips in both).

Usage: python language_hint.py   ->  results/language_hint.csv
"""
from __future__ import annotations

import csv
import json
from collections import Counter

import jiwer

from analyze import BOOT, GROUP_ORDER, by_speaker, load, p_two_sided, seeded, wer
from common import MANIFEST, RESULTS, normalise


def load_autolang() -> dict[str, dict]:
    manifest = {r["clip_id"]: r for r in csv.DictReader(open(MANIFEST, encoding="utf-8", newline=""))}
    out = {}
    for line in open(RESULTS / "groq_whisper-large-v3_phone_autolang.jsonl", encoding="utf-8"):
        d = json.loads(line)
        ref, hyp = normalise(manifest[d["clip_id"]]["reference"]), normalise(d["text"])
        o = jiwer.process_words(ref, hyp)
        out[d["clip_id"]] = {"errors": o.substitutions + o.deletions + o.insertions, "language": d["language"],
                             "text": d["text"]}
    return out


def main() -> None:
    auto = load_autolang()
    fixed = [r for r in load("phone") if r["clip_id"] in auto]
    table = []
    spk_fix = by_speaker(fixed)
    spk_auto = by_speaker([r | {"errors": auto[r["clip_id"]]["errors"]} for r in fixed])
    fr_names = [s for s in spk_fix if spk_fix[s][0]["group"] == "france"]

    def gain_of(ss: list[str]) -> float:
        return wer([c for s in ss for c in spk_auto[s]]) - wer([c for s in ss for c in spk_fix[s]])

    print(f"== language hint, phone line, {len(fixed)} test clips")
    print(f"{'group':24s} {'wrong language':>15s} {'WER no hint':>12s} {'WER fr hint':>12s} {'gain':>22s}")
    for g in GROUP_ORDER:
        gr = [r for r in fixed if r["group"] == g]
        if not gr:
            continue
        base = [r | {"errors": auto[r["clip_id"]]["errors"]} for r in gr]
        langs = Counter(auto[r["clip_id"]]["language"] for r in gr)
        wrong = sum(v for k, v in langs.items() if k.lower() not in ("french", "fr"))
        spk_b, spk_f = by_speaker(base), by_speaker(gr)
        names = list(spk_f)

        def gain(ss: list[str]) -> float:
            return wer([c for s in ss for c in spk_b[s]]) - wer([c for s in ss for c in spk_f[s]])

        rng = seeded("language_hint", g, "gain")
        vals = sorted(gain([rng.choice(names) for _ in names]) for _ in range(BOOT))
        g_obs, lo, hi = gain(names), vals[int(0.025 * BOOT)], vals[int(0.975 * BOOT) - 1]
        table.append({"group": g, "clips": len(gr), "wrong_language": wrong, "languages": dict(langs),
                      "wer_no_hint": wer(base), "wer_fr_hint": wer(gr), "gain": g_obs, "gain_lo": lo, "gain_hi": hi,
                      "gain_p": p_two_sided(vals)})
        if g != "france":  # how much the hint narrows this group's gap with France (both groups resampled)
            rng = seeded("language_hint", g, "narrowing")
            nar = sorted(gain_of([rng.choice(names) for _ in names]) - gain_of([rng.choice(fr_names) for _ in fr_names])
                         for _ in range(BOOT))
            table[-1] |= {"gap_no_hint": wer(base) - wer([c for s in fr_names for c in spk_auto[s]]),
                          "gap_fr_hint": wer(gr) - wer([c for s in fr_names for c in spk_fix[s]]),
                          "narrowing": gain_of(names) - gain_of(fr_names), "narrowing_lo": nar[int(0.025 * BOOT)],
                          "narrowing_hi": nar[int(0.975 * BOOT) - 1], "narrowing_p": p_two_sided(nar)}
        print(f"{g:24s} {wrong:3d}/{len(gr):3d} {wer(base):12.1%} {wer(gr):12.1%} "
              f"{g_obs:+6.1%} [{lo:+.1%} {hi:+.1%}] p={table[-1]['gain_p']:.3f}   {dict(langs)}")
    for r in table[1:]:
        print(f"  {r['group']}: gap with France {r['gap_no_hint']:+.1%} -> {r['gap_fr_hint']:+.1%}, narrowed by "
              f"{r['narrowing']:+.1%} [{r['narrowing_lo']:+.1%} {r['narrowing_hi']:+.1%}] p={r['narrowing_p']:.3f}")
    for r in table:
        print(f"  wrong-language examples ({r['group']}):")
        for cid in [c["clip_id"] for c in fixed if c["group"] == r["group"]
                    and auto[c["clip_id"]]["language"].lower() not in ("french", "fr")][:3]:
            print(f"    [{auto[cid]['language']}] {auto[cid]['text'][:90]}")
    with open(RESULTS / "language_hint.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[-1]), restval="")
        w.writeheader()
        w.writerows(table)


if __name__ == "__main__":
    main()
