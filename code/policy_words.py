"""Two ways for a phone agent to double-check what it heard, both on a local Whisper's per-word confidence.

  sentence   ask the caller to repeat the whole sentence if the sentence's mean log-confidence < t
  targeted   ask the caller to confirm only the words whose confidence < t

What the caller pays   words re-said per call (whole sentence for a repeat, flagged words for targeted)
                       and the share of callers interrupted at all
What slips by          share of calls where at least one error is not caught
Error caught (targeted) if any word of the wrong region is flagged; a missing word is caught if the
                       word just before or after it is flagged
Assumption (same for both policies, optimistic for both): what the caller re-says is then heard right.
Threshold t is chosen on TUNE speakers (all groups pooled) for slip <= TARGET; results on TEST speakers.

Usage: python policy_words.py [CONDITION] [MODEL]   (default: phone turbo)
Output: results/policy_<condition>_<model>.csv (table), results/curves_<condition>_<model>.csv,
        results/policy_gaps_<condition>_<model>.csv (words saved per group, gap with France under each rule)
"""
from __future__ import annotations

import csv
import json
import math
import sys

import jiwer

from analyze import BOOT, GROUP_ORDER, TARGET, bootstrap, bootstrap_gap, by_speaker, p_two_sided, seeded, wer
from common import MANIFEST, RESULTS, normalise


def tokens(text: str) -> list[str]:
    """Normalise word by word, so every token keeps the confidence of the word it came from."""
    return [t for w in text.split() for t in normalise(w).split()]


def load_local(condition: str, model: str) -> list[dict]:
    manifest = {r["clip_id"]: r for r in csv.DictReader(open(MANIFEST, encoding="utf-8", newline=""))}
    rows = []
    for line in open(RESULTS / f"local_{model}_{condition}.jsonl", encoding="utf-8"):
        d = json.loads(line)
        m = manifest[d["clip_id"]]
        ref = tokens(m["reference"])
        hyp, probs, word_logp = [], [], []
        for w in d["words"]:
            p = max(w["p"], 1e-6)
            word_logp.append(math.log(p))
            for t in normalise(w["w"]).split():
                hyp.append(t)
                probs.append(p)
        regions = []  # for each error: the confidences of the hypothesis words that could reveal it
        errors = len(ref)
        if hyp:
            out = jiwer.process_words(" ".join(ref), " ".join(hyp))
            errors = out.substitutions + out.deletions + out.insertions
            for c in out.alignments[0]:
                if c.type in ("substitute", "insert"):
                    regions.append(probs[c.hyp_start_idx:c.hyp_end_idx])
                elif c.type == "delete":
                    regions.append([probs[i] for i in (c.hyp_start_idx - 1, c.hyp_start_idx) if 0 <= i < len(hyp)])
        else:
            regions = [[]]
        rows.append({"clip_id": d["clip_id"], "group": m["group"], "speaker": m["speaker"], "split": m["split"],
                     "ref_words": len(ref), "errors": errors, "probs": probs, "misheard": bool(regions), "empty": not hyp,
                     # clip slips under targeted(t) iff some error region has no word below t
                     "slip_level": max((min(r, default=math.inf) for r in regions), default=-math.inf),
                     "sent_conf": sum(word_logp) / len(word_logp) if word_logp else -math.inf})
    return rows


def sentence(rows: list[dict], t: float) -> tuple[float, float, float]:
    """(interrupted share, words re-said per call, slip share)."""
    n = len(rows)
    rep = [r["sent_conf"] < t for r in rows]
    return (sum(rep) / n, sum(r["ref_words"] for r, x in zip(rows, rep) if x) / n,
            sum(r["misheard"] and not x for r, x in zip(rows, rep)) / n)


def targeted(rows: list[dict], t: float) -> tuple[float, float, float]:
    n = len(rows)
    flagged = [r["ref_words"] if r["empty"] else sum(p < t for p in r["probs"]) for r in rows]
    slip = sum(not r["empty"] and r["slip_level"] >= t for r in rows) / n
    return sum(f > 0 for f in flagged) / n, sum(flagged) / n, slip


POLICIES = {"sentence": (sentence, "sent_conf"), "targeted": (targeted, "probs")}


def candidates(rows: list[dict], key: str) -> list[float]:
    vals = {v for r in rows for v in (r[key] if key == "probs" else [r[key]]) if math.isfinite(v)}
    return sorted(vals) + [max(vals) + 1e-9]


def choose(rows: list[dict], name: str) -> float:
    fn, key = POLICIES[name]
    for t in candidates(rows, key):
        if fn(rows, t)[2] <= TARGET:
            return t
    return math.inf


def main(condition: str, model: str) -> None:
    rows = load_local(condition, model)
    tune = [r for r in rows if r["split"] == "tune"]
    test = [r for r in rows if r["split"] == "test"]
    groups = [g for g in GROUP_ORDER if any(r["group"] == g for r in test)]
    table, curves = [], []
    print(f"== Whisper {model}, {condition}: test clips {len(test)}, target slip {TARGET:.0%}")
    print(f"{'policy':9s} {'group':24s} {'interrupted':>17s} {'words re-said/call':>22s} {'slip':>6s} {'x france':>9s}")
    for name, (fn, key) in POLICIES.items():
        t = choose(tune, name)
        base = fn([r for r in test if r["group"] == groups[0]], t)[1]
        for g in groups:
            gr = [r for r in test if r["group"] == g]
            inter, words, slip = fn(gr, t)
            i_lo, i_hi = bootstrap(gr, lambda x: fn(x, t)[0], seeded(model, condition, name, g, "interrupted"))
            w_lo, w_hi = bootstrap(gr, lambda x: fn(x, t)[1], seeded(model, condition, name, g, "words"))
            table.append({"condition": condition, "policy": name, "threshold": t, "group": g,
                          "speakers": len(by_speaker(gr)), "clips": len(gr), "misheard": sum(r["misheard"] for r in gr) / len(gr),
                          "interrupted": inter, "interrupted_lo": i_lo, "interrupted_hi": i_hi,
                          "words": words, "words_lo": w_lo, "words_hi": w_hi, "slip": slip,
                          "words_ratio": words / base if base else float("nan")})
            print(f"{name:9s} {g:24s} {inter:6.1%} [{i_lo:.0%}-{i_hi:.0%}] {words:8.2f} [{w_lo:.2f}-{w_hi:.2f}] "
                  f"{slip:6.1%} {table[-1]['words_ratio']:8.2f}x")
        for g in groups:
            gr = [r for r in test if r["group"] == g]
            for tc in candidates(gr, key)[::3]:
                inter, words, slip = fn(gr, tc)
                curves.append({"condition": condition, "policy": name, "group": g, "threshold": tc,
                               "interrupted": inter, "words": words, "slip": slip})
    for g in groups:
        gr = [r for r in test if r["group"] == g]
        print(f"never     {g:24s} slip = misheard share {sum(r['misheard'] for r in gr) / len(gr):.1%}   {model} WER {wer(gr):.1%}")
    gaps = gap_table(tune, test, groups, condition, model)
    for path, data in ((f"policy_{condition}_{model}.csv", table), (f"curves_{condition}_{model}.csv", curves),
                       (f"policy_gaps_{condition}_{model}.csv", gaps)):
        with open(RESULTS / path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(f"saved results/policy_{condition}_{model}.csv and results/curves_{condition}_{model}.csv")


def gap_table(tune: list[dict], test: list[dict], groups: list[str], condition: str, model: str) -> list[dict]:
    """Per group: words saved by targeted vs sentence (same calls, speakers resampled), and the
    words-re-said gap with France under each rule, plus how much targeted changes that gap."""
    ts, tt = choose(tune, "sentence"), choose(tune, "targeted")
    saved = lambda x: sentence(x, ts)[1] - targeted(x, tt)[1]
    base = [r for r in test if r["group"] == groups[0]]
    out = []
    print(f"\n{'group':24s} {'words saved':>34s}   {'gap vs France, sentence':>34s}   {'gap, targeted':>34s}   {'gap change':>34s}")
    for g in groups:
        gr = [r for r in test if r["group"] == g]
        rng = seeded(model, condition, "saved", g)
        spk = by_speaker(gr)
        names = list(spk)
        vals = sorted(saved([c for s in (rng.choice(names) for _ in names) for c in spk[s]]) for _ in range(BOOT))
        row = {"group": g, "saved": saved(gr), "saved_lo": vals[int(0.025 * BOOT)],
               "saved_hi": vals[int(0.975 * BOOT) - 1], "saved_p": p_two_sided(vals),
               "saved_share": saved(gr) / sentence(gr, ts)[1]}
        if g != groups[0]:
            for name, fn, t in (("sentence", sentence, ts), ("targeted", targeted, tt)):
                lo, hi, pv = bootstrap_gap(gr, base, lambda x, fn=fn, t=t: fn(x, t)[1], seeded(model, condition, "gap", name, g))
                row |= {f"gap_{name}": fn(gr, t)[1] - fn(base, t)[1], f"gap_{name}_lo": lo, f"gap_{name}_hi": hi, f"gap_{name}_p": pv}
            # does targeted shrink the gap? (sentence gap - targeted gap) = saved(group) - saved(France)
            lo, hi, pv = bootstrap_gap(gr, base, saved, seeded(model, condition, "gap_change", g))
            row |= {"gap_change": saved(gr) - saved(base), "gap_change_lo": lo, "gap_change_hi": hi, "gap_change_p": pv}
            # added Sunday (post-hoc): sentences are not the same length in every group, so also give the gap
            # per word spoken (share of the caller's words said again), shown in words at France's sentence length
            fr_len = sum(r["ref_words"] for r in base) / len(base)
            for name, fn, t in (("sentence", sentence, ts), ("targeted", targeted, tt)):
                share = lambda x, fn=fn, t=t: fn(x, t)[1] / (sum(r["ref_words"] for r in x) / len(x))
                lo, hi, pv = bootstrap_gap(gr, base, share, seeded(model, condition, "gap_eqlen", name, g))
                row |= {f"gap_{name}_eqlen": (share(gr) - share(base)) * fr_len, f"gap_{name}_eqlen_lo": lo * fr_len,
                        f"gap_{name}_eqlen_hi": hi * fr_len, f"gap_{name}_eqlen_p": pv}
        row["sentence_words"] = sum(r["ref_words"] for r in gr) / len(gr)
        out.append(row)
        f = lambda k: f"{row[k]:+.2f} [{row[k + '_lo']:+.2f},{row[k + '_hi']:+.2f}] p={row[k + '_p']:.3f}" if k in row else ""
        print(f"{g:24s} {f('saved'):>34s}   {f('gap_sentence'):>34s}   {f('gap_targeted'):>34s}   {f('gap_change'):>34s}")
    keys = list(out[-1])
    return [{k: r.get(k, "") for k in keys} for r in out]


def _self_check() -> None:
    r = {"ref_words": 3, "probs": [0.9, 0.2, 0.95], "empty": False, "slip_level": 0.2,
         "misheard": True, "sent_conf": -0.5}
    assert targeted([r], 0.5) == (1.0, 1.0, 0.0)        # the doubtful word is flagged: error caught, 1 word
    assert targeted([r], 0.1) == (0.0, 0.0, 1.0)        # nothing flagged: error slips
    assert sentence([r], -0.4) == (1.0, 3.0, 0.0)       # whole sentence repeated
    assert tokens("L'appel, vingt-deux") == ["l", "appel", "22"]


if __name__ == "__main__":
    _self_check()
    main(sys.argv[1] if len(sys.argv) > 1 else "phone", sys.argv[2] if len(sys.argv) > 2 else "turbo")
