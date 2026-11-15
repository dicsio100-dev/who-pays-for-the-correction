"""Turn cached transcriptions into the study's numbers.

Definitions (fixed before looking at results):
  word error rate (WER)  errors / reference words, after normalise() on both sides
  misheard clip          a clip with at least one word error (one wrong digit = wrong address)
  agent policy           accept the clip if confidence >= t, otherwise ask the caller to repeat
  error that slips by    an accepted clip that was misheard
  TARGET                 the share of clips we allow to slip by; t is chosen on the TUNE
                         speakers (all groups pooled, like a company would), results are
                         reported on the TEST speakers only
  uncertainty            95% intervals from 1000 resamples of speakers (not clips)

Usage: python analyze.py [CONDITION ...]   (default: clean phone)
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict

import jiwer

from common import MANIFEST, RESULTS, normalise

MODEL = "whisper-large-v3"
TARGET = 0.05
BOOT = 1000
GROUP_ORDER = ["france", "canada", "afrique_subsaharienne", "cameroun_yaounde"]


def load(condition: str, tag: str = "") -> list[dict]:
    manifest = {r["clip_id"]: r for r in csv.DictReader(open(MANIFEST, encoding="utf-8", newline=""))}
    path = RESULTS / f"groq_{MODEL}_{condition}{'_' + tag if tag else ''}.jsonl"
    rows = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        m = manifest.get(d["clip_id"])
        if m is None or math.isnan(d["avg_logprob"]):
            continue
        ref, hyp = normalise(m["reference"]), normalise(d["text"])
        out = jiwer.process_words(ref, hyp)
        errors = out.substitutions + out.deletions + out.insertions
        rows.append({"clip_id": d["clip_id"], "group": m["group"], "speaker": m["speaker"],
                     "split": m["split"], "ref_words": len(ref.split()), "errors": errors,
                     "misheard": errors > 0, "conf": d["avg_logprob"], "ref": ref, "hyp": hyp})
    return rows


def wer(rows: list[dict]) -> float:
    words = sum(r["ref_words"] for r in rows)
    return sum(r["errors"] for r in rows) / words if words else float("nan")


def policy(rows: list[dict], t: float) -> tuple[float, float]:
    """(share asked to repeat, share of errors that slip by) for threshold t."""
    n = len(rows)
    repeat = sum(r["conf"] < t for r in rows) / n
    slip = sum(r["conf"] >= t and r["misheard"] for r in rows) / n
    return repeat, slip


def choose_threshold(tune: list[dict], target: float = TARGET) -> float:
    """Lowest threshold (fewest repeats) that keeps slipped errors at or under target."""
    for t in sorted({r["conf"] for r in tune}):
        if policy(tune, t)[1] <= target:
            return t
    return max(r["conf"] for r in tune) + 1e-9


def by_speaker(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["speaker"]].append(r)
    return out


def seeded(*key: object) -> random.Random:
    """One fixed random stream per statistic, so adding or reordering a computation never changes another's numbers."""
    return random.Random("|".join(map(str, (14, *key))))


def bootstrap(rows: list[dict], stat, rng: random.Random) -> tuple[float, float]:
    spk = by_speaker(rows)
    names = list(spk)
    vals = sorted(stat([c for s in (rng.choice(names) for _ in names) for c in spk[s]]) for _ in range(BOOT))
    return vals[int(0.025 * BOOT)], vals[int(0.975 * BOOT) - 1]


def p_two_sided(vals: list[float]) -> float:
    """Bootstrap p for 'no difference', looking both ways: twice the smaller tail around 0, capped at 1."""
    return min(1.0, 2 * min(sum(v <= 0 for v in vals), sum(v >= 0 for v in vals)) / len(vals))


def bootstrap_gap(rows: list[dict], base: list[dict], stat, rng: random.Random) -> tuple[float, float, float]:
    """95% interval of stat(rows) - stat(base), speakers resampled in each group independently,
    plus the two-sided bootstrap p (small = the gap is not chance)."""
    a, b = by_speaker(rows), by_speaker(base)
    na, nb = list(a), list(b)
    gaps = sorted(stat([c for s in (rng.choice(na) for _ in na) for c in a[s]]) -
                  stat([c for s in (rng.choice(nb) for _ in nb) for c in b[s]]) for _ in range(BOOT))
    return gaps[int(0.025 * BOOT)], gaps[int(0.975 * BOOT) - 1], p_two_sided(gaps)


def phone_widening(group: str, test_only: bool = False) -> tuple[float, float, float, float]:
    """Difference-in-differences: (group - France) WER gap on phone minus the same gap on clean audio.
    Same clips in both conditions; speakers resampled. Returns (observed, lo, hi, two-sided p)."""
    keep = (lambda r: r["split"] == "test") if test_only else (lambda r: True)
    sc, sp = by_speaker([r for r in load("clean") if keep(r)]), by_speaker([r for r in load("phone") if keep(r)])
    g_spk = [s for s in sc if sc[s][0]["group"] == group]
    f_spk = [s for s in sc if sc[s][0]["group"] == "france"]

    def stat(gs: list[str], fs: list[str]) -> float:
        def pick(d: dict, ss: list[str]) -> list[dict]:
            return [c for s in ss for c in d[s]]
        return (wer(pick(sp, gs)) - wer(pick(sp, fs))) - (wer(pick(sc, gs)) - wer(pick(sc, fs)))

    rng = seeded("phone_widening", group, test_only)
    vals = sorted(stat([rng.choice(g_spk) for _ in g_spk], [rng.choice(f_spk) for _ in f_spk]) for _ in range(BOOT))
    return stat(g_spk, f_spk), vals[int(0.025 * BOOT)], vals[int(0.975 * BOOT) - 1], p_two_sided(vals)


def decisions(rows: list[dict], crossfit: bool) -> tuple[list[dict], str]:
    """Attach r["t"] (the threshold that applies to the clip). Main: chosen on tune, report on test.
    Cross-fit (robustness, uses every speaker): each half is judged with the threshold chosen on the other."""
    halves = {s: [r for r in rows if r["split"] == s] for s in ("tune", "test")}
    if not crossfit:
        t = choose_threshold(halves["tune"])
        return [r | {"t": t} for r in halves["test"]], f"t = {t:.3f} chosen on tune, results on test"
    t_tune, t_test = choose_threshold(halves["tune"]), choose_threshold(halves["test"])
    out = [r | {"t": t_tune} for r in halves["test"]] + [r | {"t": t_test} for r in halves["tune"]]
    return out, f"cross-fit: t = {t_tune:.3f} / {t_test:.3f}, results on all speakers"


def repeat_share(rows: list[dict]) -> float:
    return sum(r["conf"] < r["t"] for r in rows) / len(rows)


def auc(rows: list[dict]) -> float:
    """How well the sentence score spots its own mistakes: the chance that a correctly heard clip scores higher
    than a misheard one (1 = perfect alarm, 0.5 = coin flip)."""
    pos = [r["conf"] for r in rows if r["misheard"]]
    neg = [r["conf"] for r in rows if not r["misheard"]]
    if not pos or not neg:
        return float("nan")
    return sum((n > q) + 0.5 * (n == q) for n in neg for q in pos) / (len(pos) * len(neg))


def mean_conf(rows: list[dict]) -> float:
    """Whisper's average sentence confidence (avg log-prob): lower = less sure = asked to repeat more, at ANY line."""
    return sum(r["conf"] for r in rows) / len(rows)


def slip_share(rows: list[dict]) -> float:
    return sum(r["conf"] >= r["t"] and r["misheard"] for r in rows) / len(rows)


def false_alarm(rows: list[dict]) -> float:
    """Among clips Whisper heard with no error, the share the rule still asks to repeat: the cost at equal error."""
    ok = [r for r in rows if not r["misheard"]]
    return sum(r["conf"] < r["t"] for r in ok) / len(ok) if ok else float("nan")


def report(condition: str, crossfit: bool = False) -> list[dict]:
    rows, how = decisions(load(condition), crossfit)
    groups = [g for g in GROUP_ORDER if any(r["group"] == g for r in rows)]
    base = [r for r in rows if r["group"] == groups[0]]
    table = []
    print(f"\n== {condition}: {how}, clips = {len(rows)}")
    print(f"{'group':24s} {'spk':>4s} {'WER':>14s} {'WER gap vs ' + groups[0]:>26s} "
          f"{'repeat':>16s} {'repeat gap':>22s} {'slip':>6s}")
    for g in groups:
        gr = [r for r in rows if r["group"] == g]
        key = (condition, crossfit, g)
        w_lo, w_hi = bootstrap(gr, wer, seeded(*key, "wer"))
        r_lo, r_hi = bootstrap(gr, repeat_share, seeded(*key, "repeat"))
        nan3 = (0.0, 0.0, float("nan"))
        wg = bootstrap_gap(gr, base, wer, seeded(*key, "wer_gap")) if g != groups[0] else nan3
        rg = bootstrap_gap(gr, base, repeat_share, seeded(*key, "repeat_gap")) if g != groups[0] else nan3
        sg = bootstrap_gap(gr, base, slip_share, seeded(*key, "slip_gap")) if g != groups[0] else nan3
        cg = bootstrap_gap(gr, base, mean_conf, seeded(*key, "conf_gap")) if g != groups[0] else nan3
        fg = bootstrap_gap(gr, base, false_alarm, seeded(*key, "false_alarm_gap")) if g != groups[0] else nan3
        rep = repeat_share(gr)
        table.append({"condition": condition, "crossfit": crossfit, "group": g, "speakers": len(by_speaker(gr)),
                      "clips": len(gr), "wer": wer(gr), "wer_lo": w_lo, "wer_hi": w_hi,
                      "wer_gap": wer(gr) - wer(base), "wer_gap_lo": wg[0], "wer_gap_hi": wg[1], "wer_gap_p": wg[2],
                      "repeat": rep, "repeat_lo": r_lo, "repeat_hi": r_hi,
                      "repeat_gap": rep - repeat_share(base), "repeat_gap_lo": rg[0], "repeat_gap_hi": rg[1],
                      "repeat_gap_p": rg[2], "slip": slip_share(gr),
                      "slip_gap": slip_share(gr) - slip_share(base), "slip_gap_lo": sg[0], "slip_gap_hi": sg[1], "slip_gap_p": sg[2],
                      "auc": auc(gr), "misheard": sum(r["misheard"] for r in gr) / len(gr),
                      "conf": mean_conf(gr), "conf_gap": mean_conf(gr) - mean_conf(base), "conf_gap_lo": cg[0],
                      "conf_gap_hi": cg[1], "conf_gap_p": cg[2], "repeat_ratio": rep / repeat_share(base),
                      "false_alarm": false_alarm(gr), "false_alarm_gap": false_alarm(gr) - false_alarm(base),
                      "false_alarm_gap_lo": fg[0], "false_alarm_gap_hi": fg[1], "false_alarm_gap_p": fg[2]})
        print(f"{g:24s} {len(by_speaker(gr)):4d} {wer(gr):6.1%} [{w_lo:.1%}-{w_hi:.1%}] "
              f"{table[-1]['wer_gap']:+6.1%} [{wg[0]:+.1%} {wg[1]:+.1%}] p={wg[2]:.3f} "
              f"{rep:6.1%} [{r_lo:.0%}-{r_hi:.0%}] {table[-1]['repeat_gap']:+6.1%} [{rg[0]:+.0%} {rg[1]:+.0%}] p={rg[2]:.3f} "
              f"{slip_share(gr):6.1%} {table[-1]['slip_gap']:+6.1%} [{sg[0]:+.0%} {sg[1]:+.0%}] p={sg[2]:.3f}")
    return table


def _self_check() -> None:
    rows = [{"conf": c, "misheard": m} for c, m in [(-0.1, False), (-0.2, True), (-0.5, True), (-0.9, False)]]
    assert policy(rows, -0.3) == (0.5, 0.25)          # two below t, one accepted error
    assert auc(rows) == 0.5                           # correct -0.1 beats both misheard (-0.2, -0.5), correct -0.9 beats neither: 2 of 4
    assert false_alarm([r | {"t": -0.3} for r in rows]) == 0.5  # of the two correct clips, only -0.9 is asked to repeat
    assert choose_threshold(rows) == -0.1 or policy(rows, choose_threshold(rows))[1] <= TARGET


if __name__ == "__main__":
    _self_check()
    all_rows = []
    for cond in sys.argv[1:] or ["clean", "phone"]:
        if (RESULTS / f"groq_{MODEL}_{cond}.jsonl").exists():
            all_rows += report(cond) + report(cond, crossfit=True)
    if all_rows:
        with open(RESULTS / "summary.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nsaved {RESULTS / 'summary.csv'}")
