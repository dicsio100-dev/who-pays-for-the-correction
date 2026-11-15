"""More fixes for the 'please repeat' rule, tested on the cached phone-line transcriptions (no new audio sent).
Added on Sunday after the first results, so everything here is post-hoc and reported whatever comes out.

Motivation (Saturday's reading, since corrected in FINDINGS Result 3): the gap looked like lower scores rather than
extra mistakes, so these fixes target the score and the cost of a check:

  scorers (what decides "doubtful"; all accent-blind, fitted on TUNE speakers only)
    base      Whisper large-v3 sentence score (avg log-prob), as in the main results
    adapt     the caller-adaptive line: the score minus lam x the mean score of the same caller's OTHER sentences
              (an agent hears several sentences per call; it never needs to know the accent). lam tuned on tune.
    agree     two recognisers: large-v3 (Groq) and large-v3-turbo (local); agreeing transcripts get a bonus D
              to their score (D tuned on tune).
    learned   logistic regression on accent-blind features (both scores, agreement, turbo's weakest word,
              length, no-speech prob), fitted on tune; score = -P(misheard).
  ways to double-check (what the caller pays, in words said again)
    repeat    "Could you repeat?": the whole sentence (as in the main results)
    readback  "I heard X, is that right?": 1 word ("yes") if it was right, 1 + the sentence if it was wrong

Line chosen on tune speakers (all groups pooled) so that at most 5 % of calls slip through with a mistake,
results on test speakers, and cross-fit (halves swapped) as in analyze.py. Same optimistic assumption as before:
what the caller says again is then heard right; with readback, the caller notices a wrong read-back.

Usage: python fixes.py            Output: results/fixes_phone.csv
"""
from __future__ import annotations

import csv
import json
import math

import jiwer
import numpy as np

from analyze import BOOT, GROUP_ORDER, TARGET, bootstrap_gap, by_speaker, load, seeded
from common import RESULTS, normalise

LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]
BONUS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0, 10.0]


def attach_features(rows: list[dict]) -> list[dict]:
    turbo = {}
    for line in open(RESULTS / "local_turbo_phone.jsonl", encoding="utf-8"):
        d = json.loads(line)
        lp = [math.log(max(w["p"], 1e-6)) for w in d["words"]]
        turbo[d["clip_id"]] = (normalise(d["text"]), sum(lp) / len(lp) if lp else -5.0, min(lp, default=-14.0))
    nsp = {json.loads(l)["clip_id"]: json.loads(l)["no_speech_prob"]
           for l in open(RESULTS / "groq_whisper-large-v3_phone.jsonl", encoding="utf-8")}
    spk = by_speaker(rows)
    out = []
    for r in rows:
        others = [c["conf"] for c in spk[r["speaker"]] if c["clip_id"] != r["clip_id"]]
        th, tconf, tmin = turbo[r["clip_id"]]
        o = jiwer.process_words(r["ref"], th or "x")
        out.append(r | {"loo": sum(others) / len(others), "agree": float(th == r["hyp"]), "tconf": tconf, "t_hyp": th,
                        "t_err": o.substitutions + o.deletions + o.insertions,
                        "tmin": tmin, "nwords": len(r["hyp"].split()), "nsp": nsp[r["clip_id"]]})
    return out


# ---------- scorers: fit(tune rows, hyper) -> score(row) ----------
def fit_base(tune, h):
    return lambda r: r["conf"]


def fit_adapt(tune, lam):
    return lambda r: r["conf"] - lam * r["loo"]


def fit_agree(tune, bonus):
    return lambda r: r["conf"] + bonus * r["agree"]


FEATS = ("conf", "tconf", "tmin", "agree", "nwords", "nsp")


def fit_learned(tune, h):
    x = np.array([[r[f] for f in FEATS] for r in tune], dtype=float)
    mu, sd = x.mean(0), x.std(0) + 1e-9
    x = np.c_[np.ones(len(x)), (x - mu) / sd]
    y = np.array([float(r["misheard"]) for r in tune])
    w = np.zeros(x.shape[1])
    for _ in range(3000):  # plain gradient descent with a small L2 penalty: no extra dependency
        p = 1 / (1 + np.exp(-x @ w))
        w -= 0.1 * (x.T @ (p - y) / len(y) + 0.01 * np.r_[0, w[1:]])
    return lambda r: -float(1 / (1 + np.exp(-(w[0] + ((np.array([r[f] for f in FEATS]) - mu) / sd) @ w[1:]))))


SCORERS = {"base": (fit_base, [None]), "adapt": (fit_adapt, LAMS), "agree": (fit_agree, BONUS), "learned": (fit_learned, [None])}


def line_for(rows: list[dict], s: dict) -> float:
    """Lowest line (fewest checks) keeping slipped mistakes <= TARGET on these rows; s = clip -> score."""
    n = len(rows)
    for t in sorted({s[r["clip_id"]] for r in rows}):
        if sum(s[r["clip_id"]] >= t and r["misheard"] for r in rows) / n <= TARGET:
            return t
    return max(s.values()) + 1e-9


def fit(tune: list[dict], name: str):
    """Pick the hyper-parameter that asks the fewest tune callers to check at the 5 % line (accent-blind)."""
    fn, hypers = SCORERS[name]
    best = None
    for h in hypers:
        sc = fn(tune, h)
        s = {r["clip_id"]: sc(r) for r in tune}
        t = line_for(tune, s)
        checks = sum(s[r["clip_id"]] < t for r in tune) / len(tune)
        if best is None or checks < best[0] - 1e-12:
            best = (checks, h, sc, t)
    return best[1], best[2], best[3]


def decide(rows: list[dict], name: str, crossfit: bool) -> list[dict]:
    """Attach check (bool) for scorer `name`, line fitted on the other half."""
    halves = {k: [r for r in rows if r["split"] == k] for k in ("tune", "test")}
    pairs = [("tune", "test")] + ([("test", "tune")] if crossfit else [])
    out = []
    for fit_on, judge in pairs:
        h, sc, t = fit(halves[fit_on], name)
        out += [r | {"check": sc(r) < t, "hyper": h} for r in halves[judge]]
    return out


def cost(r: dict, how: str) -> float:
    if not r["check"]:
        return 0.0
    if how == "repeat":
        return float(r["ref_words"])
    return 1.0 + (r["ref_words"] if r["misheard"] else 0.0)  # readback: "yes", or "no" + the sentence


checks = lambda rs: sum(r["check"] for r in rs) / len(rs)
slip = lambda rs: sum((not r["check"]) and r["misheard"] for r in rs) / len(rs)
false_alarm = lambda rs: (lambda ok: sum(r["check"] for r in ok) / len(ok))([r for r in rs if not r["misheard"]])


def words(how: str):
    return lambda rs: sum(cost(r, how) for r in rs) / len(rs)


def main() -> None:
    rows = attach_features(load("phone"))
    assert all(len(v) == 4 for v in by_speaker(rows).values())
    table = []
    for crossfit in (False, True):
        base_dec = {r["clip_id"]: r for r in decide(rows, "base", crossfit)}
        print(f"\n===== phone, {'cross-fit (all speakers)' if crossfit else 'test half'} =====")
        for name in SCORERS:
            dec = decide(rows, name, crossfit)
            hypers = sorted({str(r['hyper']) for r in dec})
            for how in ("repeat", "readback"):
                fr_rows = [r for r in dec if r["group"] == "france"]
                for g in GROUP_ORDER:
                    gr = [r for r in dec if r["group"] == g]
                    row = {"crossfit": crossfit, "scorer": name, "hyper": "/".join(hypers), "check_how": how, "group": g,
                           "checked": checks(gr), "words": words(how)(gr), "slip": slip(gr), "false_alarm": false_alarm(gr)}
                    if g != "france":
                        key = ("fixes", crossfit, name, how, g)
                        for stat_name, stat in (("checked", checks), ("words", words(how)), ("slip", slip)):
                            lo, hi, p = bootstrap_gap(gr, fr_rows, stat, seeded(*key, stat_name))
                            row |= {f"{stat_name}_gap": stat(gr) - stat(fr_rows), f"{stat_name}_gap_lo": lo,
                                    f"{stat_name}_gap_hi": hi, f"{stat_name}_gap_p": p}
                        # change of the words gap against the main rule (base scorer + repeat), same speakers resampled
                        paired = lambda rs: sum(cost(base_dec[r["clip_id"]], "repeat") - cost(r, how) for r in rs) / len(rs)
                        lo, hi, p = bootstrap_gap(gr, fr_rows, paired, seeded(*key, "gap_change"))
                        row |= {"gap_change": paired(gr) - paired(fr_rows), "gap_change_lo": lo, "gap_change_hi": hi, "gap_change_p": p}
                    table.append(row)
                show = [r for r in table if r["crossfit"] == crossfit and r["scorer"] == name and r["check_how"] == how]
                print(f"\n{name:8s} {how:8s} hyper={'/'.join(hypers)}")
                for r in show:
                    extra = (f"  words gap {r['words_gap']:+.2f} [{r['words_gap_lo']:+.2f},{r['words_gap_hi']:+.2f}] p={r['words_gap_p']:.3f}"
                             f"  checked gap {r['checked_gap']:+.0%} p={r['checked_gap_p']:.3f}"
                             f"  slip gap {r['slip_gap']:+.1%} p={r['slip_gap_p']:.3f}"
                             f"  gap change {r['gap_change']:+.2f} [{r['gap_change_lo']:+.2f},{r['gap_change_hi']:+.2f}] p={r['gap_change_p']:.3f}") if "words_gap" in r else ""
                    print(f"  {r['group']:22s} checked {r['checked']:5.0%}  words {r['words']:5.2f}  slip {r['slip']:5.1%}  FA {r['false_alarm']:4.0%}{extra}")
    keys = list(table[-1])
    with open(RESULTS / "fixes_phone.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in table])
    print("\nsaved results/fixes_phone.csv")
    extra_checks(rows)


def fixed_lines(rows: list[dict], score, per_group: bool) -> list[dict]:
    """Cross-fit decisions with a given score: one pooled line, or (label-aware, NOT usable by an accent-blind
    agent) one line per accent group, each fitted on that group's other half."""
    out = []
    for fit_on, judge in (("tune", "test"), ("test", "tune")):
        for g in (GROUP_ORDER if per_group else [None]):
            fr = [r for r in rows if r["split"] == fit_on and (g is None or r["group"] == g)]
            t = line_for(fr, {r["clip_id"]: score(r) for r in fr})
            out += [r | {"check": score(r) < t} for r in rows if r["split"] == judge and (g is None or r["group"] == g)]
    return out


def extra_checks(rows: list[dict]) -> None:
    print("\n===== caller-adaptive line with lam forced (cross-fit; no ranges) =====")
    for lam in (0.0, 0.5, 1.0):
        dec = fixed_lines(rows, lambda r, l=lam: r["conf"] - l * r["loo"], False)
        print(f"lam={lam}: " + "  ".join(f"{g[:7]} checked {checks([r for r in dec if r['group'] == g]):.0%}" for g in GROUP_ORDER))
    print("\n===== one pooled line vs one line per accent (label-aware reference; cross-fit; ranges keep each line fixed, so too narrow) =====")
    for name, per in (("one line", False), ("line per accent", True)):
        dec = fixed_lines(rows, lambda r: r["conf"], per)
        fr = [r for r in dec if r["group"] == "france"]
        print(f"{name}: " + "  ".join(f"{g[:7]} checked {checks([r for r in dec if r['group'] == g]):.0%} slip {slip([r for r in dec if r['group'] == g]):.1%}" for g in GROUP_ORDER))
        for g in GROUP_ORDER[1:]:
            gr = [r for r in dec if r["group"] == g]
            a = bootstrap_gap(gr, fr, checks, seeded("perline", name, g, "checked"))
            b = bootstrap_gap(gr, fr, slip, seeded("perline", name, g, "slip"))
            print(f"   {g:24s} checked gap {checks(gr) - checks(fr):+.1%} [{a[0]:+.0%},{a[1]:+.0%}] p={a[2]:.3f}"
                  f"   slip gap {slip(gr) - slip(fr):+.1%} [{b[0]:+.1%},{b[1]:+.1%}] p={b[2]:.3f}")
    print("\n===== choosing between the two recognisers' transcripts (test half; no ranges) =====")
    tune = [r for r in rows if r["split"] == "tune"]
    test = [r for r in rows if r["split"] == "test"]
    picked = lambda r, d: r["t_err"] if (r["t_hyp"] != r["hyp"] and r["tconf"] - r["conf"] > d) else r["errors"]
    w = lambda rs, d: sum(picked(r, d) for r in rs) / sum(r["ref_words"] for r in rs)
    d = min((x / 20 for x in range(-20, 41)), key=lambda x: w(tune, x))  # one accent-blind setting, fitted on tune
    for g in GROUP_ORDER:
        gr = [r for r in test if r["group"] == g]
        print(f"   {g:24s} large-v3 WER {w(gr, math.inf):.1%}  ->  chosen transcript {w(gr, d):.1%}   (d = {d})")


def _self_check() -> None:
    a = {"check": True, "ref_words": 8, "misheard": False}
    assert cost(a, "repeat") == 8 and cost(a, "readback") == 1
    assert cost(a | {"misheard": True}, "readback") == 9 and cost(a | {"check": False}, "readback") == 0
    rows = [{"clip_id": str(i), "misheard": i < 2} for i in range(10)]
    s = {str(i): float(i) for i in range(10)}          # the two misheard clips score lowest
    assert line_for(rows, s) == 2.0                    # checking clip 0 alone still lets 1 in 10 slip: both must be checked


if __name__ == "__main__":
    _self_check()
    main()
