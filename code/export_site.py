"""Export the numbers the explanation page shows into docs/data.json (aggregates only, no audio,
no speaker IDs: Common Voice terms forbid re-sharing the dataset and identifying speakers).

Every number on the page comes from this file, so page and analysis cannot drift apart.
Usage: python export_site.py
"""
from __future__ import annotations

import csv
import json
import math
from datetime import date
from pathlib import Path

import checks
from analyze import GROUP_ORDER, choose_threshold, load, phone_widening, policy
from common import RESULTS

SITE = Path(__file__).resolve().parent.parent / "docs"
LABELS = {"france": "France", "canada": "Canada", "afrique_subsaharienne": "Sub-Saharan Africa",
          "cameroun_yaounde": "Cameroon (Yaoundé)"}
TARGETS = [round(x / 100, 2) for x in range(1, 31)]
REPO = "https://github.com/dicsio100-dev/who-pays-for-the-correction"


def slider(condition: str) -> list[dict]:
    """For each 'errors allowed to slip by' target: threshold from tune speakers, outcome per group on test."""
    rows = load(condition)
    tune = [r for r in rows if r["split"] == "tune"]
    test = [r for r in rows if r["split"] == "test"]
    out = []
    for target in TARGETS:
        t = choose_threshold(tune, target)
        per = {}
        for g in GROUP_ORDER:
            rep, slip = policy([r for r in test if r["group"] == g], t)
            per[g] = {"repeat": round(rep, 4), "slip": round(slip, 4)}
        never = {g: round(sum(r["misheard"] for r in test if r["group"] == g) / sum(r["group"] == g for r in test), 4)
                 for g in GROUP_ORDER}
        rep_all, slip_all = policy(test, t)
        mis_all = sum(r["misheard"] for r in test) / len(test)
        out.append({"target": target, "threshold": round(t, 4), "groups": per, "baseline_never": never,
                    # every test caller pooled: the rule vs asking the same share of callers at random
                    "pooled": {"repeat": round(rep_all, 4), "slip": round(slip_all, 4),
                               "random_slip": round(mis_all * (1 - rep_all), 4), "never_slip": round(mis_all, 4)}})
    return out


def no_nan(x):
    """JSON has no NaN (browsers reject it): France's gap with itself becomes null. Every float is rounded
    once here (3 decimals = 0.1 percentage point), so the page, the figures and the write-up show the same digits."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, float):
        return round(x, 3)
    if isinstance(x, dict):
        return {k: no_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [no_nan(v) for v in x]
    return x


def summary_rows() -> list[dict]:
    with open(RESULTS / "summary.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    table = summary_rows()
    stats = {}
    for r in table:
        key = f"{r['condition']}_{'crossfit' if r['crossfit'] == 'True' else 'test'}"
        stats.setdefault(key, {})[r["group"]] = {k: (float(v) if k not in ("condition", "crossfit", "group") else v)
                                                 for k, v in r.items() if k not in ("condition", "crossfit", "group")}
    data = {"generated": date.today().isoformat(), "repo": REPO, "labels": LABELS, "order": GROUP_ORDER,
            "stats": stats, "slider": {c: slider(c) for c in ("phone", "clean")},
            "robustness": {c: checks.main(c) for c in ("clean", "phone")},
            # phone minus clean WER from the unrounded values, so the page's subtraction adds up exactly
            "phone_increase": {g: stats["phone_crossfit"][g]["wer"] - stats["clean_crossfit"][g]["wer"] for g in GROUP_ORDER},
            "phone_widening": {g: dict(zip(("gap", "lo", "hi", "p"), phone_widening(g)))
                               for g in GROUP_ORDER if g != "france"}}
    check = RESULTS / "human_check.json"  # 20 clips judged by ear by a native French speaker (make_check_page.py)
    if check.exists():
        answers = json.loads(check.read_text(encoding="utf-8"))
        with_errors = [a for a in answers if a["errors"]]
        data["human_check"] = {"clips": len(answers), "ref_ok": sum(a.get("ref") == "Oui" for a in answers),
                               "with_errors": len(with_errors),
                               "all_real": sum(a.get("err") == "Toutes" for a in with_errors),
                               "some_real": sum(a.get("err") == "Certaines" for a in with_errors),
                               "none_real": sum(a.get("err") == "Aucune" for a in with_errors)}
    with open(RESULTS / "language_hint.csv", encoding="utf-8", newline="") as f:
        data["language_hint"] = {r["group"]: {k: (v if k in ("group", "languages") else (float(v) if v else None))
                                              for k, v in r.items()} for r in csv.DictReader(f)}
    num = lambda r, text: {k: (v if k in text else (float(v) if v else None)) for k, v in r.items()}
    for key, name, text in (("targeted", "policy_phone_turbo.csv", ("condition", "policy", "group")),
                            ("targeted_gaps", "policy_gaps_phone_turbo.csv", ("group",))):
        if (RESULTS / name).exists():  # fix 2 needs the local turbo run (policy_words.py phone turbo)
            with open(RESULTS / name, encoding="utf-8", newline="") as f:
                data[key] = [num(r, text) for r in csv.DictReader(f)]
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "data.json").write_text(json.dumps(no_nan(data), ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    five = next(s for s in data["slider"]["phone"] if s["target"] == 0.05)
    assert abs(five["groups"]["france"]["repeat"] - stats["phone_test"]["france"]["repeat"]) < 1e-3, "page and analysis disagree"
    print(f"wrote {SITE / 'data.json'}: {len(TARGETS)} slider steps x 2 conditions")


if __name__ == "__main__":
    main()
