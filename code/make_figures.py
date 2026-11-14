"""Static SVG figures for the README (GitHub cannot run the interactive page), drawn from site/data.json.

  figures/wer.svg      word error rate per accent, normal audio vs phone line, with 95 % ranges
  figures/repeats.svg  share of test callers asked to repeat at every setting of the rule (phone line)
Usage: python make_figures.py
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"france": "#2a78d6", "canada": "#1baf7a", "afrique_subsaharienne": "#eb6834", "cameroun_yaounde": "#6250d6"}
INK, INK2, GRID, BG = "#10161f", "#4a5361", "#e3e7ec", "#ffffff"
FONT = 'font-family="system-ui, -apple-system, &apos;Segoe UI&apos;, Helvetica, Arial, sans-serif"'


def text(x: float, y: float, s: str, size: int = 14, color: str = INK, anchor: str = "start", weight: int = 400) -> str:
    return (f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{color}' text-anchor='{anchor}' "
            f"font-weight='{weight}' {FONT}>{escape(s)}</text>")


def wer_figure(d: dict) -> str:
    groups, L = d["order"], d["labels"]
    clean, phone = d["stats"]["clean_crossfit"], d["stats"]["phone_crossfit"]
    W, lab, right, top, row, vmax = 760, 190, 70, 56, 64, 0.25
    H = top + row * len(groups) + 40
    sx = lambda v: lab + (W - right - lab) * v / vmax
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}'>",
           f"<rect width='{W}' height='{H}' fill='{BG}'/>",
           text(0, 22, "Whisper large-v3: share of words wrong, all speakers", 17, INK, weight=700),
           text(0, 42, "Light bar = normal audio, dark bar = phone line. Thin line = 95 % range.", 13, INK2)]
    for i in range(6):
        v = vmax * i / 5
        out += [f"<line x1='{sx(v):.1f}' x2='{sx(v):.1f}' y1='{top}' y2='{H - 34}' stroke='{GRID}'/>",
                text(sx(v), H - 16, f"{100 * v:.0f} %", 12, INK2, "middle")]
    for i, g in enumerate(groups):
        y = top + i * row
        out.append(text(0, y + 30, L[g], 15))
        for k, (s, op) in enumerate(((clean[g], 0.45), (phone[g], 1.0))):
            by = y + 10 + k * 22
            out += [f"<rect x='{lab}' y='{by}' width='{sx(s['wer']) - lab:.1f}' height='16' rx='3' fill='{COLORS[g]}' opacity='{op}'/>",
                    f"<line x1='{sx(s['wer_lo']):.1f}' x2='{sx(s['wer_hi']):.1f}' y1='{by + 8}' y2='{by + 8}' stroke='{INK}' stroke-opacity='.6'/>",
                    text(W, by + 13, f"{100 * s['wer']:.1f} %", 13, INK2, "end")]
    return "\n".join(out + ["</svg>"])


def repeats_figure(d: dict) -> str:
    groups, L = d["order"], d["labels"]
    steps = d["slider"]["phone"]
    W, H, left, right, top, bottom = 760, 420, 56, 170, 60, 50
    sx = lambda v: left + (W - right - left) * (v - 0.01) / (0.30 - 0.01)
    sy = lambda v: top + (H - top - bottom) * (1 - v)
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}'>",
           f"<rect width='{W}' height='{H}' fill='{BG}'/>",
           text(0, 22, "Same rule, different bill: callers asked to repeat (phone line, test speakers)", 17, INK, weight=700),
           text(0, 42, "The rule's line is tuned on the other half of the speakers for each setting.", 13, INK2)]
    for i in range(6):
        v = i / 5
        out += [f"<line x1='{left}' x2='{W - right}' y1='{sy(v):.1f}' y2='{sy(v):.1f}' stroke='{GRID}'/>",
                text(left - 8, sy(v) + 4, f"{100 * v:.0f} %", 12, INK2, "end")]
    for v in (0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        out.append(text(sx(v), H - bottom + 20, f"{100 * v:.0f} %", 12, INK2, "middle"))
    out.append(text((left + W - right) / 2, H - 8, "Calls allowed to go through with a mistake", 13, INK2, "middle"))
    ends = []
    for g in groups:
        pts = " ".join(f"{sx(s['target']):.1f},{sy(s['groups'][g]['repeat']):.1f}" for s in steps)
        out.append(f"<polyline points='{pts}' fill='none' stroke='{COLORS[g]}' stroke-width='2' stroke-linejoin='round'/>")
        ends.append((sy(steps[-1]["groups"][g]["repeat"]), g))
    last_y = -99.0
    for y, g in sorted(ends):  # direct labels at the right end, nudged apart so they never overlap
        y = max(y, last_y + 17)
        last_y = y
        out.append(text(W - right + 8, y + 4, L[g], 13, INK))
    return "\n".join(out + ["</svg>"])


def main() -> None:
    d = json.loads((ROOT / "site" / "data.json").read_text(encoding="utf-8"))
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    (out / "wer.svg").write_text(wer_figure(d), encoding="utf-8")
    (out / "repeats.svg").write_text(repeats_figure(d), encoding="utf-8")
    print(f"wrote {out / 'wer.svg'} and {out / 'repeats.svg'}")


if __name__ == "__main__":
    main()
