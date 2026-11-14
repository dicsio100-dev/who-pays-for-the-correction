"""Build the human check page: 20 clips (5 per group, seed 14) to listen to and judge by ear.

For each clip the checker answers two questions:
  1. Does the reference sentence match what is said?          (catches bad reference text)
  2. Is each highlighted difference a real hearing error?      (catches normalisation artefacts)
Answers are kept in the browser and downloaded as verification.json.

Usage: python make_check_page.py   ->  <VERITAS_DATA>/verification/index.html
"""
from __future__ import annotations

import csv
import html
import json
import random

import jiwer

from analyze import GROUP_ORDER, load
from common import DATA, MANIFEST

PER_GROUP = 5
LABELS = {"france": "France", "canada": "Canada", "afrique_subsaharienne": "Afrique subsaharienne",
          "cameroun_yaounde": "Cameroun (Yaoundé)"}


def diff_html(ref: str, hyp: str) -> tuple[str, str]:
    """Reference and Whisper text, normalised, with the differing words marked."""
    r, h = ref.split(), hyp.split()
    out_r, out_h = [], []
    for c in jiwer.process_words(ref, hyp).alignments[0]:
        rs = " ".join(r[c.ref_start_idx:c.ref_end_idx])
        hs = " ".join(h[c.hyp_start_idx:c.hyp_end_idx])
        if c.type == "equal":
            out_r.append(html.escape(rs))
            out_h.append(html.escape(hs))
        else:
            if rs:
                out_r.append(f"<mark>{html.escape(rs)}</mark>")
            if hs:
                out_h.append(f"<mark>{html.escape(hs)}</mark>")
            if c.type == "delete":
                out_h.append("<mark class='gap'>(rien)</mark>")
    return " ".join(out_r), " ".join(out_h)


def pick() -> list[dict]:
    manifest = {r["clip_id"]: r for r in csv.DictReader(open(MANIFEST, encoding="utf-8", newline=""))}
    rows = load("phone")
    rng = random.Random(14)
    chosen = []
    for g in GROUP_ORDER:
        gr = [r for r in rows if r["group"] == g]
        mis = [r for r in gr if r["misheard"]]
        ok = [r for r in gr if not r["misheard"]]
        sample = rng.sample(mis, 3) + rng.sample(ok, 2)  # mostly errors: those are what we must verify
        rng.shuffle(sample)
        for r in sample:
            m = manifest[r["clip_id"]]
            ref_h, hyp_h = diff_html(r["ref"], r["hyp"])
            chosen.append({"id": r["clip_id"], "group": g, "label": LABELS[g], "sentence": m["reference"],
                           "clean": f"../audio/{g}/clean/{r['clip_id']}.wav",
                           "phone": f"../audio/{g}/phone/{r['clip_id']}.wav",
                           "ref_h": ref_h, "hyp_h": hyp_h, "errors": r["errors"]})
    return chosen


PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vérification à l'oreille</title>
<style>
:root { --bg:#f6f5f2; --card:#fff; --ink:#15181d; --muted:#5d6470; --line:#e3e1dc; --mark:#ffe1c2; --accent:#d9530f; --ok:#1f7a4d; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui, sans-serif; }
main { max-width:860px; margin:0 auto; padding:32px 16px 96px; }
h1 { font-size:28px; margin:0 0 8px; }
.intro { color:var(--muted); margin:0 0 24px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:20px; margin:0 0 16px; }
.top { display:flex; justify-content:space-between; gap:12px; align-items:baseline; }
.num { font-weight:700; } .grp { color:var(--muted); font-size:14px; }
.sentence { font-size:18px; margin:10px 0 14px; }
.audio { display:flex; gap:16px; flex-wrap:wrap; margin:0 0 14px; }
.audio label { font-size:13px; color:var(--muted); display:block; }
.cmp { display:grid; grid-template-columns:110px 1fr; gap:6px 12px; font-family:ui-monospace, monospace; font-size:14px; margin:0 0 16px; }
.cmp b { font-family:system-ui, sans-serif; font-weight:600; color:var(--muted); }
mark { background:var(--mark); padding:0 2px; border-radius:3px; } mark.gap { color:var(--accent); }
.q { margin:10px 0 0; } .q p { margin:0 0 6px; font-weight:600; }
.q button { font:inherit; padding:6px 14px; margin:0 6px 6px 0; border:1px solid var(--line); background:#fff; border-radius:8px; cursor:pointer; }
.q button.on { background:var(--ink); color:#fff; border-color:var(--ink); }
textarea { width:100%; min-height:44px; font:inherit; padding:8px; border:1px solid var(--line); border-radius:8px; margin-top:8px; }
.bar { position:fixed; left:0; right:0; bottom:0; background:var(--ink); color:#fff; padding:12px 16px; display:flex; justify-content:center; gap:16px; align-items:center; }
.bar button { font:inherit; padding:8px 18px; border:0; border-radius:8px; background:var(--accent); color:#fff; cursor:pointer; }
</style></head><body><main>
<h1>Vérification à l'oreille : 20 exemples</h1>
<p class="intro">Pour chaque exemple : écoute (le son normal, puis la version téléphone), lis la phrase attendue, puis réponds aux deux questions.
Les mots surlignés sont ceux que le calcul compte comme des erreurs. Dis si ce sont de vraies erreurs d'écoute, ou seulement une autre façon
d'écrire la même chose (par exemple « M. » pour « monsieur »). Tes réponses restent dans ce navigateur. À la fin, clique sur « Télécharger mes réponses ».</p>
<div id="list"></div>
</main>
<div class="bar"><span id="count"></span><button id="dl">Télécharger mes réponses</button></div>
<script>
const CLIPS = __CLIPS__;
const KEY = "veritas-check-v1";
let answers = {};
try { answers = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { answers = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(answers)); } catch (e) {} count(); }
function count() {
  const done = CLIPS.filter(c => answers[c.id] && answers[c.id].ref && (c.errors === 0 || answers[c.id].err)).length;
  document.getElementById("count").textContent = done + " / " + CLIPS.length + " exemples faits";
}
function q(c, field, text, opts) {
  const a = (answers[c.id] || {})[field];
  return '<div class="q"><p>' + text + '</p>' + opts.map(o =>
    '<button data-id="' + c.id + '" data-f="' + field + '" data-v="' + o + '"' + (a === o ? ' class="on"' : '') + '>' + o + '</button>').join("") + '</div>';
}
document.getElementById("list").innerHTML = CLIPS.map((c, i) =>
  '<section class="card"><div class="top"><span class="num">Exemple ' + (i + 1) + '</span><span class="grp">' + c.label + '</span></div>' +
  '<p class="sentence">« ' + c.sentence + ' »</p>' +
  '<div class="audio"><div><label>Son normal</label><audio controls preload="none" src="' + c.clean + '"></audio></div>' +
  '<div><label>Téléphone</label><audio controls preload="none" src="' + c.phone + '"></audio></div></div>' +
  '<div class="cmp"><b>Attendu</b><span>' + c.ref_h + '</span><b>Whisper</b><span>' + c.hyp_h + '</span></div>' +
  q(c, "ref", "1. La phrase attendue correspond-elle à ce que la personne dit ?", ["Oui", "Non", "Je ne sais pas"]) +
  (c.errors ? q(c, "err", "2. Les mots surlignés sont-ils de vraies erreurs de Whisper ?", ["Toutes", "Certaines", "Aucune"]) :
              '<div class="q"><p>2. Whisper n\\'a fait aucune erreur ici : rien à juger.</p></div>') +
  '<textarea data-id="' + c.id + '" placeholder="Remarque (facultatif)">' + (((answers[c.id] || {}).note) || "") + '</textarea></section>'
).join("");
document.addEventListener("click", e => {
  const b = e.target.closest(".q button"); if (!b) return;
  answers[b.dataset.id] = Object.assign(answers[b.dataset.id] || {}, { [b.dataset.f]: b.dataset.v });
  b.parentElement.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
  save();
});
document.addEventListener("input", e => {
  if (e.target.tagName !== "TEXTAREA") return;
  answers[e.target.dataset.id] = Object.assign(answers[e.target.dataset.id] || {}, { note: e.target.value }); save();
});
document.getElementById("dl").onclick = () => {
  const blob = new Blob([JSON.stringify(CLIPS.map(c => ({ id: c.id, group: c.group, errors: c.errors, ...(answers[c.id] || {}) })), null, 2)], { type: "application/json" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "verification.json"; a.click();
};
count();
</script></body></html>
"""


def main() -> None:
    clips = pick()
    out = DATA / "verification" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.replace("__CLIPS__", json.dumps(clips, ensure_ascii=False)), encoding="utf-8")
    missing = [c for c in clips for k in ("clean", "phone") if not (out.parent / c[k]).resolve().exists()]
    assert not missing, f"audio not found: {missing[:2]}"
    print(f"{len(clips)} clips -> {out}")


if __name__ == "__main__":
    assert diff_html("a b c", "a x c")[1] == "a <mark>x</mark> c"
    main()
