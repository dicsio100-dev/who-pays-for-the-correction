"""Whisper on this computer (faster-whisper, CPU, int8) with a confidence for every word.

Groq gives one confidence per sentence; this gives one per word, which the
"confirm only the doubtful part" policy needs.
Usage: python transcribe_local.py CONDITION [--model small|turbo] [--beam 5] [--group G] [--limit N]
Output: results/local_<model>_<condition>.jsonl, appended clip by clip (safe to stop and resume).
"""
from __future__ import annotations

import argparse
import csv
import json

from faster_whisper import WhisperModel

from common import DATA, MANIFEST, RESULTS

MODELS = {"small": "small", "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("condition", choices=["clean", "phone"])
    ap.add_argument("--model", default="small", choices=list(MODELS))
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--group")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    with open(MANIFEST, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if not args.group or r["group"] == args.group]
    rows.sort(key=lambda r: r["source"] != "commonvoice27_fr")  # main comparison first, Yaoundé complement last
    out = RESULTS / f"local_{args.model}_{args.condition}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["clip_id"] for l in open(out, encoding="utf-8") if l.strip()}
    todo = [r for r in rows if r["clip_id"] not in done][: args.limit]
    print(f"{len(done)} already done, {len(todo)} to transcribe -> {out.name}")

    model = WhisperModel(MODELS[args.model], device="cpu", compute_type="int8", cpu_threads=args.threads,
                         download_root=str(DATA / "models"))
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            segs, _ = model.transcribe(r[args.condition], language="fr", word_timestamps=True,
                                       beam_size=args.beam, temperature=0.0, condition_on_previous_text=False)
            words = [{"w": w.word.strip(), "p": round(w.probability, 4)} for s in segs for w in s.words]
            f.write(json.dumps({"clip_id": r["clip_id"], "group": r["group"], "condition": args.condition,
                                "model": args.model, "beam": args.beam, "text": " ".join(w["w"] for w in words),
                                "words": words}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")


if __name__ == "__main__":
    main()
