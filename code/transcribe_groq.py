"""Send clips to Whisper large-v3 on Groq's free tier and cache every answer.

Usage:
  python transcribe_groq.py CONDITION [--group G] [--limit N] [--prompt-file F] [--auto-language]
  CONDITION = clean | phone

Each answer is appended to results/groq_<model>_<condition>[_<tag>].jsonl as soon
as it arrives, so a crash or a rate limit never loses work and nothing is sent twice.
Confidence: Groq returns, for each segment, avg_logprob (the mean log-probability
of its tokens). We keep the duration-weighted mean over segments as the clip's
confidence; exp(avg_logprob) reads as "how sure, on average, per token".
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import requests

from common import MANIFEST, RESULTS

URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MODEL = "whisper-large-v3"
MIN_GAP_S = 3.1  # free tier allows 20 requests per minute


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["clip_id"] for line in f if line.strip()}


def transcribe(session: requests.Session, wav: str, prompt: str | None, language: str | None = "fr") -> dict:
    data = {"model": MODEL, "response_format": "verbose_json", "temperature": "0"}
    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt
    for attempt in range(8):
        with open(wav, "rb") as f:
            r = session.post(URL, data=data, files={"file": (Path(wav).name, f, "audio/wav")}, timeout=120)
        if r.status_code == 200:
            left = r.headers.get("x-ratelimit-remaining-requests")
            if left is not None and int(left) % 100 == 0:
                print(f"  requests left today: {left}")
            return r.json()
        if r.status_code in (429, 500, 502, 503):
            wait = float(r.headers.get("retry-after", 0) or 0) or min(60, 5 * 2 ** attempt)
            print(f"  {r.status_code}, waiting {wait:.0f} s")
            time.sleep(wait)
            continue
        raise RuntimeError(f"Groq {r.status_code}: {r.text[:300]}")
    raise RuntimeError("Groq kept refusing after 8 attempts")


def clip_confidence(resp: dict) -> tuple[float, float]:
    segs = resp.get("segments") or []
    if not segs:
        return float("nan"), float("nan")
    total = sum(max(s["end"] - s["start"], 0.01) for s in segs)
    mean_lp = sum(s["avg_logprob"] * max(s["end"] - s["start"], 0.01) for s in segs) / total
    return mean_lp, max(s["no_speech_prob"] for s in segs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("condition", choices=["clean", "phone"])
    ap.add_argument("--group")
    ap.add_argument("--split", choices=["tune", "test"])
    ap.add_argument("--source", help="only clips from this source, e.g. commonvoice27_fr")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--prompt-file", help="text file whose content is passed to Whisper as context")
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. 'prompt'")
    ap.add_argument("--auto-language", action="store_true",
                    help="do not tell Whisper the language (baseline for the 'say it is French' fix)")
    args = ap.parse_args()

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY missing: create a free key on console.groq.com and set it in your environment")
    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip() if args.prompt_file else None

    with open(MANIFEST, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (not args.group or r["group"] == args.group)
                and (not args.split or r["split"] == args.split) and (not args.source or r["source"] == args.source)]
    out = RESULTS / f"groq_{MODEL}_{args.condition}{'_' + args.tag if args.tag else ''}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    todo = [r for r in rows if r["clip_id"] not in done][: args.limit]
    print(f"{len(done)} already done, {len(todo)} to send -> {out.name}")

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {key}", "User-Agent": "Mozilla/5.0"})
    last = 0.0
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            time.sleep(max(0.0, MIN_GAP_S - (time.time() - last)))
            last = time.time()
            resp = transcribe(session, r[args.condition], prompt, None if args.auto_language else "fr")
            lp, nsp = clip_confidence(resp)
            f.write(json.dumps({"clip_id": r["clip_id"], "group": r["group"], "condition": args.condition,
                                "tag": args.tag, "model": MODEL, "text": resp.get("text", "").strip(),
                                "avg_logprob": lp, "no_speech_prob": nsp,
                                "language": resp.get("language", "fr" if not args.auto_language else "")}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")


if __name__ == "__main__":
    main()
