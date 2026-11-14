"""Build the study manifest: which clips, which speakers, which accent group,
which half (tune / test), plus a phone-quality copy of every clip.

Usage:
  python prepare.py yaounde                 # SLR57 Yaoundé read speech
  python prepare.py cv path/to/validated.tsv path/to/clips_dir

Rules (fixed before looking at any result):
  - at most CLIPS_PER_SPEAKER clips per speaker, so a few very active
    volunteers cannot decide the result on their own;
  - speakers, not clips, are split 50/50 into "tune" (where we choose the
    confirmation threshold) and "test" (where we report results).
"""
from __future__ import annotations

import csv
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from common import AUDIO, MANIFEST, RAW

SEED = 14
CLIPS_PER_SPEAKER = 4
MAX_SPEAKERS = {"france": 50, "canada": 50, "afrique_subsaharienne": 50, "cameroun_yaounde": 30}
FIELDS = ["clip_id", "group", "source", "speaker", "split", "reference", "duration", "clean", "phone"]

# Phone line: keep 300-3400 Hz, 8 kHz sampling, G.711 mu-law codec (what a landline carries),
# then back to 16 kHz PCM so every condition reaches Whisper in the same file format.
PHONE_FILTER = "highpass=f=300,lowpass=f=3400"


def run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {res.stderr[-400:]}")


def to_clean_wav(src: Path, dst: Path) -> None:
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)])


def to_phone_wav(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(".ulaw.wav")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-af", PHONE_FILTER,
         "-ar", "8000", "-c:a", "pcm_mulaw", str(tmp)])
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-ar", "16000", "-c:a", "pcm_s16le", str(dst)])
    tmp.unlink()


def select(by_speaker: dict[str, list[dict]], group: str, rng: random.Random) -> list[dict]:
    """Pick speakers with enough clips, cap clips per speaker, split speakers tune/test."""
    eligible = sorted(s for s, clips in by_speaker.items() if len(clips) >= CLIPS_PER_SPEAKER)
    rng.shuffle(eligible)
    chosen = eligible[: MAX_SPEAKERS[group]]
    rows = []
    for i, spk in enumerate(sorted(chosen)):
        split = "tune" if i % 2 == 0 else "test"
        for clip in rng.sample(by_speaker[spk], CLIPS_PER_SPEAKER):
            rows.append({**clip, "group": group, "speaker": spk, "split": split})
    return rows


def materialise(rows: list[dict]) -> list[dict]:
    """Write clean and phone copies; fill duration and file paths."""
    out = []
    for r in rows:
        group_dir = AUDIO / r["group"]
        (group_dir / "clean").mkdir(parents=True, exist_ok=True)
        (group_dir / "phone").mkdir(parents=True, exist_ok=True)
        clean = group_dir / "clean" / f"{r['clip_id']}.wav"
        phone = group_dir / "phone" / f"{r['clip_id']}.wav"
        if not clean.exists():
            to_clean_wav(Path(r["src"]), clean)
        if not phone.exists():
            to_phone_wav(clean, phone)
        out.append({k: r.get(k, "") for k in FIELDS} | {
            "duration": f"{sf.info(clean).duration:.2f}", "clean": str(clean), "phone": str(phone)})
    return out


def from_yaounde() -> list[dict]:
    base = RAW / "African_Accented_French"
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    with open(base / "transcripts/train/yaounde/fn_text.txt", encoding="utf-8") as f:
        for line in f:
            path, _, text = line.strip().partition(" ")
            spk, name = Path(path).parent.name, Path(path).name
            wav = base / "speech/train/yaounde/read" / spk / name
            if wav.exists() and text.strip():
                by_speaker[spk].append({"clip_id": Path(name).stem, "source": "openslr57_yaounde",
                                        "reference": text.strip(), "src": str(wav)})
    return select(by_speaker, "cameroun_yaounde", random.Random(SEED))


# validated.tsv stores the variant's display name, not its code (fr-metro, fr-namerica, fr-safrica).
CV_GROUPS = {"Français de métropole": "france",
             "Français d'Amérique du Nord": "canada",
             "Français d'Afrique subsaharienne et des îles africaines": "afrique_subsaharienne"}


def from_common_voice(tsv: Path, clips_dir: Path) -> list[dict]:
    """Common Voice validated clips; accent group from the speaker's self-declared variant."""
    per_group: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with open(tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            group = CV_GROUPS.get((row.get("variant") or "").strip())
            if not group or not row.get("sentence"):
                continue
            per_group[group][row["client_id"]].append({
                "clip_id": Path(row["path"]).stem, "source": "commonvoice27_fr",
                "reference": row["sentence"].strip(), "src": str(clips_dir / row["path"])})
    rows: list[dict] = []
    for group in CV_GROUPS.values():
        rows += select(per_group[group], group, random.Random(SEED))
    return rows


def write_manifest(new_rows: list[dict]) -> None:
    old = []
    if MANIFEST.exists():
        with open(MANIFEST, encoding="utf-8", newline="") as f:
            old = [r for r in csv.DictReader(f) if r["source"] not in {n["source"] for n in new_rows}]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(old + new_rows)


def summary(rows: list[dict]) -> None:
    stats: dict[str, dict[str, set | int]] = defaultdict(lambda: {"speakers": set(), "clips": 0})
    for r in rows:
        stats[r["group"]]["speakers"].add(r["speaker"])
        stats[r["group"]]["clips"] += 1
    for g, s in stats.items():
        print(f"{g:24s} {len(s['speakers']):3d} speakers  {s['clips']:4d} clips")


if __name__ == "__main__":
    if sys.argv[1:2] == ["yaounde"]:
        rows = materialise(from_yaounde())
    elif sys.argv[1:2] == ["cv"]:
        rows = materialise(from_common_voice(Path(sys.argv[2]), Path(sys.argv[3])))
    else:
        sys.exit(__doc__)
    assert all(Path(r["phone"]).exists() for r in rows), "missing phone copy"
    assert len({r["speaker"] for r in rows if r["split"] == "tune"} &
               {r["speaker"] for r in rows if r["split"] == "test"}) == 0, "speaker in both halves"
    write_manifest(rows)
    summary(rows)
