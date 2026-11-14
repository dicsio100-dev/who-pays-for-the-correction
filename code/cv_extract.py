"""Pull only the clips we need out of Common Voice 27.0 French (28.5 GB) without
saving the archive: the .tar.gz is read as a stream, twice.

  pass 1: keep validated.tsv (who said what, with which variant / accent)
  pass 2: keep only the mp3 files of the speakers selected by prepare.py

Needs MDC_API_KEY (Mozilla Data Collective) in the environment, and the dataset
terms accepted on the website by the account that owns the key.
Usage: python cv_extract.py
"""
from __future__ import annotations

import os
import tarfile
from pathlib import Path

import requests

import prepare
from common import RAW

DATASET_ID = "cmu5kub4600pkmh078n7n5ogl"  # Common Voice Scripted Speech 27.0 - French
API = "https://mozilladatacollective.com/api"
OUT = RAW / "cv27_fr"


def presigned_url() -> str:
    key = os.environ.get("MDC_API_KEY")
    if not key:
        raise SystemExit("MDC_API_KEY missing: create a key on Mozilla Data Collective and set it in your environment")
    r = requests.post(f"{API}/datasets/{DATASET_ID}/download",
                      headers={"Authorization": f"Bearer {key}"}, timeout=60)
    if r.status_code != 200:
        raise SystemExit(f"Mozilla refused the download ({r.status_code}): {r.text[:300]}")
    info = r.json()
    print(f"archive {info.get('filename')}  {int(info.get('sizeBytes', 0)) / 1e9:.1f} GB")
    return info["downloadUrl"]


def stream_members(url: str, wanted, stop_when_done: bool):
    """Yield (member, fileobj) for archive members accepted by wanted(name)."""
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            for m in tar:
                if m.isfile() and wanted(m.name):
                    yield m, tar.extractfile(m)
                    if stop_when_done and wanted.done():
                        return


class Want:
    def __init__(self, names: set[str]):
        self.left = set(names)

    def __call__(self, member_name: str) -> bool:
        return Path(member_name).name in self.left

    def take(self, member_name: str) -> None:
        self.left.discard(Path(member_name).name)

    def done(self) -> bool:
        return not self.left


def save(fileobj, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        while chunk := fileobj.read(1 << 20):
            f.write(chunk)


def main() -> None:
    url = presigned_url()
    tsv = OUT / "validated.tsv"
    if not tsv.exists():
        print("pass 1: looking for validated.tsv")
        want = Want({"validated.tsv"})
        for m, fo in stream_members(url, want, stop_when_done=True):
            save(fo, tsv)
            want.take(m.name)
        assert tsv.exists(), "validated.tsv not found in the archive"

    clips_dir = OUT / "clips"
    rows = prepare.from_common_voice(tsv, clips_dir)
    need = {Path(r["src"]).name for r in rows if not Path(r["src"]).exists()}
    print(f"{len(rows)} clips selected, {len(need)} still to extract")
    if need:
        print("pass 2: extracting selected clips")
        want = Want(need)
        for m, fo in stream_members(presigned_url(), want, stop_when_done=True):
            save(fo, clips_dir / Path(m.name).name)
            want.take(m.name)
        assert want.done(), f"{len(want.left)} clips not found in the archive"

    done = prepare.materialise(rows)
    prepare.write_manifest(done)
    prepare.summary(done)


if __name__ == "__main__":
    main()
