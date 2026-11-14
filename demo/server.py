"""Local demo: say something (or drop a recording) and see what the phone agent would do.

  python -E -s demo/server.py   ->   open http://127.0.0.1:8770

Whisper large-v3-turbo runs on this computer: no audio leaves the machine.
The two rules use the thresholds chosen in the study (tune speakers, <= 5 % slipped errors),
read from results/policy_phone_<model>.csv, so the demo behaves exactly like the measured agent.
It uses large-v3-turbo when its study results exist, otherwise Whisper small (and says so).
"""
from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))

from faster_whisper import WhisperModel  # noqa: E402

from common import DATA, RESULTS  # noqa: E402
from prepare import to_clean_wav, to_phone_wav  # noqa: E402
from transcribe_local import MODELS  # noqa: E402

PORT = 8770
MAX_BYTES = 10 * 1024 * 1024  # about a minute of audio is far below this


def load_thresholds() -> tuple[str, dict[str, float]]:
    """(model, thresholds) from the best model whose study results exist."""
    for model in ("turbo", "small"):
        path = RESULTS / f"policy_phone_{model}.csv"
        if path.exists():
            with open(path, encoding="utf-8", newline="") as f:
                return model, {r["policy"]: float(r["threshold"]) for r in csv.DictReader(f)}
    raise SystemExit("no results/policy_phone_<model>.csv: run `python policy_words.py phone turbo` (or small) first")


def decide(words: list[dict], t: dict[str, float]) -> dict:
    """Same rules as policy_words.py: sentence = mean log-confidence, targeted = each word's confidence."""
    if not words:
        return {"sentence_conf": None, "repeat": True, "flagged": []}
    conf = sum(math.log(max(w["p"], 1e-6)) for w in words) / len(words)
    return {"sentence_conf": round(conf, 4), "repeat": conf < t["sentence"],
            "flagged": [i for i, w in enumerate(words) if w["p"] < t["targeted"]]}


class Handler(BaseHTTPRequestHandler):
    model: WhisperModel
    model_name: str
    thresholds: dict[str, float]

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/thresholds":
            self._json(200, self.thresholds | {"model": self.model_name})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        url = urlparse(self.path)
        if url.path != "/transcribe":
            return self._json(404, {"error": "not found"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._json(400, {"error": "bad Content-Length"})
        if not 0 < size <= MAX_BYTES:
            return self._json(413, {"error": f"audio must be between 1 byte and {MAX_BYTES // 1_000_000} MB"})
        phone = parse_qs(url.query).get("phone", ["1"])[0] == "1"
        with tempfile.TemporaryDirectory() as tmp:
            src, clean, tel = Path(tmp) / "upload.bin", Path(tmp) / "clean.wav", Path(tmp) / "phone.wav"
            src.write_bytes(self.rfile.read(size))
            try:
                to_clean_wav(src, clean)
                if phone:
                    to_phone_wav(clean, tel)
            except RuntimeError as err:  # ffmpeg could not read the file
                print(json.dumps({"event": "decode_failed", "error": str(err)[-200:]}))
                return self._json(422, {"error": "could not read this audio file"})
            segs, info = self.model.transcribe(str(tel if phone else clean), language="fr", word_timestamps=True,
                                               beam_size=1, temperature=0.0, condition_on_previous_text=False)
            words = [{"w": w.word.strip(), "p": round(w.probability, 4)} for s in segs for w in s.words]
        result = {"phone": phone, "model": self.model_name, "duration": round(info.duration, 2), "words": words,
                  "text": " ".join(w["w"] for w in words), "thresholds": self.thresholds} | decide(words, self.thresholds)
        print(json.dumps({"event": "transcribed", "phone": phone, "seconds": result["duration"],
                          "words": len(words), "repeat": result["repeat"], "flagged": len(result["flagged"])}))
        self._json(200, result)

    def log_message(self, fmt: str, *args) -> None:  # keep the console to the JSON lines above
        pass


def _self_check() -> None:
    t = {"sentence": -0.3, "targeted": 0.5}
    d = decide([{"w": "rue", "p": 0.9}, {"w": "Mbarga", "p": 0.2}], t)
    assert d["flagged"] == [1] and d["repeat"] is True  # mean log(0.9, 0.2) = -0.86 < -0.3
    assert decide([], t)["repeat"] is True


if __name__ == "__main__":
    _self_check()
    Handler.model_name, Handler.thresholds = load_thresholds()
    Handler.model = WhisperModel(MODELS[Handler.model_name], device="cpu", compute_type="int8", cpu_threads=8,
                                 download_root=str(DATA / "models"))
    print(json.dumps({"event": "ready", "url": f"http://127.0.0.1:{PORT}", "model": Handler.model_name,
                      "thresholds": Handler.thresholds}))
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
