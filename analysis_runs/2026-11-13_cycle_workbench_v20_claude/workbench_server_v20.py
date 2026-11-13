"""Local server for the v20 multi-cycle workbench.

Made by Claude (Claude Code, Opus 5.5), 2026-11-13. Serves the page, keeps the
sample library in project files, writes one PNG per saved sample, and reads the
June v19 samples so they can be opened in the new page.

    python workbench_server_v20.py            -> http://127.0.0.1:8766/
    python workbench_server_v20.py --out DIR  (write samples somewhere else, e.g. for a test)
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
PAGE = HERE / "cycle_workbench_v20.html"
V19_JSON = HERE.parents[0] / "2026-06-04_segment_experiments" / "segment_cycle_samples_v19_saved.json"
OUT = HERE  # replaced by --out

CSV_FIELDS = [
    "saved_at", "id", "series", "anchor_type", "pre_days", "post_days", "reference", "reference_anchor",
    "cycle", "cycle_anchor", "amp_scale", "time_scale", "shift_days", "rmse", "overlap_days",
    "visual_score", "note", "data_end", "tool",
]


def paths() -> dict[str, Path]:
    return {
        "json": OUT / "cycle_workbench_v20_samples.json",
        "csv": OUT / "cycle_workbench_v20_samples.csv",
        "backups": OUT / "sample_backups",
        "snapshots": OUT / "snapshots",
    }


def read_samples() -> list[dict]:
    p = paths()["json"]
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("samples", [])


def write_csv(samples: list[dict]) -> None:
    with paths()["csv"].open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for s in samples:
            for c in s.get("cycles", []):
                if c.get("is_reference"):
                    continue
                w.writerow({
                    "saved_at": s.get("saved_at"), "id": s.get("id"), "series": s.get("series"),
                    "anchor_type": s.get("anchor_type"), "pre_days": s.get("pre_days"), "post_days": s.get("post_days"),
                    "reference": s.get("reference"), "reference_anchor": s.get("reference_anchor"),
                    "cycle": c.get("id"), "cycle_anchor": c.get("anchor"), "amp_scale": c.get("amp"),
                    "time_scale": c.get("time"), "shift_days": c.get("shift"), "rmse": c.get("rmse"),
                    "overlap_days": c.get("overlap_days"), "visual_score": s.get("visual_score"),
                    "note": s.get("note"), "data_end": s.get("data_end"), "tool": s.get("tool"),
                })


def save(payload: dict) -> dict:
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("samples must be a list")
    p = paths()
    p["backups"].mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    if p["json"].exists():  # keep the previous library before overwriting it
        (p["backups"] / f"cycle_workbench_v20_samples_{stamp}.json").write_text(
            p["json"].read_text(encoding="utf-8"), encoding="utf-8")
    doc = {
        "version": "v20",
        "tool": "cycle_workbench_v20 (made by Claude)",
        "project_saved_at": datetime.now(timezone.utc).isoformat(),
        "samples": samples,
    }
    p["json"].write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(samples)
    out = {"ok": True, "samples": len(samples), "json": str(p["json"]), "csv": str(p["csv"])}
    snap = payload.get("snapshot")
    if snap and snap.get("png") and re.fullmatch(r"s\d+", str(snap.get("id", ""))):
        p["snapshots"].mkdir(parents=True, exist_ok=True)
        raw = base64.b64decode(snap["png"].split(",", 1)[-1])
        target = p["snapshots"] / f"{snap['id']}.png"
        target.write_bytes(raw)
        out["snapshot"] = str(target)
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "CycleWorkbenchV20/1.0"

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj: dict) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._json(200, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html", "/cycle_workbench_v20.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/samples":
            self._json(200, {"ok": True, "samples": read_samples()})
        elif path == "/v19_samples":
            data = json.loads(V19_JSON.read_text(encoding="utf-8")) if V19_JSON.exists() else {}
            self._json(200, {"ok": True, "samples": data.get("samples", [])})
        elif m := re.fullmatch(r"/snapshots/(s\d+)\.png", path):
            f = paths()["snapshots"] / f"{m.group(1)}.png"
            if f.exists():
                self._send(200, f.read_bytes(), "image/png")
            else:
                self._json(404, {"ok": False, "error": "no snapshot for this sample"})
        elif path == "/status":
            self._json(200, {"ok": True, "out": str(OUT), "samples": len(read_samples())})
        else:
            self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/save":
            self._json(404, {"ok": False, "error": "not found"})
            return
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self._json(200, save(json.loads(raw.decode("utf-8"))))
        except Exception as exc:  # report the error to the page
            self._json(400, {"ok": False, "error": str(exc)})

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    OUT = Path(args.out).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Workbench v20 (made by Claude): http://{args.host}:{args.port}/")
    print(f"Samples: {paths()['json']}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
