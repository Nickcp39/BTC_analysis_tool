from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


OUT = Path(__file__).resolve().parent
SAVED_JSON = OUT / "segment_cycle_samples_v19_saved.json"
SAVED_CSV = OUT / "segment_cycle_samples_v19_saved.csv"
BACKUP_DIR = OUT / "sample_backups"
DUPLICATE_TOLERANCE = {"amp": 0.03, "time": 0.03, "shift": 10}


def _now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_csv(samples: list[dict]) -> None:
    fields = [
        "saved_at",
        "case_id",
        "case_name",
        "pair",
        "left_cycle",
        "right_cycle",
        "anchor_type",
        "window_id",
        "window_label",
        "left_anchor",
        "right_anchor",
        "pre_days",
        "post_days",
        "amp_scale",
        "time_scale",
        "shift_days",
        "rmse",
        "overlap_days",
        "visual_score",
        "note",
    ]
    with SAVED_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({field: sample.get(field, "") for field in fields})


def _comparison_key(sample: dict) -> tuple:
    return (
        sample.get("left_cycle", ""),
        sample.get("right_cycle", ""),
        sample.get("anchor_type", ""),
        sample.get("window_id", ""),
        sample.get("left_anchor", ""),
        sample.get("right_anchor", ""),
        sample.get("pre_days", ""),
        sample.get("post_days", ""),
    )


def _params_are_near(a: dict, b: dict) -> bool:
    return (
        abs(float(a.get("amp_scale") or 0) - float(b.get("amp_scale") or 0)) <= DUPLICATE_TOLERANCE["amp"]
        and abs(float(a.get("time_scale") or 0) - float(b.get("time_scale") or 0)) <= DUPLICATE_TOLERANCE["time"]
        and abs(float(a.get("shift_days") or 0) - float(b.get("shift_days") or 0)) <= DUPLICATE_TOLERANCE["shift"]
    )


def _dedupe_samples(samples: list[dict]) -> list[dict]:
    out: list[dict] = []
    for sample in samples:
        duplicate_index = next(
            (
                i
                for i, old in enumerate(out)
                if _comparison_key(old) == _comparison_key(sample) and _params_are_near(old, sample)
            ),
            -1,
        )
        if duplicate_index >= 0:
            out[duplicate_index] = {
                **out[duplicate_index],
                **sample,
                "replaced_sample": out[duplicate_index].get("saved_at"),
                "duplicate_policy": (
                    "same comparison and params within "
                    f"amp {DUPLICATE_TOLERANCE['amp']}, "
                    f"time {DUPLICATE_TOLERANCE['time']}, "
                    f"shift {DUPLICATE_TOLERANCE['shift']}d"
                ),
            }
        else:
            out.append(sample)
    return out


def save_payload(payload: dict) -> dict:
    samples = payload.get("samples") or []
    if not isinstance(samples, list):
        raise ValueError("payload.samples must be a list")
    samples = _dedupe_samples(samples)
    payload = {
        **payload,
        "samples": samples,
        "version": payload.get("version") or "v19",
        "project_saved_at": datetime.now(timezone.utc).isoformat(),
        "duplicate_policy": DUPLICATE_TOLERANCE,
    }
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (BACKUP_DIR / f"segment_cycle_samples_v19_{_now_slug()}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(samples)
    return {
        "ok": True,
        "samples": len(samples),
        "json": str(SAVED_JSON),
        "csv": str(SAVED_CSV),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "SegmentSampleSaveServer/1.0"

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/status":
            self._send_json(
                200,
                {
                    "ok": True,
                    "json_exists": SAVED_JSON.exists(),
                    "json": str(SAVED_JSON),
                    "csv": str(SAVED_CSV),
                },
            )
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/save_samples":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            result = save_payload(payload)
            self._send_json(200, result)
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Segment sample save server: http://{args.host}:{args.port}/status")
    print(f"Saving to: {SAVED_JSON}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
