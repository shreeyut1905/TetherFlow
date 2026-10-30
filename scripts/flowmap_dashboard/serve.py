# SPDX-FileCopyrightText: Copyright (c) 2026 Shreeyut Maheshwari. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local dashboard comparing flow-map training runs that use ``FlowMapDashboardCallback``.

Each run directory is a run's ``log_config.save_path``; the callback writes its artifacts below
``<run_dir>/dashboard/``:

    run.json                       run label and description
    metrics.jsonl                  one JSON object per logged training iteration
    eval/<iteration:07d>/meta.json eval metadata, written last (atomically) once the images exist
    eval/<iteration:07d>/p<j>_nfe<k>.jpg

The server uses only the standard library, so any Python 3.10+ runs it, and binds to localhost:

    python scripts/flowmap_dashboard/serve.py \
        --run stage1=TETHERFLOW_OUTPUT/tetherflow/flux_tetherflow/config_tetherflow_stage1 \
        --run stage2=TETHERFLOW_OUTPUT/tetherflow/flux_tetherflow/config_tetherflow_stage2 --port 8765

From another machine, forward the port: ``ssh -L 8765:localhost:8765 <host>``.
"""

import argparse
import json
import math
import mimetypes
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_FILES = {"index.html", "app.js", "style.css"}
RUN_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
EVAL_DIR_RE = re.compile(r"^\d{7}$")
MEDIA_NAME_RE = re.compile(r"^p\d{1,3}_nfe\d{1,3}\.jpg$")


def _finite_or_none(value: Any) -> Any:
    """JSON cannot carry NaN/inf (a diverged loss); send them as null so one bad value cannot break the page."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite_or_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(v) for v in value]
    return value


class RunState:
    """Incrementally parsed artifacts of one run directory."""

    def __init__(self, key: str, run_dir: Path, label: str | None):
        self.key = key
        self.run_dir = run_dir
        self.dashboard_dir = run_dir / "dashboard"
        self.cli_label = label
        self.rows: list[dict[str, Any]] = []
        self._offset = 0
        self._pending = b""
        self._lock = threading.Lock()

    def _read_new_metrics(self) -> None:
        path = self.dashboard_dir / "metrics.jsonl"
        if not path.is_file():
            return
        size = path.stat().st_size
        if size < self._offset:
            # The file was replaced (e.g. a fresh run in the same directory): start over.
            self.rows, self._offset, self._pending = [], 0, b""
        if size == self._offset:
            return
        with open(path, "rb") as f:
            f.seek(self._offset)
            chunk = f.read(size - self._offset)
        self._offset = size
        data = self._pending + chunk
        lines = data.split(b"\n")
        # Keep a trailing partial line for the next read.
        self._pending = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("iteration"), int):
                # A resumed run re-logs from its checkpoint: the abandoned attempt's rows from that
                # iteration on no longer describe the run, so drop them.
                while self.rows and self.rows[-1]["iteration"] >= row["iteration"]:
                    self.rows.pop()
                self.rows.append(_finite_or_none(row))

    def _read_evals(self) -> list[dict[str, Any]]:
        eval_root = self.dashboard_dir / "eval"
        if not eval_root.is_dir():
            return []
        evals = []
        for entry in sorted(eval_root.iterdir()):
            if not (entry.is_dir() and EVAL_DIR_RE.match(entry.name)):
                continue
            meta_path = entry / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(meta, dict):
                continue
            meta["dir"] = entry.name
            evals.append(_finite_or_none(meta))
        return evals

    def _read_info(self) -> dict[str, Any]:
        path = self.dashboard_dir / "run.json"
        try:
            info = json.loads(path.read_text()) if path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            info = {}
        if self.cli_label:
            info["label"] = self.cli_label
        info.setdefault("label", self.key)
        return info

    def _is_current(self, meta: dict[str, Any], by_iteration: dict[int, dict[str, Any]]) -> bool:
        """Whether an eval belongs to the run's current attempt rather than one abandoned by a resume.

        The callback logs an iteration's metrics before its eval, so an eval past the last logged iteration, or
        older than the current attempt's row for its iteration, was written by an abandoned attempt (until the
        resumed run reaches that iteration and overwrites it).
        """
        if not self.rows:
            return True
        iteration = meta.get("iteration")
        if not isinstance(iteration, int) or iteration > self.rows[-1]["iteration"]:
            return False
        row = by_iteration.get(iteration)
        return row is None or (meta.get("wall_time") or 0) >= (row.get("wall_time") or 0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._read_new_metrics()
            rows = list(self.rows)
            by_iteration = {row["iteration"]: row for row in rows}
            keys = sorted({k for row in rows for k in row if k != "iteration"})
            columns = {"iteration": [row["iteration"] for row in rows]}
            for k in keys:
                columns[k] = [row.get(k) for row in rows]
            return {
                "key": self.key,
                "info": self._read_info(),
                "metrics": columns,
                "evals": [meta for meta in self._read_evals() if self._is_current(meta, by_iteration)],
            }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "FlowMapDashboard/1.0"
    runs: dict[str, RunState] = {}

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - BaseHTTPRequestHandler's signature
        pass

    def _send(self, status: int, body: bytes, content_type: str, cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self) -> None:
        self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path.startswith("/static/"):
            return self._static(path[len("/static/") :])
        if path == "/api/state":
            body = json.dumps({"runs": [run.snapshot() for run in self.runs.values()]}, allow_nan=False).encode()
            return self._send(HTTPStatus.OK, body, "application/json")
        if path.startswith("/media/"):
            return self._media(path[len("/media/") :])
        return self._not_found()

    def _static(self, name: str) -> None:
        if name not in STATIC_FILES:
            return self._not_found()
        content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type == "application/javascript":
            content_type += "; charset=utf-8"
        self._send(HTTPStatus.OK, (STATIC_DIR / name).read_bytes(), content_type)

    def _media(self, rest: str) -> None:
        # /media/<run_key>/<eval_dir>/<file>, every part whitelisted so no path can escape the run.
        parts = rest.split("/")
        if len(parts) != 3:
            return self._not_found()
        run_key, eval_dir, name = parts
        run = self.runs.get(run_key)
        if run is None or not EVAL_DIR_RE.match(eval_dir) or not MEDIA_NAME_RE.match(name):
            return self._not_found()
        path = run.dashboard_dir / "eval" / eval_dir / name
        if not path.is_file():
            return self._not_found()
        # Eval images never change once their meta.json exists, so the browser may cache them.
        self._send(HTTPStatus.OK, path.read_bytes(), "image/jpeg", cache="max-age=86400")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="KEY=RUN_DIR[=LABEL]: a run's log_config.save_path; repeat once per run (order sets the colors)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    runs = {}
    for spec in args.run:
        key, sep, rest = spec.partition("=")
        if not sep or not RUN_KEY_RE.match(key):
            parser.error(f"--run expects KEY=RUN_DIR[=LABEL] with KEY in [A-Za-z0-9_-], got {spec!r}")
        run_dir, _, label = rest.partition("=")
        if key in runs:
            parser.error(f"duplicate run key {key!r}")
        runs[key] = RunState(key, Path(run_dir).expanduser().resolve(), label or None)
    DashboardHandler.runs = runs

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Serving {len(runs)} run(s) at http://{args.host}:{args.port}/", flush=True)
    for run in runs.values():
        print(f"  {run.key}: {run.run_dir}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
