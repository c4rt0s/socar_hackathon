#!/usr/bin/env python3
"""VFM + PTA engineering dashboard server (Python standard library only).

    python dashboard/server.py                 # default dataset (configs/default.json)
    python dashboard/server.py --config my.json --port 8800

Serves dashboard/static/index.html and a JSON API over vfm.workbench.Workbench:

    GET  /api/status                      -> {"status", "ready", "error"}
    GET  /api/overview                    -> wells, PTA events, settings, notes
    GET  /api/well?name=GL-08             -> daily series baseline vs coupled, tests, IPR, estimators
    GET  /api/pta?event=GL-08@2025-08-19T09:00 -> log-log series, radial window, results, QC flags
    POST /api/settings   {partial CouplingSettings}           -> recompute, save judgement, overview
    POST /api/pta/rerun  {"event": id, "overrides": {...}}     -> rerun one build-up, pta payload
    POST /api/export     {"name": "scenario_a"}                -> write scenario folder
    POST /api/reset      {}                                    -> discard judgement (baseline defaults)
    GET  /api/link                         -> PTA vs VFM values per build-up (kh, skin, p*, p-bar vs Pr, PI vs J)
    GET  /api/models                       -> allocation models, status, active model, benchmark summary
    GET  /api/models/well?name=GL-08&models=physics_wls,gbm -> daily series per model + tests
    POST /api/models/active  {"name": "gbm"}            -> set active model (computes in background)
    POST /api/models/compute {"names": ["gbm", "rf"]}   -> compute models in background
    GET  /api/benchmark                    -> walk-forward / LOTO results per model, per well, per test
    POST /api/benchmark/run  {"models": [...]}           -> rerun scripts/09 in background
    GET  /api/corr?source=well_tests|daily|hourly&well=all|GL-01&method=pearson|spearman|kendall
    GET  /api/scatter?source=..&well=..&x=col&y=col
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import traceback
from urllib.parse import parse_qs, urlparse
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vfm.config import load_config  # noqa: E402
from vfm.coupling import CouplingSettings  # noqa: E402
from vfm.workbench import Workbench, to_jsonable  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
STATE = {"wb": None, "error": None}


def _prepare(cfg_path):
    try:
        wb = Workbench(load_config(cfg_path), progress=lambda m: print(f"[workbench] {m}", flush=True))
        STATE["wb"] = wb
        wb.prepare()
        print("[workbench] ready", flush=True)
    except Exception as exc:  # surfaced through /api/status
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()


class Handler(BaseHTTPRequestHandler):
    server_version = "VFMDashboard/1.0"

    def log_message(self, fmt, *args):  # quieter console
        if "/api/status" not in (args[0] if args else ""):
            sys.stderr.write("[http] " + fmt % args + "\n")

    def _send(self, code, payload=None, content_type="application/json", body: bytes | None = None,
              filename: str | None = None):
        data = body if body is not None else json.dumps(to_jsonable(payload), allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        if filename:
            safe_fn = "".join(c for c in filename if c.isalnum() or c in "._-")
            self.send_header("Content-Disposition", f'attachment; filename="{safe_fn}"')
        self.end_headers()
        self.wfile.write(data)

    def _wb(self):
        wb = STATE["wb"]
        if wb is None or wb.status != "ready":
            self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"error": STATE["error"] or "workbench not ready",
                                                        "status": getattr(wb, "status", "starting")})
            return None
        return wb

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ("/", "/index.html"):
                return self._send(200, body=(STATIC / "index.html").read_bytes(), content_type="text/html; charset=utf-8")
            if url.path.startswith("/static/"):
                f = (STATIC / url.path[len("/static/"):]).resolve()
                if STATIC.resolve() in f.parents and f.is_file():
                    ctype = {".js": "text/javascript", ".css": "text/css", ".html": "text/html"}.get(f.suffix, "application/octet-stream")
                    return self._send(200, body=f.read_bytes(), content_type=ctype)
                return self._send(404, {"error": "not found"})
            if url.path == "/api/status":
                wb = STATE["wb"]
                return self._send(200, {"status": getattr(wb, "status", "starting"), "ready": bool(wb and wb.status == "ready"),
                                        "error": STATE["error"]})
            wb = self._wb()
            if wb is None:
                return
            if url.path == "/api/overview":
                return self._send(200, wb.overview_payload())
            if url.path == "/api/well":
                return self._send(200, wb.well_payload(q["name"]))
            if url.path == "/api/pta":
                return self._send(200, wb.pta_payload(q["event"]))
            if url.path == "/api/link":
                return self._send(200, wb.link_payload())
            if url.path == "/api/models":
                return self._send(200, wb.models_payload())
            if url.path == "/api/models/well":
                names = [n for n in (q.get("models") or wb.active_model).split(",") if n]
                return self._send(200, wb.model_well_payload(q["name"], names))
            if url.path == "/api/benchmark":
                return self._send(200, wb.benchmark_payload())
            if url.path == "/api/corr":
                return self._send(200, wb.correlation_payload(q.get("source", "well_tests"), q.get("well", "all"),
                                                              q.get("method", "pearson")))
            if url.path == "/api/vfm/diagnostics":
                return self._send(200, wb.vfm_diagnostics_payload(q.get("method", "pearson")))
            if url.path == "/api/scatter":
                return self._send(200, wb.scatter_payload(q.get("source", "well_tests"), q.get("well", "all"), q["x"], q["y"]))
            if url.path == "/api/monitoring":
                start_ms = int(q["start_ms"]) if q.get("start_ms") else None
                end_ms = int(q["end_ms"]) if q.get("end_ms") else None
                max_pts = int(q["max_pts"]) if q.get("max_pts") else 1800
                return self._send(200, wb.monitoring_payload(
                    q["name"], res=q.get("res", "auto"), start_ms=start_ms, end_ms=end_ms,
                    max_pts=max_pts, model=q.get("model") or None))
            if url.path == "/api/export/vfm":
                model = q.get("model") or wb.active_model or "physics_wls"
                rate_type = q.get("rate_type", "alloc")
                fmt = q.get("format", "csv")
                df = wb.vfm_deliverable_df(model, rate_type=rate_type)
                raw, ctype, ext = wb.export_table_bytes(df, fmt=fmt, sheet_name="VFM")
                return self._send(200, body=raw, content_type=ctype, filename=f"vfm_submission_{model}.{ext}")
            if url.path == "/api/export/pta":
                source = q.get("source", "hybrid_intersection")
                fmt = q.get("format", "csv")
                df = wb.pta_deliverable_df(source=source)
                raw, ctype, ext = wb.export_table_bytes(df, fmt=fmt, sheet_name="PTA")
                return self._send(200, body=raw, content_type=ctype, filename=f"pta_submission_{source}.{ext}")
            if url.path == "/api/export/report":
                raw = wb.engineering_report_html_bytes()
                return self._send(200, body=raw, content_type="text/html; charset=utf-8",
                                  filename="socar_vfm_pta_engineering_report.html")
            return self._send(404, {"error": f"unknown endpoint {url.path}"})
        except KeyError as exc:
            self._send(400, {"error": f"bad request: {exc}"})
        except Exception as exc:
            traceback.print_exc()
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):  # noqa: N802
        url = urlparse(self.path)
        try:
            wb = self._wb()
            if wb is None:
                return
            body = self._body()
            if url.path == "/api/settings":
                wb.update_settings(body)
                return self._send(200, wb.overview_payload())
            if url.path == "/api/pta/rerun":
                return self._send(200, wb.rerun_pta_event(body["event"], body.get("overrides") or {}))
            if url.path == "/api/pta/manual_fit":
                return self._send(200, wb.manual_pta_fit_payload(body))
            if url.path == "/api/pta/rate_basis":
                wb.set_pta_rate_basis(str(body["rate_basis"]))
                return self._send(200, wb.overview_payload())
            if url.path == "/api/audit":
                return self._send(200, wb.audit_payload(body))
            if url.path == "/api/upload_and_run":
                return self._send(200, wb.upload_and_run_payload(body))
            if url.path == "/api/export":
                return self._send(200, wb.export(str(body.get("name") or "")))
            if url.path == "/api/models/active":
                wb.set_active_model(str(body["name"]))
                return self._send(200, wb.models_payload())
            if url.path == "/api/vfm/retrain_custom":
                return self._send(200, wb.retrain_custom_vfm_payload(body))
            if url.path == "/api/vfm/grid_search":
                return self._send(200, wb.grid_search_vfm_payload(body))
            if url.path == "/api/models/compute":
                for n in body.get("names") or []:
                    if n not in wb.model_daily and wb.model_status.get(n) != "running":
                        threading.Thread(target=wb.compute_model, args=(n,), daemon=True).start()
                return self._send(200, wb.models_payload())
            if url.path == "/api/benchmark/run":
                wb.run_benchmark_async(body.get("models") or None)
                return self._send(200, {"status": wb.bench_status})
            if url.path == "/api/reset":
                with wb.lock:
                    wb.settings = CouplingSettings(enabled=False)
                    wb.pta_overrides = {}
                    wb.pta_rate_basis = str(wb.cfg.section("pta").get("rate_basis", "oil"))
                    wb.run_pta()
                    wb.apply()
                    wb.save_judgement()
                return self._send(200, wb.overview_payload())
            return self._send(404, {"error": f"unknown endpoint {url.path}"})
        except KeyError as exc:
            self._send(400, {"error": f"bad request: {exc}"})
        except Exception as exc:
            traceback.print_exc()
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def main():
    ap = argparse.ArgumentParser(description="VFM + PTA engineering dashboard")
    ap.add_argument("--config", help="dataset config JSON (merged over configs/default.json)")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    dash = load_config(args.config).section("dashboard")
    host, port = args.host or dash.get("host", "127.0.0.1"), args.port or int(dash.get("port", 8765))
    threading.Thread(target=_prepare, args=(args.config,), daemon=True).start()
    srv = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"Dashboard on {url} (baseline loads in the background; first run takes about a minute)", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
