#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


TOOL_ROOT = Path(__file__).resolve().parent
SRC_DIR = TOOL_ROOT / "src"


def metadata() -> dict[str, Any]:
    return {
        "name": "gemorna",
        "functions": [
            "generate_cds_open",
            "generate_cds_closed",
            "generate_5utr",
            "generate_3utr",
            "score_5utr",
            "score_3utr",
        ],
        "default_device": "cpu",
    }


def _load_service():
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    import torch
    from gemorna_services import GemornaService

    return GemornaService(device=torch.device("cpu"))


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def call_with_service(service, function: str, payload: dict[str, Any]) -> dict[str, Any]:
    seed = payload.get("seed")
    if function == "generate_cds_open":
        result = service.generate_cds_open(payload["protein_sequence"], seed=seed)
    elif function == "generate_cds_closed":
        result = service.generate_cds_closed(payload["protein_sequence"], seed=seed)
    elif function == "generate_5utr":
        result = service.generate_utr("5utr", payload["length"], seed=seed)
    elif function == "generate_3utr":
        result = service.generate_utr("3utr", payload["length"], seed=seed)
    elif function == "score_5utr":
        result = service.score_utr("5utr", payload["sequence"])
    elif function == "score_3utr":
        result = service.score_utr("3utr", payload["sequence"])
    else:
        raise ValueError(f"未知 GemORNA 函数: {function}")
    return _to_jsonable(result)


def run_call(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.input).read_text())
    try:
        service = _load_service()
        result = call_with_service(service, args.function, payload)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False))
    return 0


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GemORNAAdapter/0.1"

        def do_GET(self) -> None:
            if self.path == "/health":
                self._write_json({"status": "ok", "tool": "gemorna"})
                return
            self.send_error(404)

        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/shutdown":
                self._write_json({"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if self.path != "/call":
                self.send_error(404)
                return
            try:
                result = call_with_service(
                    service,
                    body["function"],
                    body.get("input", {}),
                )
                self._write_json({"result": result})
            except Exception as exc:
                self._write_json({"error": str(exc)}, status=400)

        def log_message(self, format: str, *args) -> None:
            return

        def _write_json(self, payload: dict[str, Any], status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def run_serve(args: argparse.Namespace) -> int:
    control_dir = Path(args.control_dir)
    control_dir.mkdir(parents=True, exist_ok=True)
    service = _load_service()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    ready = {
        "url": f"http://127.0.0.1:{server.server_port}",
        "transport": "http",
        "device": "cpu",
    }
    (control_dir / "ready.json").write_text(json.dumps(ready, ensure_ascii=False))
    server.serve_forever()
    server.server_close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser("GemORNA MCP adapter")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("metadata")

    call_parser = sub.add_parser("call")
    call_parser.add_argument("--function", required=True)
    call_parser.add_argument("--input", required=True)
    call_parser.add_argument("--output", required=True)

    serve_parser = sub.add_parser("serve")
    serve_parser.add_argument("--control-dir", required=True)

    args = parser.parse_args()
    if args.command == "metadata":
        print(json.dumps(metadata(), ensure_ascii=False))
        return 0
    if args.command == "call":
        return run_call(args)
    return run_serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
