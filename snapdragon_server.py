from __future__ import annotations

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import laya_snapdragon
from laya_snapdragon import Agent


class Runtime:
    def __init__(self, models: Path) -> None:
        # The multilingual ONNX export names this output act_logits; the English
        # upstream checkpoint uses act. The tensor content and shape are equal.
        laya_snapdragon.OUTPUTS = ["logits", "act_logits"]
        self.agent = Agent(models, device="npu")
        self.lock = threading.Lock()

    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            return self.agent.predict(state, questions)


def make_handler(runtime: Runtime):
    class Handler(BaseHTTPRequestHandler):
        server_version = "laya-multilingual-qnn/0.1"

        def send_json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self.send_json(HTTPStatus.OK, {"status": "ok", "backend": "qnn-htp", "variant": "multilingual"})

        def do_POST(self) -> None:
            if self.path not in ("/decision", "/v1/systemone"):
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                questions = request["questions"]
                if not isinstance(questions, dict) or not questions:
                    raise ValueError("questions must be a non-empty object")
                started = time.perf_counter()
                result = runtime.predict(request["state"], questions)
                result["local_runtime"] = {
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    "backend": "qnn-htp",
                    "variant": "multilingual",
                    "output_tokens": 0,
                }
                self.send_json(HTTPStatus.OK, result)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve multilingual Laya decisions on Snapdragon QNN")
    parser.add_argument("--models", type=Path, default=Path("models/laya-multilingual-npu/runtime"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()

    runtime = Runtime(args.models)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(f"Laya QNN multilingual listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

