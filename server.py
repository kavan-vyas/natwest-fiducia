"""Minimal localhost chat server backed by Ollama.

Run:  python3 server.py
Open: http://localhost:8000

No dependencies — Python standard library only.
"""

import json
import re
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"
PORT = 8000

INDEX_HTML = (Path(__file__).parent / "index.html").read_bytes()


def ask_ollama(prompt: str) -> str:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    content = data["message"]["content"]
    # strip <think> blocks in case the model emits them anyway
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        prompt = body.get("prompt", "").strip()
        if not prompt:
            self._send_json({"error": "empty prompt"}, status=400)
            return
        try:
            answer = ask_ollama(prompt)
            self._send_json({"answer": answer})
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=502)

    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    print(f"Serving on http://localhost:{PORT}  (model: {MODEL})")
    HTTPServer(("localhost", PORT), Handler).serve_forever()
2