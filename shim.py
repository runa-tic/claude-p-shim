#!/usr/bin/env python3
"""claude-p-shim: an Anthropic Messages API endpoint backed by headless `claude -p`.

A client that speaks the Messages API (system, tools, messages -> text or tool_use) is pointed at
this server instead of api.anthropic.com. Each request is flattened into one prompt, answered by a
tool-less `claude -p` call whose reply is forced into a JSON shape, and wrapped back into the API's
response shape, streaming or not. The client keeps its own loop and runs its own tools.

This is a translation layer, not the API. What differs, and matters:
  - the model picks tools from schemas pasted as text, not through its native tool interface
  - history is flattened into a transcript; images and thinking blocks are dropped
  - every request starts a fresh CLI process: seconds of latency, no prompt caching
  - "streaming" is the finished reply replayed as SSE events
  - max_tokens, temperature, stop_sequences and tool ids from the model are not honoured

Personal, local use: binds to 127.0.0.1 and runs under whatever login `claude` already has.
Stdlib only.  Run:  python shim.py [--port 8765]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TIMEOUT = int(os.environ.get("SHIM_TIMEOUT", "300"))
MODEL_OVERRIDE = os.environ.get("SHIM_MODEL")           # force one model whatever the client asks
PARALLEL = threading.Semaphore(int(os.environ.get("SHIM_MAX_PARALLEL", "4")))
PING_EVERY = 5                                           # seconds between SSE keep-alives

# The reply shape `claude -p --json-schema` is held to. `input` is left open: each tool brings its
# own schema, and those are shown to the model as text.
REPLY_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "input": {"type": "object"}},
                "required": ["name", "input"],
            },
        },
    },
    "required": ["text", "tool_calls"],
})

PROTOCOL = """

# How to act (read carefully)
You are reached through a text relay. The ONLY function you can call is `StructuredOutput`.
The actions listed under "Actions" are NOT functions you can call: invoking one directly fails with
"No such tool available". They belong to the caller. To request an action, call `StructuredOutput`
once and put the request in its `tool_calls` array as {"name": "<action name>", "input": {...}},
with an `input` that satisfies that action's input schema. The caller runs it and sends the result
back in the next request as a [tool_result]. You may request several actions in one reply.
When you need no action, put your answer in `text` and leave `tool_calls` empty.
Every listed action is available this way; never say one is unavailable, and never invent a
result. `text` alongside requested actions is shown to the user as brief commentary.
"""

# The model's last instruction before it replies; the direct-call habit is strong (see README).
TAIL_TOOLS = ("Give the assistant's next step by calling StructuredOutput exactly once. Request "
              "actions only inside its tool_calls field; do not call an action directly.")


def shim_cwd() -> Path:
    """One fixed, empty directory to run `claude` in, so it finds no project CLAUDE.md or settings."""
    uid = os.getuid() if hasattr(os, "getuid") else None
    d = Path(tempfile.gettempdir()) / ("claude-p-shim-cwd" + (f"-{uid}" if uid is not None else ""))
    d.mkdir(mode=0o700, exist_ok=True)
    return d


def _text_of(content) -> str:
    """Flatten a Messages `content` value (string or block list) that should be plain text."""
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            out.append(b.get("text", ""))
        elif b.get("type") == "image":
            out.append("[image omitted by relay]")
    return "\n".join(out)


def render_system(req: dict) -> str:
    system = _text_of(req.get("system") or "")
    tools = [t for t in req.get("tools") or [] if isinstance(t, dict) and t.get("name")]
    if not tools:
        return system
    parts = [system, PROTOCOL, "# Actions"]
    for t in tools:
        schema = t.get("input_schema") or t.get("parameters") or {}
        parts.append(f"## {t['name']}\n{t.get('description', '')}\n"
                     f"input schema: {json.dumps(schema, ensure_ascii=False)}")
    choice = req.get("tool_choice") or {}
    if isinstance(choice, dict):
        if choice.get("type") == "tool":
            parts.append(f"You MUST request the action `{choice.get('name')}` in this reply.")
        elif choice.get("type") == "any":
            parts.append("You MUST request at least one action in this reply.")
        elif choice.get("type") == "none":
            parts.append("Do not request any action in this reply.")
    return "\n\n".join(parts)


def render_transcript(messages: list, has_tools: bool = False) -> str:
    lines = ["<conversation>"]
    for m in messages or []:
        role, content = m.get("role", "user"), m.get("content")
        if isinstance(content, str):
            lines.append(f"[{role}]\n{content}")
            continue
        for b in content or []:
            kind = b.get("type") if isinstance(b, dict) else None
            if kind == "text":
                lines.append(f"[{role}]\n{b.get('text', '')}")
            elif kind == "tool_use":
                lines.append(f"[assistant tool_call id={b.get('id')} name={b.get('name')}]\n"
                             f"{json.dumps(b.get('input', {}), ensure_ascii=False)}")
            elif kind == "tool_result":
                flag = " error" if b.get("is_error") else ""
                lines.append(f"[tool_result id={b.get('tool_use_id')}{flag}]\n{_text_of(b.get('content'))}")
            elif kind == "image":
                lines.append(f"[{role}]\n[image omitted by relay]")
            # thinking / redacted_thinking blocks are dropped
    lines.append("</conversation>")
    lines.append(TAIL_TOOLS if has_tools else "Give the assistant's next reply.")
    return "\n\n".join(lines)


class CliError(RuntimeError):
    pass


def call_claude(req: dict) -> dict:
    """Run one `claude -p` for this request; return {"text", "tool_calls", "usage"}."""
    exe = shutil.which("claude") or "claude"
    has_tools = bool(req.get("tools"))
    model = MODEL_OVERRIDE or req.get("model") or "claude-haiku-4-5"
    cwd = shim_cwd()
    sys_file = cwd / f"system-{uuid.uuid4().hex}.txt"
    sys_file.write_text(render_system(req) or "You are a helpful assistant.", encoding="utf-8")
    argv = [exe, "-p", "--model", model, "--strict-mcp-config", "--no-session-persistence",
            "--system-prompt-file", str(sys_file), "--output-format", "json"]
    if has_tools:
        argv += ["--json-schema", REPLY_SCHEMA]
    argv += ["--tools", ""]                      # variadic, so last with its one empty value
    env = {**os.environ, "CLAUDE_BRAIN_NO_HOOK": "1", "ENABLE_CLAUDEAI_MCP_SERVERS": "false"}
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        with PARALLEL:
            proc = subprocess.run(argv, input=render_transcript(req.get("messages"), has_tools),
                                  capture_output=True, text=True, encoding="utf-8",
                                  cwd=str(cwd), env=env, timeout=TIMEOUT, creationflags=flags)
    except subprocess.TimeoutExpired:
        raise CliError(f"claude -p timed out after {TIMEOUT}s")
    finally:
        sys_file.unlink(missing_ok=True)
    # `claude -p` reports API / model / usage errors on stdout with rc=1; report both streams.
    try:
        out = json.loads(proc.stdout)
    except ValueError:
        out = None
    if proc.returncode != 0 or not isinstance(out, dict) or out.get("is_error"):
        detail = " | ".join(x for x in (proc.stdout.strip()[:600], proc.stderr.strip()[:300]) if x)
        raise CliError(f"claude -p failed (rc={proc.returncode}): {detail or 'no output'}")
    if has_tools:
        so = out.get("structured_output")
        if not isinstance(so, dict):
            raise CliError(f"claude -p returned no structured_output: {str(out.get('result'))[:300]}")
        text, calls = so.get("text") or "", so.get("tool_calls") or []
    else:
        text, calls = out.get("result") or "", []
    known = {t.get("name") for t in req.get("tools") or [] if isinstance(t, dict)}
    calls = [c for c in calls if isinstance(c, dict) and c.get("name") in known]
    return {"text": text, "tool_calls": calls, "usage": out.get("usage") or {}, "model": model}


def to_message(reply: dict) -> dict:
    content = []
    if reply["text"].strip() or not reply["tool_calls"]:
        content.append({"type": "text", "text": reply["text"]})
    for c in reply["tool_calls"]:
        content.append({"type": "tool_use", "id": "toolu_shim_" + uuid.uuid4().hex[:20],
                        "name": c["name"], "input": c.get("input") or {}})
    u = reply["usage"]
    return {
        "id": "msg_shim_" + uuid.uuid4().hex[:20], "type": "message", "role": "assistant",
        "model": reply["model"], "content": content,
        "stop_reason": "tool_use" if reply["tool_calls"] else "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0),
                  "cache_creation_input_tokens": u.get("cache_creation_input_tokens", 0),
                  "cache_read_input_tokens": u.get("cache_read_input_tokens", 0)},
    }


def sse_events(msg: dict):
    """Replay a finished message as the Messages API's streaming event sequence."""
    start = {**msg, "content": [], "stop_reason": None,
             "usage": {**msg["usage"], "output_tokens": 0}}
    yield "message_start", {"type": "message_start", "message": start}
    for i, block in enumerate(msg["content"]):
        if block["type"] == "text":
            yield "content_block_start", {"type": "content_block_start", "index": i,
                                          "content_block": {"type": "text", "text": ""}}
            yield "content_block_delta", {"type": "content_block_delta", "index": i,
                                          "delta": {"type": "text_delta", "text": block["text"]}}
        else:
            yield "content_block_start", {"type": "content_block_start", "index": i,
                                          "content_block": {**block, "input": {}}}
            yield "content_block_delta", {"type": "content_block_delta", "index": i,
                                          "delta": {"type": "input_json_delta",
                                                    "partial_json": json.dumps(block["input"])}}
        yield "content_block_stop", {"type": "content_block_stop", "index": i}
    yield "message_delta", {"type": "message_delta",
                            "delta": {"stop_reason": msg["stop_reason"], "stop_sequence": None},
                            "usage": {"output_tokens": msg["usage"]["output_tokens"]}}
    yield "message_stop", {"type": "message_stop"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):       # one line per request is logged in do_POST instead
        pass

    def _json(self, status: int, body: dict):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, kind: str, message: str):
        self._json(status, {"type": "error", "error": {"type": kind, "message": message}})

    def _event(self, name: str, data: dict):
        self.wfile.write(f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/v1/models":
            ids = ["claude-haiku-4-5", "claude-sonnet-5-5", "claude-opus-5-5"]
            self._json(200, {"data": [{"type": "model", "id": i, "display_name": i} for i in ids],
                             "has_more": False})
        elif path in ("", "/health"):
            self._json(200, {"ok": True, "backend": "claude -p"})
        else:
            self._error(404, "not_found_error", f"no route {path}")

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        except ValueError:
            return self._error(400, "invalid_request_error", "body is not JSON")
        if path == "/v1/messages/count_tokens":
            rough = (len(render_system(req)) + len(render_transcript(req.get("messages")))) // 4
            return self._json(200, {"input_tokens": rough})
        if path != "/v1/messages":
            return self._error(404, "not_found_error", f"no route {path}")

        t0, box = time.time(), {}

        def work():
            try:
                box["msg"] = to_message(call_claude(req))
            except CliError as e:
                box["err"] = str(e)
            except Exception as e:           # never leave the client hanging on a shim bug
                box["err"] = f"shim error: {e!r}"

        if not req.get("stream"):
            work()
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            th = threading.Thread(target=work, daemon=True)
            th.start()
            try:
                while th.is_alive():
                    th.join(PING_EVERY)
                    if th.is_alive():
                        self._event("ping", {"type": "ping"})
            except OSError:
                return                       # client went away

        msg, kind = box.get("msg"), "error"
        if msg:
            n = sum(b["type"] == "tool_use" for b in msg["content"])
            kind = f"tool_use x{n}" if n else "text"
        print(f"[shim] {req.get('model')} msgs={len(req.get('messages') or [])} "
              f"tools={len(req.get('tools') or [])} stream={bool(req.get('stream'))} "
              f"-> {kind} in {time.time() - t0:.1f}s" + (f"  {box['err'][:200]}" if not msg else ""),
              file=sys.stderr, flush=True)

        try:
            if req.get("stream"):
                if msg:
                    for name, data in sse_events(msg):
                        self._event(name, data)
                else:
                    self._event("error", {"type": "error",
                                          "error": {"type": "api_error", "message": box["err"]}})
                self.close_connection = True
            elif msg:
                self._json(200, msg)
            else:
                self._error(500, "api_error", box["err"])
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=int(os.environ.get("SHIM_PORT", "8765")))
    args = ap.parse_args()
    if not shutil.which("claude"):
        sys.exit("claude CLI not found on PATH")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)   # loopback only, by design
    print(f"[shim] http://127.0.0.1:{args.port}/v1/messages -> claude -p  (cwd {shim_cwd()})",
          file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
