#!/usr/bin/env python3
"""End-to-end check: a minimal agent loop (the client owns the loop and the tools) against the shim.

  python shim.py &            # in another terminal
  python test_loop.py         # raw HTTP, non-streaming
  python test_loop.py --sdk   # the real `anthropic` SDK with streaming, if installed

Passes when the model calls search_files, then read_file, and answers with the planted fact.
"""
import json
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8765"
MODEL = "claude-haiku-4-5-20251001"
SECRET = "heliotrope-47"

TOOLS = [
    {"name": "search_files", "description": "Regex search inside files under the notes folder. Returns matching file names.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
    {"name": "read_file", "description": "Read one file from the notes folder by name.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
]


def fixture() -> Path:
    d = Path(tempfile.mkdtemp(prefix="shim-test-"))
    (d / "groceries.md").write_text("milk, eggs, rye bread", encoding="utf-8")
    (d / "deploy-notes.md").write_text(f"The staging deploy passphrase is {SECRET}. Rotate monthly.", encoding="utf-8")
    (d / "ideas.md").write_text("write a relay that turns a CLI into an API", encoding="utf-8")
    return d


def run_tool(root: Path, name: str, args: dict) -> str:
    if name == "search_files":
        rx = re.compile(args["pattern"], re.I)
        return "\n".join(p.name for p in sorted(root.iterdir()) if rx.search(p.read_text(encoding="utf-8"))) or "(no matches)"
    if name == "read_file":
        p = root / Path(args["path"]).name
        return p.read_text(encoding="utf-8") if p.exists() else f"no such file: {args['path']}"
    return f"unknown tool {name}"


def post(body: dict) -> dict:
    r = urllib.request.Request(BASE + "/v1/messages", json.dumps(body).encode(),
                               {"content-type": "application/json", "x-api-key": "shim"})
    with urllib.request.urlopen(r, timeout=600) as resp:
        return json.loads(resp.read())


def sdk_call(client, body: dict) -> dict:
    with client.messages.stream(**body) as s:
        return s.get_final_message().model_dump()


def main():
    use_sdk = "--sdk" in sys.argv
    client = None
    if use_sdk:
        import anthropic
        client = anthropic.Anthropic(base_url=BASE, api_key="shim")
    root = fixture()
    messages = [{"role": "user", "content": "What is the staging deploy passphrase? It is somewhere in my notes."}]
    seen = []
    for turn in range(1, 7):
        body = {"model": MODEL, "max_tokens": 1024, "tools": TOOLS, "messages": messages,
                "system": "You answer questions from the user's notes folder using the tools."}
        msg = sdk_call(client, body) if use_sdk else post(body)
        calls = [b for b in msg["content"] if b["type"] == "tool_use"]
        text = " ".join(b["text"] for b in msg["content"] if b["type"] == "text")
        print(f"turn {turn}: stop={msg['stop_reason']} calls={[(c['name'], c['input']) for c in calls]} text={text[:90]!r}")
        if msg["stop_reason"] != "tool_use":
            ok = SECRET in text and "search_files" in seen and "read_file" in seen
            print("PASS" if ok else f"FAIL (tools used: {seen})")
            sys.exit(0 if ok else 1)
        messages.append({"role": "assistant", "content": [
            {k: v for k, v in b.items() if k in ("type", "text", "id", "name", "input")} for b in msg["content"]]})
        results = []
        for c in calls:
            seen.append(c["name"])
            results.append({"type": "tool_result", "tool_use_id": c["id"], "content": run_tool(root, c["name"], c["input"])})
        messages.append({"role": "user", "content": results})
    print("FAIL (no final answer in 6 turns)")
    sys.exit(1)


if __name__ == "__main__":
    main()
