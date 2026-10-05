# claude-p-shim

An Anthropic Messages API endpoint on localhost, answered by headless `claude -p`.

A client that speaks the Messages API (system, tools, messages in; text or `tool_use` out) is
pointed at `http://127.0.0.1:8765` instead of `api.anthropic.com`. The client keeps its own agent
loop and runs its own tools; the shim only translates each request into one tool-less `claude -p`
call and wraps the reply back into the API's shape, streaming or not.

```
python shim.py                 # serves http://127.0.0.1:8765/v1/messages
python test_loop.py            # a 3-turn agent loop: search_files -> read_file -> answer
```

Stdlib only. Environment: `SHIM_PORT`, `SHIM_MODEL` (force one model), `SHIM_TIMEOUT` (s, 300),
`SHIM_MAX_PARALLEL` (4).

## How a request is translated

1. `system` plus the tool schemas, rendered as text under "Actions", go to `--system-prompt-file`.
2. `messages` are flattened into a transcript on stdin: `[user]`, `[assistant]`,
   `[assistant tool_call id=… name=…]`, `[tool_result id=…]`.
3. `claude -p --tools "" --strict-mcp-config --no-session-persistence --output-format json
   --json-schema {text, tool_calls[]}` runs in an empty directory.
4. `structured_output.tool_calls` become `tool_use` blocks with invented ids; `stop_reason` is
   `tool_use` or `end_turn`; usage is copied from the CLI's report.

## What this is not

It is a translation layer, not the API:

- The model chooses tools from schemas pasted as text, not through its native tool interface.
  Its habit of calling a named tool directly is strong: with a soft prompt it called
  `search_files` natively, got "No such tool available" three times and gave up. The prompt now
  says that `StructuredOutput` is the only callable function. Verified 3/3 on the test loop with
  Haiku 4.5 (2026-10-04); not tested beyond that.
- History is flattened; images and thinking blocks are dropped.
- Every request starts a new CLI process: about 7–8 s per turn here, and no prompt caching.
- Streaming is the finished reply replayed as SSE events, with pings while waiting.
- `max_tokens`, `temperature` and `stop_sequences` are ignored.
- Token counts include the CLI's own overhead.
- `--bare` is deliberately not used: it reads only `ANTHROPIC_API_KEY`, never the CLI login.

Local, personal use: it binds to loopback only and runs under whatever login `claude` already
has. Do not use it as the backend for a published measurement of another agent; its own errors
are indistinguishable from the agent's.
