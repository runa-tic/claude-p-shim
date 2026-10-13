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
`SHIM_MAX_PARALLEL` (4), `SHIM_API_KEY`, `SHIM_HOSTS`.

## The key

The endpoint spends whatever login `claude` has, so every `/v1` request must present the shim's
key as `x-api-key` (what the SDKs send from `ANTHROPIC_API_KEY`) or `Authorization: Bearer`, and
must name a loopback `Host`. The key is `SHIM_API_KEY` if set; otherwise one is generated on first
run into `~/.claude-p-shim-key` (mode 600) and reused.

```
export ANTHROPIC_BASE_URL=http://127.0.0.1:8765
export ANTHROPIC_API_KEY=$(cat ~/.claude-p-shim-key)
```

Without the key a request gets 401; with a foreign `Host` (a rebound DNS name) it gets 403. A web
page cannot send either header cross-origin without a preflight, and `OPTIONS` is not served.

A client that reaches the shim through a forwarder sends the forwarder's name as `Host`: a guest
under QEMU's user-mode network sends `10.0.2.2`. Allow such values, exactly as sent and
comma-separated, with `SHIM_HOSTS=10.0.2.2,10.0.2.2:8765`. The key is still required.

## How a request is translated

1. `system` plus the tool schemas, rendered as text under "Actions", go to `--system-prompt-file`.
2. `messages` are flattened into a transcript on stdin: `[<tag> user]`, `[<tag> assistant]`,
   `[<tag> assistant tool_call id=… name=…]`, `[<tag> tool_result id=…]`. The tag is random per
   request, so a tool result that contains `[user]` or `</conversation>` cannot pass for a turn.
   A trailing text-only assistant message is a prefill: it is shown as the start of the reply and
   only the continuation comes back.
3. `claude -p --tools "" --strict-mcp-config --setting-sources "" --no-session-persistence
   --output-format json --json-schema {text, tool_calls[]}` runs in an empty private directory.
   The empty `--setting-sources` keeps the user's settings, plugins and their hooks out; without
   it a call here loaded two user plugins and fired a `SessionStart` hook (CLI 2.1.296).
4. `structured_output.tool_calls` become `tool_use` blocks with invented ids; `stop_reason` is
   `tool_use` or `end_turn`; usage is copied from the CLI's report.

A reply that breaks the tool contract is a 500 `api_error`, not a quiet `end_turn`: a call to a
tool the request did not define, no call under `tool_choice` `any` or `tool`, or a call under
`none`. The shim does not retry; the SDKs retry a 500 themselves. A malformed body is a 400. When
the client disconnects, the running `claude -p` is killed.

## What this is not

It is a translation layer, not the API:

- The model chooses tools from schemas pasted as text, not through its native tool interface.
  Its habit of calling a named tool directly is strong: with a soft prompt it called
  `search_files` natively, got "No such tool available" three times and gave up. The prompt now
  says that `StructuredOutput` is the only callable function. Verified 3/3 on the test loop with
  Haiku 4.5, non-streaming (2026-10-13, CLI 2.1.296); not tested beyond that. The `--sdk`
  streaming path of the test has not been run against a real login.
- History is flattened; images and thinking blocks are dropped.
- Every request starts a new CLI process: about 7–8 s per turn here, and no prompt caching.
- Streaming is the finished reply replayed as SSE events, with pings while waiting.
- `max_tokens`, `temperature` and `stop_sequences` are ignored.
- `tool_choice` and prefill are instructions in the prompt, checked (tool_choice) or trusted
  (prefill) afterwards, not constraints on decoding.
- Token counts include the CLI's own overhead.
- `--bare` is deliberately not used: it reads only `ANTHROPIC_API_KEY`, never the CLI login.

Not affiliated with or endorsed by Anthropic. It spends the `claude` login of whoever runs it;
whether that use fits your plan's terms is yours to check.

Local, personal use: it binds to loopback only, asks for its key, and runs under whatever login
`claude` already has. Do not use it as the backend for a published measurement of another agent; its own errors
are indistinguishable from the agent's.

## Licence

MIT, see `LICENSE`.
