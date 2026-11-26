---
title: Agent Harness Demo
emoji: "\U0001F527"
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
---

# Agent Harness Demo

Combined demo: AI agent harness with plan-and-execute LangGraph agent, tool publisher API, faithfulness evaluation, and frontend — all in one deployable package.

## Architecture

```
Client  ->  FastAPI (/chat, /packages)  ->  LangGraph plan-and-execute  ->  Together AI (Qwen/Qwen3.5-9B)
                                                   |                                  ^
                                             API sub-app (tool execution)             |
                                             (Rust, isolated process)           Context Manager
                                                   |                          (TDA clustering)
                                             Tool functions                         |
                                        (built-in + installed packages)     Redis (sessions +
                                                   |                        archived clusters)
                                 ┌─────────────────┼─────────────────┐
                           Capability          Session            Package
                           enforcement         tunnel              signing
                        (OSAccess, env,     (mTLS, host        (Ed25519 verify
                         fn filtering)      allowlist,          at load time)
                                            audit log)
```

### Code to Agent

This harness implements the Code to Agent model: instead of calling tools on a remote MCP server, the harness loads self-describing, signed tool code into a local sandbox. The key ideas:

**Function kinds** — Every tool declares what kind of function it is:

| Kind | What it does | Exposed to model? | Where it runs | Shipped as code? |
|------|-------------|-------------------|---------------|-----------------|
| `gate` | Auth, permissions, consent | No | API | No |
| `fetch` | Reads data | Yes | API, called from sandbox | Client call only |
| `processor` | Transforms, filters, calculates | Yes | Sandbox (local) | Yes |
| `commit` | Writes and side effects | Yes, flagged | API, may need confirmation | Client call only |

Gates guard, fetches read, processors think, commits act. Only processors travel.

**Declared capabilities** (`AI/capabilities.py`) — Each tool declares what it needs at runtime:

| Capability | Target example | What it grants |
|-----------|---------------|---------------|
| `network` | `api.example.com` | Network access to that host |
| `filesystem` | `/data/*.json` | Read access to matching paths |
| `env` | `API_KEY` | Access to that environment variable |
| `subprocess` | `ffmpeg` | Not supported in Monty (flagged at inspection) |

Enforcement happens at three levels:
- **OSAccess per tool**: Each tool call gets a restricted virtual filesystem (only declared paths) and environment (only declared vars). Tools with no filesystem/env capabilities get an empty sandbox.
- **Filtered external_lookup**: Each tool can only call its own function in the sandbox — no access to other tools' code.
- **Startup inspection**: `inspect_tools()` runs at boot, checking that declared capabilities are valid (paths exist, env vars are set, processors don't claim network access). Warnings are logged.

**Session tunnel** (`AI/tunnel.py`) — A private, mutually authenticated channel between the sandbox and an external API that exists for one session, then closes.

- **Mutual TLS**: Client cert + server cert verification when configured
- **Short-lived keys**: Generated per session, discarded on close
- **Host allowlisting**: Only declared network capabilities are reachable through the tunnel
- **Request signing**: Every request carries the session ID and a sequence number
- **Audit logging**: Every request/response is logged with method, path, status, and size

Tools that need to call an external API declare a `tunnel` config in their definition:

```python
TOOL = ToolDefinition(
    name="weather_fetch",
    kind="fetch",
    tunnel=TunnelConfig(
        base_url="https://api.weather.example.com",
        allowed_hosts=["api.weather.example.com"],
        auth_header="Authorization",
        auth_env_var="WEATHER_API_KEY",  # resolved at runtime, never shipped
    ),
    ...
)
```

The harness opens the tunnel when the sandbox context starts and closes it when the session ends. Fetch and commit tools make requests through the tunnel; processors never touch the network.

**Package signing** (`AI/package_signing.py`) — Ed25519 signatures verify that tool code was published by the API owner before the harness loads it.

Signing (publisher side):
```python
from AI.package_signing import generate_keypair, sign_package

private_key, public_key = generate_keypair()
sig = sign_package(tool_definition, source_code, private_key, "weather-api.example.com")
# Ship sig alongside the package
```

Verification (harness side):
```python
from AI.package_signing import verify_package

trusted_keys = {"weather-api.example.com": public_key_pem}
valid = verify_package(tool_definition, source_code, sig, trusted_keys)
```

The canonical payload is `SHA-256(definition_json) || SHA-256(source_code)` — both the contract and the implementation are covered. `inspect_tools()` runs signature verification at startup when a tool carries a `signature` and `source_code`. Unsigned tools pass (signing is opt-in until the ecosystem matures). Invalid signatures produce a warning.

**Package lifecycle** (`AI/package_lifecycle.py`) — Install, inspect, and discard tool packages at runtime.

The `PackageRegistry` wraps the static tool registry and adds runtime management:

```
POST /packages/install   — install a package (definition + source + optional signature)
GET  /packages           — inspect all installed packages
GET  /packages/{name}    — inspect a single package
DELETE /packages/{name}  — discard a package (built-ins are protected)
```

Install flow: verify signature → compile source → register into live registry → available on next turn. Discard flow: remove from registry → disappears on next turn. All external packages are discarded when the session ends.

Built-in tools cannot be replaced or discarded. Duplicate installs are rejected (discard first). Invalid signatures block installation.

**Placement** — Tools declare `local` or `remote`. Processors default to local. The rule: read locally, commit remotely, move whichever is smaller.

**Inspection before execution** — The harness checks the tool's signature, reads its declared kind and capabilities, and rejects anything outside the grant. Commits with `requires_confirmation` need explicit user approval before executing.

### Core loop

1. User sends a message to `/chat`
2. The context manager checks whether conversation history exceeds the token budget
3. If over budget: TDA persistent homology clusters messages by topic, archives least-relevant clusters to Redis, rehydrates archived clusters that match the current query
4. The LLM streams a response, optionally requesting tool calls
5. Gate tools are never exposed to the model
6. Each tool call executes inside a Pydantic Monty sandbox (Rust-based, memory-limited, time-limited)
7. Capability enforcement: each tool gets a restricted virtual filesystem, environment, and function access
8. Fetch/commit tools route through the session tunnel (mTLS, host allowlisting, audit logging)
9. Commits with `requires_confirmation` check with the user before executing
10. The sandbox HMAC-signs the execution (tool name + input + output) using `itsdangerous`
11. An inline evaluator verifies the signature before the result enters the conversation
12. Each AI message carries its own evaluation inline — intermediate messages get tool_executions + causality + interstitial, the final response gets faithfulness
13. The full conversation is persisted to Redis
14. On session end: all tunnels close, all external packages are discarded

### Package installation (runtime)

1. API owner signs the tool package (definition + source) with their Ed25519 private key
2. Package arrives at `POST /packages/install` with the definition, source code, and signature
3. Harness verifies the signature against trusted keys (or trust-on-first-use)
4. Source code is compiled and the callable is extracted
5. Declared capabilities are inspected (paths exist, env vars set, no illegal claims)
6. Tool is registered into the live `PackageRegistry` and available on the next turn
7. On session end or `DELETE /packages/{name}`: tool is removed from the registry

### Verification pipeline

Four independent checks protect the tool-calling chain:

**HMAC execution verification** (`AI/evaluator.py`, `AI/signing.py`)
- At execution time, the sandbox signs every tool call with a server-side secret
- Before the LLM sees a tool result, the evaluator verifies the signature
- Catches: fabricated tool results, tampered outputs, expired executions (5 min TTL)

**Semantic faithfulness checking** (`AI/faithfulness.py`)
- After the final AI response, BERT sentence embeddings (`all-MiniLM-L6-v2`) compute cosine similarity between each tool output and the AI's text
- Per-tool extractors provide precision fallback (numeric matching for calculator, temporal matching for time)
- Catches: AI ignoring tool results, contradicting values, hallucinating over outputs
- Configurable: annotate-only (default) or enforce with re-prompting

**Interstitial faithfulness** (inline on intermediate AI messages)
- Between tool-calling rounds, verifies the model's reasoning text reflects the previous round's tool outputs
- Two checks: semantic similarity and value carryforward (did key tokens transfer)
- Catches: model misinterpreting intermediate results, passing wrong values between chained calls

**Causal ordering verification** (`AI/causality.py`)
- Two layers: static dependency DAG + runtime data-flow tracing
- Static: verifies tools were called in the correct dependency order as declared in `depends_on`
- Data-flow: traces each tool's input values back to preceding tool outputs or the user message using word-boundary-aware matching
- Catches: model skipping prerequisite steps, ungrounded inputs, broken chains

### TDA context management (`AI/context.py`, `AI/embeddings.py`, `AI/tokenizer.py`)

Manages the LLM's context window using Topological Data Analysis. Messages are embedded with BERT and clustered via Vietoris-Rips persistent homology (gudhi). The persistence diagram's H0 features (connected components) represent topic clusters — the epsilon threshold is chosen adaptively from the largest gap in H0 death times.

**Clustering**: Each turn, all active messages are embedded and fed into a Vietoris-Rips filtration. Connected components at the chosen epsilon form topic clusters. Cluster membership is recomputed every turn.

**Scoring**: Each cluster's centroid is compared to the current query via cosine similarity.

**Budget packing**: Greedy knapsack packs highest-scoring clusters into the token budget (default 32K, configurable via `CONTEXT_MAX_TOKENS`).

**Archival**: Overflow clusters are archived to Redis as frozen snapshots. Key format: `context:clusters:{session_id}`.

**Rehydration**: Before clustering, archived cluster centroids are compared to the current query. If similarity exceeds the threshold (0.5), those clusters rejoin the active set.

## Project structure

```
AI/
  agent.py              LLM config, tool-calling loop, stream_reply()
  context.py            TDA context manager — clustering, scoring, archival, rehydration
  embeddings.py         Shared BERT embedding singleton (all-MiniLM-L6-v2)
  tokenizer.py          Qwen tokenizer singleton for accurate token counting
  capabilities.py       Capability enforcement — per-tool OSAccess, filtered external_lookup, inspection
  package_signing.py    Ed25519 package signing — sign/verify tool definitions + source code
  package_lifecycle.py  Runtime install/inspect/discard for tool packages
  tunnel.py             Session tunnel — mTLS httpx client, host allowlisting, audit logging
  sandbox.py            Pydantic Monty pool, session management, LangChain tool wrappers
  signing.py            HMAC signing/verification via itsdangerous
  evaluator.py          Inline HMAC verification (deterministic, no LLM)
  faithfulness.py       BERT semantic similarity + per-tool extractor checks
  causality.py          Two-layer causal evaluation (static DAG + data-flow tracing)
  redis_store.py        Redis connection lifecycle and key/value operations
  sessions.py           LangChain <-> Pydantic model conversion, Redis load/save
  models/
    meta.py             Base model with id + created_at/updated_at timestamps
    message.py          Typed message hierarchy, ToolExecution, FaithfulnessResult, CausalResult, InterstitialCheck
    context.py          ArchivedCluster model for persisted cluster snapshots
    session.py          Session model wrapping a message list
  tools/
    definition.py       ToolDefinition model — Code-to-Agent extension of MCP tool schema
    __init__.py         Tool registry (ALL_TOOLS, TOOLS_BY_NAME, list_tools(), compat aliases)
    calculator.py       Arithmetic expression evaluator (processor)
    get_timezone_offset.py  UTC offset lookup by city/IANA timezone (fetch)
    get_current_time.py     Current time with timezone offset (fetch, depends on get_timezone_offset)
    unit_converter.py   Convert between common unit pairs (processor)
    word_counter.py     Word/character/sentence counting (processor)
    dice_roller.py      Standard dice notation roller (processor)
    string_tools.py     Hash computation and base64 encoding (processor)
main.py                 FastAPI transport layer with SSE streaming and /test endpoint
```

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai) running locally with `qwen3.5:4b` pulled
- Redis running on localhost:6379

### Install and run

```bash
# Install dependencies
uv sync

# Pull the model
ollama pull qwen3.5:4b

# Start Redis (Docker or Homebrew)
docker run -d -p 6379:6379 --name redis redis
# or: brew services start redis

# Start the server
uv run uvicorn main:app --reload --port 8000
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `TOOL_HMAC_SECRET` | Random (per-process) | HMAC signing key. Set in production for consistency across restarts |
| `CONTEXT_MAX_TOKENS` | `32768` | Token budget for context window management |
| `TOKENIZER_MODEL` | `Qwen/Qwen3-4B` | HuggingFace model ID for the tokenizer |

## API

### POST /chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 17.5% of 2340?"}'
```

Response is an SSE stream:

```
data: {"session_id": "a1b2c3..."}

data: {"delta": "17.5"}
data: {"delta": "% of"}
data: {"delta": " 2,340 is "}
data: {"delta": "409.5"}

data: "[DONE]"
```

### Request body

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | User message |
| `system` | string (optional) | Override the system prompt |

### GET /tools

Returns all tool definitions in Code-to-Agent format (gates excluded).

```bash
curl http://localhost:8000/tools | python -m json.tool
```

### POST /packages/install

Install a tool package at runtime. The request carries the definition, source code, and optional Ed25519 signature:

```bash
curl -X POST http://localhost:8000/packages/install \
  -H "Content-Type: application/json" \
  -d '{
    "definition": {
      "name": "weather_fetch",
      "description": "Get current weather for a city",
      "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
      "kind": "fetch",
      "placement": "remote",
      "capabilities": [{"kind": "network", "target": "api.weather.example.com"}]
    },
    "source_code": "def weather_fetch(city: str) -> str:\n    return f\"Sunny in {city}\"\n",
    "publisher": "weather-api.example.com",
    "signature": {
      "publisher": "weather-api.example.com",
      "public_key": "-----BEGIN PUBLIC KEY-----\n...",
      "signature": "base64-encoded-ed25519-signature",
      "payload_hash": "sha256-of-canonical-payload"
    }
  }'
```

### GET /packages

List all installed packages with their status, publisher, and verification info.

### GET /packages/{name}

Inspect a single installed package.

### DELETE /packages/{name}

Discard an installed package (built-in tools are protected).

### GET /test

Runs the test suite and streams pytest output as SSE events.

```bash
curl -N http://localhost:8000/test
```

## Tools

### Built-in tools

| Tool | Kind | Description | Dependencies |
|------|------|-------------|-------------|
| `get_timezone_offset` | fetch | UTC offset lookup by city name or IANA timezone | None |
| `get_current_time` | fetch | Current date/time for a UTC offset | `get_timezone_offset` |
| `calculator` | processor | Arithmetic expression evaluator | None |
| `unit_converter` | processor | Convert between units (km/miles, kg/lbs, celsius/fahrenheit, etc.) | None |
| `word_counter` | processor | Count words, characters, and sentences in text | None |
| `dice_roller` | processor | Roll dice using standard notation (2d6, 1d20+3) | None |
| `hash_string` | processor | Compute cryptographic hashes (md5, sha1, sha256, sha512) | None |
| `base64_encode` | processor | Base64-encode a string | None |

### Adding a new tool

1. Create `AI/tools/your_tool.py`:

```python
from pydantic import BaseModel, Field
from .definition import ToolDefinition

class YourToolInput(BaseModel):
    arg: str = Field(description="What this argument is for")

def your_tool(arg: str) -> str:
    """What the tool does — this is the description the model sees."""
    return f"result: {arg}"

def _extractor(output: str):
    return [(output, "exact")]

TOOL = ToolDefinition(
    name="your_tool",
    description="What the tool does.",
    inputSchema=YourToolInput.model_json_schema(),
    kind="processor",        # gate | fetch | processor | commit
    placement="local",       # local | remote
    capabilities=[],         # declared runtime needs
    requires_confirmation=False,  # True for dangerous commits
    fn=your_tool,
    extractor=_extractor,
    depends_on=[],           # tools that must be called first
)
```

2. Import in `AI/tools/__init__.py` and add to `ALL_TOOLS`.

The tool automatically gets: LangChain integration, sandbox execution, HMAC signing, faithfulness checking, causal ordering verification, and function-kind enforcement.

### Tool definition format (Code to Agent)

Each tool definition extends MCP's tool schema with:

```json
{
  "name": "get_current_time",
  "description": "Return the current date and time as ISO 8601...",
  "inputSchema": { "properties": { "tz_offset_hours": { "type": "integer" } } },
  "kind": "fetch",
  "placement": "local",
  "capabilities": [],
  "requires_confirmation": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Tool name |
| `description` | string | What it does — the model reads this |
| `inputSchema` | JSON Schema | Parameter types and descriptions |
| `kind` | `gate\|fetch\|processor\|commit` | Function kind (gates are never exposed to the model) |
| `placement` | `local\|remote` | Where the code runs |
| `capabilities` | array | Declared runtime needs (network, filesystem, env, subprocess) |
| `tunnel` | object (optional) | Session tunnel config: `base_url`, `allowed_hosts`, `auth_header`, `auth_env_var`, TLS cert paths |
| `requires_confirmation` | boolean | Whether the harness must get user approval before executing |
| `signature` | PackageSignature (optional) | Ed25519 signature from the publisher, verified at load time |
| `source_code` | string (optional) | Tool source code, used for signature verification |

### Evaluation results on AI messages

Each AI message carries its own evaluation inline:

**Intermediate messages** (tool-calling rounds):
```json
{
  "role": "ai",
  "content": "I'll look up the timezone offsets...",
  "tool_calls": [{"name": "get_timezone_offset", "args": {"location": "Tokyo"}}],
  "tool_executions": [
    {"tool_name": "get_timezone_offset", "tool_call_id": "call_abc",
     "tool_input": {"location": "Tokyo"}, "tool_output": "Asia/Tokyo UTC+9.0",
     "signature": "eyJ0..."}
  ],
  "causality": {"valid": true},
  "interstitial": {"semantic_score": 0.71, "faithful": true}
}
```

**Final response**:
```json
{
  "role": "ai",
  "content": "The current time in Tokyo is 2026-11-13T14:03:32+09:00.",
  "causality": {"valid": true, "data_flow": [
    {"from_tool": "get_timezone_offset", "to_tool": "get_current_time", "values": ["9"]}
  ]},
  "faithfulness": {"semantic_score": 0.60, "contained": true, "faithful": true}
}
```

## Session persistence

Sessions are stored in Redis with a 1-hour TTL. Each session contains the full typed message history including tool executions with HMAC signatures, inline faithfulness and causality results, and interstitial checks between rounds.

Session key format: `chat:session:{uuid}`

Archived context clusters are stored under `context:clusters:{uuid}` with the same TTL.

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | HTTP transport |
| `langchain-core`, `langchain-openai` | LLM interaction and tool-calling |
| `pydantic-monty` | Rust-based sandboxed Python interpreter |
| `redis[hiredis]` | Session persistence |
| `httpx` | Session tunnel — async HTTP client with mTLS support |
| `cryptography` | Ed25519 key generation, package signing and verification |
| `itsdangerous` | HMAC-signed timed tokens for tool verification |
| `sentence-transformers` | BERT embeddings for faithfulness and context clustering |
| `torch` | Tensor operations for embeddings |
| `gudhi` | Vietoris-Rips persistent homology for TDA context clustering |
| `transformers` | Qwen tokenizer for accurate token counting |
| `scipy` | Connected components for cluster extraction |
| `python-dateutil` | Temporal parsing for time tool extractor |
