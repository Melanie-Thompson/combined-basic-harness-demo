"""FastAPI transport layer.  uvicorn main:app --reload --port 8000"""

import asyncio
import json
import logging
import os
import random
import string
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import BaseMessage
from pathlib import Path
from pydantic import BaseModel

from AI import redis_store, sessions
from AI.benchmark import run_benchmark, BenchmarkReport, TestResult, FAITHFULNESS_THRESHOLD
from AI.comparison import run_inline_baseline, run_all_baselines
from AI.embeddings import semantic_similarity
from AI.faithfulness import check_faithfulness
from AI.models.message import ToolExecutionRecord
from AI.graph import stream_reply
from AI.package_lifecycle import PackageManifest, PackageRegistry
from AI.package_signing import PackageSignature
from AI.sandbox import close_sandbox, inspect_tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("agent")

_session_id: str = ""
_registry: PackageRegistry | None = None

# ── In-memory log ring buffer (for /logs/stream) ──────────────────────────

_log_buffer: deque[dict] = deque(maxlen=500)
_log_subscribers: list[asyncio.Queue] = []


class _SSELogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        _log_buffer.append(entry)
        for q in _log_subscribers:
            try:
                q.put_nowait(entry)
            except asyncio.QueueFull:
                pass


_sse_handler = _SSELogHandler()
_sse_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(_sse_handler)

# ── Dev auth state ─────────────────────────────────────────────────────────

_otp_codes: dict[str, str] = {}
_user_sessions: dict[str, str] = {}  # cookie -> user_id
_SESSION_COOKIE = "harness_session"

# ── Pending tool confirmations ─────────────────────────────────────────────

_pending_confirmations: dict[str, asyncio.Future] = {}


def _load_packages_from_registry(harness_registry: PackageRegistry) -> int:
    """Load packages from the bundled API registry into the harness registry."""
    from api.registry import PackageRegistry as APIRegistry
    from api.signing import KeyStore

    key_store = KeyStore()
    api_registry = APIRegistry(key_store=key_store)
    count = api_registry.load_all()
    if count == 0:
        log.info("no packages found in bundled API registry")
        return 0

    installed = 0
    for pkg_info in api_registry.list_packages():
        name = pkg_info["definition"]["name"]
        manifest_data = api_registry.get_manifest(name)
        if manifest_data is None:
            continue

        sig = None
        if manifest_data.get("signature"):
            sig = PackageSignature(**manifest_data["signature"])

        manifest = PackageManifest(
            definition=manifest_data["definition"],
            source_code=manifest_data["source_code"],
            signature=sig,
            publisher=manifest_data.get("publisher", ""),
        )
        try:
            harness_registry.install(manifest)
            installed += 1
            log.info("installed package '%s' from bundled registry", name)
        except ValueError as e:
            log.warning("failed to install '%s': %s", name, e)

    return installed


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _session_id, _registry
    await redis_store.connect()
    _registry = PackageRegistry()
    installed = _load_packages_from_registry(_registry)
    inspect_tools(registry=_registry)
    _session_id = sessions.new_session_id()
    log.info("session %s initialised (%d tools from API)", _session_id, installed)
    yield
    if _registry:
        discarded = _registry.discard_all_external()
        if discarded:
            log.info("session end: discarded %d external packages: %s",
                     len(discarded), discarded)
    await redis_store.close()
    await close_sandbox()


app = FastAPI(lifespan=lifespan)

_allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://localhost:5500,http://127.0.0.1:3000,http://127.0.0.1:5500").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount bundled API sub-app ─────────────────────────────────────────────
try:
    from main_api import app as api_app
    app.mount("/api", api_app)
    log.info("mounted bundled API at /api")
except ImportError:
    log.info("no bundled API found — running standalone")

# ── Serve frontend static files ───────────────────────────────────────────
_frontend_dir = Path(__file__).parent / "frontend"
if _frontend_dir.is_dir():
    from fastapi.responses import FileResponse

    @app.get("/")
    async def serve_index():
        return FileResponse(_frontend_dir / "index.html")

    @app.get("/style.css")
    async def serve_css():
        return FileResponse(_frontend_dir / "style.css", media_type="text/css")

    @app.get("/app.js")
    async def serve_js():
        return FileResponse(_frontend_dir / "app.js", media_type="application/javascript")


def sse(payload: dict | str, event: str | None = None) -> str:
    body = payload if isinstance(payload, str) else json.dumps(payload)
    if event:
        return f"event: {event}\ndata: {body}\n\n"
    return f"data: {body}\n\n"


def _get_user(request: Request) -> str | None:
    cookie = request.cookies.get(_SESSION_COOKIE)
    if cookie and cookie in _user_sessions:
        return _user_sessions[cookie]
    return None


# ── Auth endpoints ─────────────────────────────────────────────────────────

class OTPSendRequest(BaseModel):
    email: str


class OTPVerifyRequest(BaseModel):
    email: str
    code: str


@app.post("/auth/otp/send")
async def otp_send(req: OTPSendRequest):
    code = "".join(random.choices(string.digits, k=6))
    _otp_codes[req.email] = code
    log.info("otp.dev_code email=%s code=******", req.email)
    # Emit as a structured log event so the frontend can auto-fill
    for q in _log_subscribers:
        try:
            q.put_nowait({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "level": "INFO",
                "logger": "auth",
                "event": "otp.dev_code",
                "email": req.email,
                "code": code,
            })
        except asyncio.QueueFull:
            pass
    return {"sent": True}


@app.post("/auth/otp/verify")
async def otp_verify(req: OTPVerifyRequest, request: Request):
    expected = _otp_codes.get(req.email)
    if expected is None or req.code != expected:
        return JSONResponse({"detail": "Invalid or expired code"}, status_code=401)

    del _otp_codes[req.email]
    user_id = req.email
    session_token = uuid.uuid4().hex
    _user_sessions[session_token] = user_id

    response = JSONResponse({"user_id": user_id})
    response.set_cookie(
        _SESSION_COOKIE, session_token,
        httponly=True, samesite="lax", max_age=3600,
    )
    log.info("user authenticated: %s", user_id)
    return response


@app.post("/sessions/logout")
async def logout(request: Request):
    cookie = request.cookies.get(_SESSION_COOKIE)
    if cookie and cookie in _user_sessions:
        del _user_sessions[cookie]
    response = JSONResponse({"ok": True})
    response.delete_cookie(_SESSION_COOKIE)
    return response


# ── Agent chat (SSE with named events) ─────────────────────────────────────

class AgentChatRequest(BaseModel):
    prompt: str
    conversation_id: str | None = None


@app.post("/agent/chat")
async def agent_chat(req: AgentChatRequest, request: Request):
    conversation_id = req.conversation_id or _session_id

    history = await sessions.load(conversation_id)
    turn_messages: list[BaseMessage] = []

    async def events():
        yield sse({"conversation_id": conversation_id}, event="conversation_started")
        yield sse({"iteration": 1}, event="thinking")

        current_text = ""
        actual_tool: str | None = None
        actual_blocked = False
        tool_output = ""
        tool_args: dict | None = None
        start_ms = time.monotonic()

        try:
            async for evt in stream_reply(
                req.prompt,
                history=history,
                turn_messages=turn_messages,
                session_id=conversation_id,
                registry=_registry,
            ):
                evt_type = evt.get("type", "")
                if evt_type == "token":
                    yield sse({"token": evt["token"]}, event="token")
                    current_text += evt["token"]
                elif evt_type == "tool_call":
                    actual_tool = evt["tool"]
                    tool_args = evt.get("args")
                    yield sse({"tool": evt["tool"], "args": evt["args"]}, event="tool_call")
                elif evt_type == "tool_result":
                    tool_output += evt.get("result", "") + "\n"
                    yield sse({"tool": evt["tool"], "result": evt["result"]}, event="tool_result")
                elif evt_type == "tool_blocked":
                    actual_blocked = True
                    current_text = evt.get("reason", "")
                    yield sse({"reason": evt["reason"], "stage": "relevance", "tool": evt.get("tool", ""), "category": "irrelevant"}, event="blocked")

        except Exception as exc:
            log.exception("agent chat stream failed")
            yield sse({"error": str(exc)}, event="error")

        latency_ms = (time.monotonic() - start_ms) * 1000

        if turn_messages:
            await sessions.save(conversation_id, history + turn_messages)

        yield sse({"text": current_text}, event="response")

        try:
            result = _score_query(
                prompt=req.prompt,
                actual_tool=actual_tool,
                actual_blocked=actual_blocked,
                response_text=current_text,
                tool_output=tool_output,
                tool_args=tool_args,
                latency_ms=latency_ms,
                baselines=[],
            )
            await _persist_eval()
            yield sse(result.to_dict(), event="eval")
        except Exception as exc:
            log.exception("eval scoring failed")

        async def _run_baseline_bg(prompt, tool_out, result_obj):
            try:
                from AI.graph import MODEL as AGENT_MODEL
                baseline = await run_inline_baseline(prompt, AGENT_MODEL, registry=_registry)
                bd = baseline.to_dict()
                if tool_out.strip() and bd["response_text"].strip():
                    bd["faithfulness_score"] = round(
                        semantic_similarity(tool_out.strip(), bd["response_text"].strip()), 3
                    )
                else:
                    bd["faithfulness_score"] = None
                result_obj.baselines = [bd]
                await _persist_eval()
            except Exception as exc:
                log.warning("baseline comparison failed: %s", exc)

        asyncio.create_task(_run_baseline_bg(req.prompt, tool_output, result))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Gated tool confirmation ───────────────────────────────────────────────

class ConfirmRequest(BaseModel):
    approved: bool


@app.post("/agent/confirm/{pending_id}")
async def confirm_tool(pending_id: str, req: ConfirmRequest):
    future = _pending_confirmations.get(pending_id)
    if future is None:
        return {"error": f"no pending action '{pending_id}'"}
    future.set_result(req.approved)
    return {"confirmed": req.approved, "pending_id": pending_id}


# ── Log streaming ─────────────────────────────────────────────────────────

@app.get("/logs/stream")
async def log_stream(request: Request, catchup: int = 0):
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    _log_subscribers.append(queue)

    async def events():
        try:
            if catchup:
                for entry in list(_log_buffer)[-catchup:]:
                    yield sse(entry, event="log")

            while True:
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield sse(entry, event="log")
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"

                if await request.is_disconnected():
                    break
        finally:
            _log_subscribers.remove(queue)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Redis introspection ───────────────────────────────────────────────────

@app.get("/redis/keys")
async def redis_keys():
    if redis_store._redis is not None:
        try:
            r = redis_store._redis
            keys = await r.keys(f"*{_session_id}*")
            entries = []
            for key in sorted(keys):
                key_type = await r.type(key)
                ttl = await r.ttl(key)
                value: Any = None
                if key_type == "string":
                    raw = await r.get(key)
                    try:
                        value = json.loads(raw) if raw else None
                    except (json.JSONDecodeError, TypeError):
                        value = raw

                db = "sessions" if "session" in key else "conversations"
                entries.append({
                    "db": db,
                    "key": key,
                    "type": key_type,
                    "ttl": ttl if ttl > 0 else None,
                    "value": value,
                })
            return entries
        except Exception as e:
            return JSONResponse([{"db": "error", "key": str(e), "type": "error", "ttl": None, "value": None}])

    entries = []
    for key in sorted(redis_store._memory.keys()):
        raw = redis_store._memory[key]
        try:
            value = json.loads(raw) if raw else None
        except (json.JSONDecodeError, TypeError):
            value = raw
        db = "sessions" if "session" in key else "conversations"
        entries.append({
            "db": db,
            "key": key,
            "type": "string",
            "ttl": None,
            "value": value,
        })
    for key in sorted(redis_store._memory_lists.keys()):
        entries.append({
            "db": "conversations",
            "key": key,
            "type": "list",
            "ttl": None,
            "value": redis_store._memory_lists[key],
        })
    return entries


# ── TDA topology stream ───────────────────────────────────────────────────

@app.get("/tda/stream")
async def tda_stream(request: Request, catchup: bool = False):
    async def events():
        try:
            while True:
                try:
                    snapshot = await _build_tda_snapshot()
                    yield sse(snapshot, event="topology")
                except Exception as e:
                    yield sse({"error": str(e)}, event="error")

                await asyncio.sleep(5.0)

                if await request.is_disconnected():
                    break
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _build_tda_snapshot() -> dict:
    """Build a TDA topology snapshot for the frontend visualization."""
    _empty = {"betti": [0, 0], "h0_features": [], "h1_features": [], "nodes": [], "edges": []}
    try:
        from AI.context import ContextManager
        from AI.embeddings import embed
        import gudhi

        history = await sessions.load(_session_id)
        if not history or len(history) < 2:
            return _empty

        texts = []
        roles = []
        previews = []
        for msg in history:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content.strip():
                texts.append(content)
                roles.append(msg.type)
                previews.append(content[:120])

        if len(texts) < 2:
            return _empty

        embeddings = embed(texts)

        rips = gudhi.RipsComplex(points=embeddings, max_edge_length=2.0)
        st = rips.create_simplex_tree(max_dimension=2)
        dgm = st.persistence()

        h0_features = []
        h1_features = []
        for dim, (birth, death) in dgm:
            feat = {"birth": float(birth), "death": float(death) if death != float("inf") else None}
            if dim == 0:
                h0_features.append(feat)
            elif dim == 1:
                h1_features.append(feat)

        b0 = sum(1 for d, _ in dgm if d == 0)
        b1 = sum(1 for d, _ in dgm if d == 1)

        ctx = ContextManager(_session_id)
        clusters = ctx._cluster(embeddings)

        nodes = []
        for i, members in enumerate(clusters):
            persistence = 0.0
            for m in members:
                for dim, (birth, death) in dgm:
                    if dim == 0 and death != float("inf"):
                        persistence = max(persistence, death - birth)

            label = texts[members[0]][:60] if members else ""
            messages = [
                {"idx": m, "role": roles[m], "text": previews[m]}
                for m in members
            ]
            nodes.append({
                "cluster_id": i,
                "label": label,
                "size": len(members),
                "persistence": round(persistence, 3),
                "messages": messages,
            })

        return {
            "betti": [b0, b1],
            "h0_features": h0_features,
            "h1_features": h1_features,
            "nodes": nodes,
            "edges": [],
        }
    except Exception as e:
        log.warning("TDA snapshot failed: %s", e)
        return _empty


# ── Original /chat endpoint (kept for backwards compat) ───────────────────

class ChatRequest(BaseModel):
    message: str
    system: str | None = None


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    history = await sessions.load(_session_id)
    turn_messages: list[BaseMessage] = []
    deltas: list[str] = []

    kwargs: dict = {"history": history, "turn_messages": turn_messages, "session_id": _session_id, "registry": _registry}
    if req.system is not None:
        kwargs["system"] = req.system

    error: Exception | None = None
    try:
        async for evt in stream_reply(req.message, **kwargs):
            if evt.get("type") == "token":
                deltas.append(evt["token"])
    except Exception as exc:
        log.exception("stream failed")
        error = exc

    if turn_messages:
        await sessions.save(_session_id, history + turn_messages)

    async def events():
        yield sse({"session_id": _session_id})
        if error:
            yield sse({"error": f"{type(error).__name__}: {error}"})
        else:
            for d in deltas:
                yield sse({"delta": d})
        yield sse("[DONE]")

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Tools ──────────────────────────────────────────────────────────────────

@app.get("/tools")
async def tools():
    """Return tool definitions from the registry (all from API, gates excluded)."""
    result = []
    if _registry:
        for defn in _registry.all_tools:
            if defn.kind == "gate":
                continue
            entry = {
                "name": defn.name,
                "description": defn.description,
                "inputSchema": defn.inputSchema,
                "kind": defn.kind,
                "placement": defn.placement,
                "capabilities": [c.model_dump() for c in defn.capabilities],
                "requires_confirmation": defn.requires_confirmation,
            }
            if defn.tunnel:
                entry["tunnel"] = {
                    "base_url": defn.tunnel.base_url,
                    "allowed_hosts": defn.tunnel.allowed_hosts,
                }
            result.append(entry)
    return {"tools": result}


# ── Package management ─────────────────────────────────────────────────────

class InstallRequest(BaseModel):
    definition: dict
    source_code: str
    publisher: str = ""
    signature: dict | None = None


@app.post("/packages/install")
async def install_package(req: InstallRequest):
    if _registry is None:
        return {"error": "registry not initialised"}
    sig = None
    if req.signature:
        sig = PackageSignature(**req.signature)
    manifest = PackageManifest(
        definition=req.definition,
        source_code=req.source_code,
        signature=sig,
        publisher=req.publisher,
    )
    try:
        record = _registry.install(manifest)
    except ValueError as e:
        return {"error": str(e)}
    return {
        "installed": record.name,
        "publisher": record.publisher,
        "kind": record.kind,
        "signature_verified": record.signature_verified,
        "warnings": record.warnings,
        "total_tools": _registry.installed_count,
    }


@app.get("/packages")
async def list_packages():
    if _registry is None:
        return {"error": "registry not initialised"}
    return {"packages": _registry.inspect()}


@app.get("/packages/{name}")
async def inspect_package(name: str):
    if _registry is None:
        return {"error": "registry not initialised"}
    result = _registry.inspect(name)
    if not result:
        return {"error": f"package '{name}' not found"}
    return {"package": result[0]}


@app.delete("/packages/{name}")
async def discard_package(name: str):
    if _registry is None:
        return {"error": "registry not initialised"}
    try:
        removed = _registry.discard(name)
    except ValueError as e:
        return {"error": str(e)}
    if not removed:
        return {"error": f"package '{name}' not found"}
    return {"discarded": name, "total_tools": _registry.installed_count}


# ── Test runner ────────────────────────────────────────────────────────────

@app.get("/test")
async def run_tests() -> StreamingResponse:
    async def stream_pytest():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pytest", "tests/", "-v", "-s", "--tb=short", "--no-header",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield sse({"line": line.decode(errors="replace").rstrip()})

        code = await proc.wait()
        yield sse({"exit_code": code})
        yield sse("[DONE]")

    return StreamingResponse(
        stream_pytest(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Live eval (cumulative, scored on every query) ────────────────────────

_eval_report = BenchmarkReport()


def _score_query(
    prompt: str,
    actual_tool: str | None,
    actual_blocked: bool,
    response_text: str,
    tool_output: str,
    tool_args: dict | None,
    latency_ms: float,
    baselines: list[dict] | None = None,
) -> TestResult:
    """Score a single live query and append to the cumulative report."""
    has_output = bool(tool_output and tool_output.strip())
    has_response = bool(response_text and response_text.strip())

    faithfulness_score = 0.0
    is_faithful = False
    faithfulness_mismatches: list[str] = []

    if has_output:
        records = [
            ToolExecutionRecord(
                tool_call_id=f"live_{int(time.time() * 1000)}",
                tool_name=actual_tool or "unknown",
                tool_input=tool_args or {},
                tool_output=tool_output,
                signature="live_eval",
            )
        ]
        faith_map = check_faithfulness(response_text, records)
        faith_res = next(iter(faith_map.values()), None)
        if faith_res is not None:
            faithfulness_score = faith_res.semantic_score
            is_faithful = faith_res.faithful
            faithfulness_mismatches = faith_res.mismatches

    failure_reasons: list[str] = []
    if actual_blocked:
        pass
    else:
        if actual_tool and not has_output:
            failure_reasons.append("tool produced no output")
        if not has_response:
            failure_reasons.append("empty final response")
        elif has_output and not is_faithful:
            detail = ", ".join(faithfulness_mismatches) if faithfulness_mismatches else f"{faithfulness_score:.3f}"
            failure_reasons.append(f"unfaithful: {detail}")

    passed = len(failure_reasons) == 0
    actual_tools = [actual_tool] if actual_tool else []

    result = TestResult(
        case_id=f"live_{int(time.time() * 1000)}",
        prompt=prompt,
        expected_tool=actual_tool,
        expected_blocked=actual_blocked,
        actual_tool=actual_tool,
        actual_tools=actual_tools,
        actual_blocked=actual_blocked,
        response_text=response_text,
        tool_output=tool_output,
        tool_args=tool_args,
        latency_ms=latency_ms,
        passed=passed,
        failure_reasons=failure_reasons,
        faithfulness_score=faithfulness_score,
        faithful=is_faithful,
        faithfulness_mismatches=faithfulness_mismatches,
        baselines=baselines or [],
    )

    _eval_report.record_result(result, k=1)
    return result


async def _persist_eval():
    try:
        assert redis_store._redis is not None
        await redis_store._redis.set(
            f"eval:{_session_id}",
            json.dumps(_eval_report.to_dict()),
            ex=redis_store.SESSION_TTL,
        )
    except Exception:
        pass


@app.get("/eval/results")
async def eval_results():
    """Get cumulative eval results from live queries."""
    if _eval_report.total > 0:
        return _eval_report.to_dict()
    try:
        assert redis_store._redis is not None
        raw = await redis_store._redis.get(f"eval:{_session_id}")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return {"total": 0, "results": []}


@app.post("/eval/reset")
async def eval_reset():
    """Reset cumulative eval results."""
    global _eval_report
    _eval_report = BenchmarkReport(k_runs=1)
    try:
        assert redis_store._redis is not None
        await redis_store._redis.delete(f"eval:{_session_id}")
    except Exception:
        pass
    return {"reset": True}


@app.post("/session/new")
async def session_new():
    """Create a fresh session — resets eval accumulator."""
    global _session_id, _eval_report
    _session_id = sessions.new_session_id()
    _eval_report = BenchmarkReport(k_runs=1)
    log.info("new session %s", _session_id)
    return {"session_id": _session_id}


@app.post("/benchmark/run")
async def benchmark_run():
    """Run the synthetic benchmark suite (batch eval)."""
    report = await run_benchmark(
        stream_reply,
        registry=_registry,
        session_id=_session_id,
    )
    return report.to_dict()


# ── Context store ──────────────────────────────────────────────────────────

@app.get("/context/tools")
async def context_tool_results():
    """List all tool results in the Redis context store for the current session."""
    try:
        results = await redis_store.list_tool_results(_session_id)
        return {"session_id": _session_id, "results": results}
    except Exception as e:
        return {"error": str(e)}


@app.get("/context/tools/{call_id}")
async def context_tool_result(call_id: str):
    """Get a single tool result from the context store."""
    try:
        result = await redis_store.get_tool_result(_session_id, call_id)
        if result is None:
            return {"error": f"no result for call_id '{call_id}'"}
        return result
    except Exception as e:
        return {"error": str(e)}


@app.delete("/context/tools")
async def context_clear():
    """Clear tool context for the current session."""
    try:
        count = await redis_store.clear_tool_context(_session_id)
        return {"cleared": count, "session_id": _session_id}
    except Exception as e:
        return {"error": str(e)}
