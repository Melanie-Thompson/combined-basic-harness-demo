"""Code to Agent API — bundled as a sub-app.

Mounted at /api by the main harness when present.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from api.causality import close_session as close_causality_session, get_validator
from api.effect_monitor import monitor_effects
from api.registry import PackageRegistry
from api.signing import KeyStore
from api.tunnel_auth import TunnelAuthMiddleware, get_tunnel_registry

log = logging.getLogger("api")

_key_store = KeyStore()
_registry = PackageRegistry(key_store=_key_store)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    count = _registry.load_all()
    log.info("loaded %d packages, %d publishers",
             count, len(_key_store.list_publishers()))
    yield
    tunnel_reg = get_tunnel_registry()
    for s in tunnel_reg.active_sessions:
        tunnel_reg.close_session(s["session_id"])


app = FastAPI(
    title="Code to Agent API",
    description="Publish signed tool packages for agent harnesses",
    lifespan=lifespan,
)
app.add_middleware(TunnelAuthMiddleware)


@app.get("/packages")
async def list_packages():
    return {"packages": _registry.list_packages()}


@app.get("/packages/{name}")
async def get_package(name: str):
    manifest = _registry.get_manifest(name)
    if manifest is None:
        return {"error": f"package '{name}' not found"}
    return manifest


class PublishRequest(BaseModel):
    definition: dict[str, Any]
    source_code: str
    publisher: str = ""


@app.post("/packages")
async def publish_package(req: PublishRequest):
    try:
        pkg = _registry.add_package(
            definition=req.definition,
            source_code=req.source_code,
            publisher=req.publisher,
        )
    except ValueError as e:
        return {"error": str(e)}
    return {
        "published": pkg.name,
        "publisher": pkg.publisher,
        "signed": pkg.signature is not None,
    }


@app.delete("/packages/{name}")
async def remove_package(name: str):
    if not _registry.remove_package(name):
        return {"error": f"package '{name}' not found"}
    return {"removed": name}


class ExecuteRequest(BaseModel):
    args: dict[str, Any] = {}
    session_id: str = "default"
    user_message: str = ""


@app.post("/packages/{name}/execute")
async def execute_tool(name: str, req: ExecuteRequest):
    pkg = _registry.get(name)
    if pkg is None:
        return {"error": f"package '{name}' not found"}

    validator = get_validator(req.session_id)
    causal = validator.check_before_execute(name, req.args, pkg.definition)
    if not causal.valid:
        return {
            "error": f"causality violation: {'; '.join(causal.violations)}",
            "tool": name,
            "causality": causal.to_dict(),
        }

    data_flow, flow_warnings = validator.trace_data_flow(
        name, req.args, req.user_message
    )

    try:
        namespace: dict[str, Any] = {}
        exec(compile(pkg.source_code, f"<package:{name}>", "exec"), namespace)  # noqa: S102
        fn = namespace.get(name)
        if fn is None:
            available = [k for k in namespace if callable(namespace[k]) and not k.startswith("_")]
            fn = namespace[available[0]] if len(available) == 1 else None
        if fn is None:
            return {"error": f"no callable found in package '{name}'"}

        with monitor_effects() as report:
            result = fn(**req.args)

        effects = report.to_list()
        if effects:
            log.info("tool '%s' effects: %s", name, effects)

        result_str = str(result)
        validator.record_execution(name, req.args, result_str)

        causality_info = causal.to_dict()
        if data_flow:
            causality_info["data_flow"] = [e.to_dict() for e in data_flow]
        if flow_warnings:
            causality_info["flow_warnings"] = flow_warnings

        return {
            "result": result_str,
            "tool": name,
            "effects": effects,
            "causality": causality_info,
        }
    except Exception as e:
        log.exception("tool execution failed: %s", name)
        return {"error": f"{type(e).__name__}: {e}", "tool": name}


@app.post("/keys/{publisher}")
async def generate_key(publisher: str):
    public_key = _key_store.generate(publisher)
    return {"publisher": publisher, "public_key": public_key}


@app.get("/keys/{publisher}")
async def get_public_key(publisher: str):
    try:
        public_key = _key_store.get_public_key(publisher)
    except FileNotFoundError:
        return {"error": f"no key for publisher '{publisher}'"}
    return {"publisher": publisher, "public_key": public_key}


@app.get("/keys")
async def list_publishers():
    return {"publishers": _key_store.list_publishers()}


@app.get("/tunnels")
async def list_tunnels():
    reg = get_tunnel_registry()
    return {"tunnels": reg.active_sessions}


@app.delete("/tunnels/{session_id}")
async def close_tunnel(session_id: str):
    reg = get_tunnel_registry()
    count = reg.close_session(session_id)
    close_causality_session(session_id)
    return {"closed": count, "session_id": session_id}


@app.delete("/sessions/{session_id}/causality")
async def reset_causality(session_id: str):
    close_causality_session(session_id)
    return {"reset": session_id}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "packages": _registry.count,
        "publishers": len(_key_store.list_publishers()),
        "tunnels": len(get_tunnel_registry().active_sessions),
    }
