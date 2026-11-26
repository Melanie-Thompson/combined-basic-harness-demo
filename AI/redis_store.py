"""Redis connection management with in-memory fallback for deployments without Redis."""

import json
import logging
import os

SESSION_TTL = 3600  # 1 hour
KEY_PREFIX = "chat:session"
CLUSTER_PREFIX = "context:clusters"
TOOL_CTX_PREFIX = "tool_ctx"

log = logging.getLogger("redis_store")

_redis = None
_memory: dict[str, str] = {}
_memory_lists: dict[str, list[str]] = {}


async def connect(url: str | None = None) -> None:
    global _redis
    redis_url = url or os.getenv("REDIS_URL", "")
    if not redis_url:
        log.info("no REDIS_URL set — using in-memory store")
        return
    try:
        import redis.asyncio as aioredis
        _redis = aioredis.from_url(redis_url, decode_responses=True)
        await _redis.ping()
        log.info("connected to Redis at %s", redis_url.split("@")[-1] if "@" in redis_url else redis_url)
    except Exception as e:
        log.warning("Redis connection failed (%s) — falling back to in-memory store", e)
        _redis = None


async def close() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


def _key(session_id: str) -> str:
    return f"{KEY_PREFIX}:{session_id}"


async def get(session_id: str) -> str | None:
    if _redis is not None:
        return await _redis.get(_key(session_id))
    return _memory.get(_key(session_id))


async def set(session_id: str, data: str) -> None:
    if _redis is not None:
        await _redis.set(_key(session_id), data, ex=SESSION_TTL)
    else:
        _memory[_key(session_id)] = data


def _cluster_key(session_id: str) -> str:
    return f"{CLUSTER_PREFIX}:{session_id}"


async def get_clusters(session_id: str) -> str | None:
    if _redis is not None:
        return await _redis.get(_cluster_key(session_id))
    return _memory.get(_cluster_key(session_id))


async def set_clusters(session_id: str, data: str) -> None:
    if _redis is not None:
        await _redis.set(_cluster_key(session_id), data, ex=SESSION_TTL)
    else:
        _memory[_cluster_key(session_id)] = data


# ── Tool context store ────────────────────────────────────────────────────

def _tool_ctx_key(session_id: str, call_id: str) -> str:
    return f"{TOOL_CTX_PREFIX}:{session_id}:{call_id}"


def _tool_ctx_index_key(session_id: str) -> str:
    return f"{TOOL_CTX_PREFIX}:{session_id}:_index"


async def store_tool_result(session_id: str, call_id: str, tool_name: str, result: str) -> None:
    key = _tool_ctx_key(session_id, call_id)
    value = json.dumps({"tool": tool_name, "call_id": call_id, "result": result})
    if _redis is not None:
        await _redis.set(key, value, ex=SESSION_TTL)
        await _redis.rpush(_tool_ctx_index_key(session_id), call_id)
        await _redis.expire(_tool_ctx_index_key(session_id), SESSION_TTL)
    else:
        _memory[key] = value
        idx_key = _tool_ctx_index_key(session_id)
        _memory_lists.setdefault(idx_key, []).append(call_id)


async def get_tool_result(session_id: str, call_id: str) -> dict | None:
    if _redis is not None:
        raw = await _redis.get(_tool_ctx_key(session_id, call_id))
    else:
        raw = _memory.get(_tool_ctx_key(session_id, call_id))
    if raw is None:
        return None
    return json.loads(raw)


async def list_tool_results(session_id: str) -> list[dict]:
    index_key = _tool_ctx_index_key(session_id)
    if _redis is not None:
        call_ids = await _redis.lrange(index_key, 0, -1)
    else:
        call_ids = _memory_lists.get(index_key, [])
    results = []
    for cid in call_ids:
        entry = await get_tool_result(session_id, cid)
        if entry:
            results.append(entry)
    return results


async def clear_tool_context(session_id: str) -> int:
    index_key = _tool_ctx_index_key(session_id)
    if _redis is not None:
        call_ids = await _redis.lrange(index_key, 0, -1)
        count = 0
        for cid in call_ids:
            count += await _redis.delete(_tool_ctx_key(session_id, cid))
        await _redis.delete(index_key)
        return count
    else:
        call_ids = _memory_lists.pop(index_key, [])
        count = 0
        for cid in call_ids:
            key = _tool_ctx_key(session_id, cid)
            if key in _memory:
                del _memory[key]
                count += 1
        return count
