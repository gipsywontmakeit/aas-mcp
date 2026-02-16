"""
MCP server for Asset Administration Shell (AAS) BaSyx integration (READ-ONLY).

LLM-navigation upgrades (no hardcode for specific questions):
- ALWAYS uses encode=True for any BaSyx call that takes AAS/Submodel IDs in the path
- build_index() + search(): lightweight discovery (avoid full list scans where possible)
- resolve_identifier(): idShort/URL resolution with candidates
- describe_shell(): compact "map" of shell + linked submodels
- describe_submodel(): outline + validPaths (avoid guessing idShort paths)
- get_submodel_element(): NotFound returns suggestions + hint (anti-loop UX)
- TTL cache + in-flight dedupe (prevents repeated/loop calls hammering BaSyx)
- TOOL_START / TOOL_END / TOOL_ERROR logging with dt_ms and result summaries

IMPORTANT:
FastMCP turns @app.tool functions into tool objects. Tools must NOT call other tools
directly (it becomes 'FunctionTool' not callable). Therefore:
- All shared logic lives in internal *_impl helpers
- Tools are thin wrappers that call those helpers
"""

import asyncio
import functools
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, List, Dict

import httpx
from fastmcp import FastMCP
from shellsmith.clients import AsyncClient
from shellsmith.config import config

logger = logging.getLogger("aas_mcp")

# -----------------------------
# Logging helpers
# -----------------------------

_TOOL_CALL_SEQ = 0


def _next_call_id() -> int:
    global _TOOL_CALL_SEQ
    _TOOL_CALL_SEQ += 1
    return _TOOL_CALL_SEQ


def _summarize_result(res: Any) -> str:
    """Compact summary of results to avoid huge log lines."""
    try:
        if res is None:
            return "None"
        if isinstance(res, dict):
            keys = list(res.keys())
            return f"dict(nkeys={len(keys)}, keys={keys[:12]})"
        if isinstance(res, list):
            return f"list(n={len(res)})"
        if isinstance(res, (str, int, float, bool)):
            s = str(res)
            s = s[:160] + ("..." if len(s) > 160 else "")
            return f"{type(res).__name__}({s})"
        return f"{type(res).__name__}"
    except Exception:
        return f"{type(res).__name__}(unprintable)"


def _pick_kwargs(kwargs: dict) -> str:
    interesting = [
        "shell_id",
        "submodel_id",
        "id_short_path",
        "host",
        "timeout",
        "refresh",
        "query",
        "kind",
        "value",
    ]
    picked = {k: kwargs.get(k) for k in interesting if k in kwargs}
    return ", ".join(f"{k}={picked[k]!r}" for k in picked.keys()) or "(no-key-args)"


def log_tool(fn):
    """Decorator for MCP tools that logs start/end/error with timing."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        call_id = _next_call_id()
        t0 = time.perf_counter()

        logger.info(
            "TOOL_START id=%s tool=%s args=[%s]",
            call_id,
            fn.__name__,
            _pick_kwargs(kwargs),
        )

        try:
            res = await fn(*args, **kwargs)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            logger.info(
                "TOOL_END   id=%s tool=%s ok=true dt_ms=%.1f result=%s",
                call_id,
                fn.__name__,
                dt_ms,
                _summarize_result(res),
            )
            return res
        except Exception as e:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logger.exception(
                "TOOL_ERROR id=%s tool=%s ok=false dt_ms=%.1f err=%r",
                call_id,
                fn.__name__,
                dt_ms,
                e,
            )
            raise

    return wrapper


# -----------------------------
# TTL cache + in-flight dedupe
# -----------------------------

@dataclass
class _CacheEntry:
    expires_at: float
    value: Any


_CACHE: dict[str, _CacheEntry] = {}
_INFLIGHT: dict[str, asyncio.Future] = {}
_CACHE_LOCK = asyncio.Lock()
_NOTFOUND_COUNTER: dict[str,int] = {}

def _now_s() -> float:
    return time.time()


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def _args_key(tool: str, kwargs: dict) -> str:
    payload = {"tool": tool, "kwargs": kwargs}
    h = hashlib.sha1(_stable_json(payload).encode("utf-8")).hexdigest()
    return f"{tool}:{h}"


async def _cached_call(*, key: str, ttl_s: float, fn: Callable[[], Any]) -> Tuple[Any, bool]:
    async with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry and entry.expires_at > _now_s():
            return entry.value, True

        fut = _INFLIGHT.get(key)
        if fut is None:
            fut = asyncio.get_running_loop().create_future()
            _INFLIGHT[key] = fut
            is_owner = True
        else:
            is_owner = False

    if not is_owner:
        val = await fut
        return val, True

    try:
        val = await fn()
        async with _CACHE_LOCK:
            _CACHE[key] = _CacheEntry(expires_at=_now_s() + ttl_s, value=val)
            if not fut.done():
                fut.set_result(val)
            _INFLIGHT.pop(key, None)
        return val, False
    except Exception as e:
        async with _CACHE_LOCK:
            _INFLIGHT.pop(key, None)
            if not fut.done():
                fut.set_exception(e)
        raise


def _with_meta(*, tool: str, host: str, encode: Optional[bool], dt_ms: float, cache_hit: bool, data: Any) -> dict:
    return {
        "meta": {
            "tool": tool,
            "host": host,
            "encode": encode,
            "dt_ms": round(dt_ms, 1),
            "cache_hit": bool(cache_hit),
        },
        "data": data,
    }


# -----------------------------
# Index + discovery helpers
# -----------------------------

_INDEX: dict[str, Any] = {
    "built_at": 0.0,
    "host": None,
    "shell_by_idshort": {},
    "submodels_by_idshort": {},
    "shell_submodels": {},
}


def _norm(s: str) -> str:
    return (s or "").strip()


def _looks_like_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _rank_match(text: str, q: str) -> int:
    t = (text or "").lower()
    ql = (q or "").lower()
    if not t or not ql:
        return 99
    if t == ql:
        return 0
    if t.startswith(ql):
        return 1
    if ql in t:
        return 2
    return 99


def _top_matches(items: List[dict], q: str, limit: int) -> List[dict]:
    scored: List[Tuple[int, dict]] = []
    for it in items:
        idshort = it.get("idShort") or ""
        sid = it.get("id") or ""
        score = min(_rank_match(idshort, q), _rank_match(sid, q))
        if score < 99:
            scored.append((score, it))
    scored.sort(key=lambda x: x[0])
    return [it for _, it in scored[: max(1, limit)]]


def _unwrap_shellsmith_list(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "result" in obj:
            return obj.get("result")
        if "data" in obj:
            return obj.get("data")
    return obj


async def _build_index_impl(host: str) -> dict:
    async with AsyncClient(host=host) as client:
        shells_raw = await client.get_shells()
        submodels_raw = await client.get_submodels()

    shells_items = _unwrap_shellsmith_list(shells_raw)
    subs_items = _unwrap_shellsmith_list(submodels_raw)

    shell_by_idshort: Dict[str, str] = {}
    shell_submodels: Dict[str, List[str]] = {}

    if isinstance(shells_items, list):
        for sh in shells_items:
            if not isinstance(sh, dict):
                continue
            sid = sh.get("id")
            idshort = sh.get("idShort")
            if sid and idshort:
                shell_by_idshort[idshort] = sid

            sm_ids: List[str] = []
            for smref in (sh.get("submodels") or []):
                if not isinstance(smref, dict):
                    continue
                keys = smref.get("keys") or []
                for k in keys:
                    if isinstance(k, dict) and k.get("type") == "Submodel" and k.get("value"):
                        sm_ids.append(k["value"])
            if sid:
                shell_submodels[sid] = sm_ids

    submodels_by_idshort: Dict[str, List[str]] = {}
    if isinstance(subs_items, list):
        for sm in subs_items:
            if not isinstance(sm, dict):
                continue
            smid = sm.get("id")
            idshort = sm.get("idShort")
            if smid and idshort:
                submodels_by_idshort.setdefault(idshort, []).append(smid)

    _INDEX["built_at"] = _now_s()
    _INDEX["host"] = host
    _INDEX["shell_by_idshort"] = shell_by_idshort
    _INDEX["submodels_by_idshort"] = submodels_by_idshort
    _INDEX["shell_submodels"] = shell_submodels

    return {
        "built_at": _INDEX["built_at"],
        "host": host,
        "shells_indexed": len(shell_by_idshort),
        "submodels_indexed": sum(len(v) for v in submodels_by_idshort.values()),
    }


async def _ensure_index_impl(host: str, max_age_s: float = 60.0) -> None:
    if _INDEX.get("host") != host or (_now_s() - float(_INDEX.get("built_at", 0.0))) > max_age_s:
        await _build_index_impl(host)


async def _search_impl(kind: str, query: str, host: str, limit: int = 10) -> dict:
    query = _norm(query)
    if not query:
        return {"meta": {"note": "empty query"}, "data": []}

    await _ensure_index_impl(host)

    if kind == "shell":
        items = [{"idShort": k, "id": v} for k, v in _INDEX["shell_by_idshort"].items()]
        results = _top_matches(items, query, limit)
        return {"meta": {"kind": kind, "query": query, "limit": limit}, "data": results}

    if kind == "submodel":
        items: List[dict] = []
        for idshort, ids in _INDEX["submodels_by_idshort"].items():
            for sid in ids:
                items.append({"idShort": idshort, "id": sid})
        results = _top_matches(items, query, limit)
        return {"meta": {"kind": kind, "query": query, "limit": limit}, "data": results}

    return {"error": "InvalidKind", "message": "kind must be 'shell' or 'submodel'"}


async def _resolve_identifier_impl(value: str, host: str, kinds: List[str]) -> dict:
    value = _norm(value)
    if not value:
        return {"error": "EmptyValue"}

    if _looks_like_url(value):
        return {"input": value, "detected": "url", "resolved": {"id": value}, "candidates": []}

    await _ensure_index_impl(host)

    resolved = None
    candidates: List[dict] = []

    if "shell" in kinds:
        sid = _INDEX["shell_by_idshort"].get(value)
        if sid:
            resolved = {"type": "shell", "idShort": value, "id": sid}

    if resolved is None and "submodel" in kinds:
        smids = _INDEX["submodels_by_idshort"].get(value) or []
        if len(smids) == 1:
            resolved = {"type": "submodel", "idShort": value, "id": smids[0]}
        elif len(smids) > 1:
            candidates = [{"type": "submodel", "idShort": value, "id": x} for x in smids]

    if resolved is None:
        for k in kinds:
            hits = await _search_impl(k, value, host=host, limit=5)
            for it in hits.get("data", []):
                candidates.append({"type": k, "idShort": it.get("idShort"), "id": it.get("id")})

        return {
            "input": value,
            "detected": "idShort",
            "resolved": None,
            "candidates": candidates,
            "message": "Not found as exact idShort. See candidates.",
        }

    return {"input": value, "detected": "idShort", "resolved": resolved, "candidates": candidates}


# -----------------------------
# Outline + path helpers
# -----------------------------

def _outline_element(elem: dict, depth: int, max_children: int) -> dict:
    mt = elem.get("modelType")
    out = {"idShort": elem.get("idShort"), "modelType": mt}

    if mt == "Property":
        out["valueType"] = elem.get("valueType")
        out["semanticId"] = elem.get("semanticId")
        return out

    if depth <= 0:
        return out

    children = elem.get("value")
    if isinstance(children, list):
        out["children"] = [
            _outline_element(c, depth - 1, max_children)
            for c in children[:max_children]
            if isinstance(c, dict)
        ]
        if len(children) > max_children:
            out["children_truncated"] = True
            out["children_total"] = len(children)

    return out


def _collect_paths(elem: dict, base: str, depth: int, max_children: int, acc: List[str]) -> None:
    if not isinstance(elem, dict):
        return
    idshort = elem.get("idShort")
    if not idshort:
        return
    path = f"{base}.{idshort}" if base else idshort
    acc.append(path)
    if depth <= 0:
        return
    children = elem.get("value")
    if isinstance(children, list):
        for c in children[:max_children]:
            _collect_paths(c, path, depth - 1, max_children, acc)


def _suggest_paths(paths: List[str], bad: str, limit: int = 10) -> List[str]:
    bad_l = (bad or "").lower()
    scored: List[Tuple[int, str]] = []
    for p in paths:
        pl = p.lower()
        score = 99
        if pl == bad_l:
            score = 0
        elif pl.endswith("/" + bad_l) or pl.startswith(bad_l):
            score = 1
        elif bad_l in pl:
            score = 2
        scored.append((score, p))
    scored.sort(key=lambda x: x[0])
    return [p for s, p in scored[:limit] if s < 99]


# -----------------------------
# HTTP error helpers
# -----------------------------

def _is_http_404(e: Exception) -> bool:
    if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
        return e.response.status_code == 404
    return False


def _http_err_details(e: Exception) -> dict:
    if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
        return {
            "status_code": e.response.status_code,
            "url": str(e.request.url) if e.request is not None else None,
            "text": (e.response.text or "")[:400],
        }
    return {"error": repr(e)}


async def _describe_submodel_impl(
    submodel: str,
    host: str,
    depth: int = 2,
    max_children: int = 50,
) -> dict:
    t0 = time.perf_counter()

    rid = await _resolve_identifier_impl(submodel, host=host, kinds=["submodel"])
    if rid.get("resolved") is None:
        return {"error": "SubmodelNotResolved", "details": rid}

    submodel_id = rid["resolved"]["id"]

    key_sm = _args_key("client.get_submodel", {"submodel_id": submodel_id, "host": host})

    async def _do_sm():
        async with AsyncClient(host=host) as client:
            return await client.get_submodel(submodel_id, encode=True)

    sm_obj, hit = await _cached_call(key=key_sm, ttl_s=60.0, fn=_do_sm)

    elements = (sm_obj.get("submodelElements") or []) if isinstance(sm_obj, dict) else []

    outline = {
        "id": sm_obj.get("id") if isinstance(sm_obj, dict) else submodel_id,
        "idShort": sm_obj.get("idShort") if isinstance(sm_obj, dict) else rid["resolved"].get("idShort"),
        "modelType": sm_obj.get("modelType") if isinstance(sm_obj, dict) else "Submodel",
        "elements": [
            _outline_element(e, depth=depth, max_children=max_children)
            for e in (elements[:max_children] if isinstance(elements, list) else [])
            if isinstance(e, dict)
        ],
        "elements_truncated": isinstance(elements, list) and len(elements) > max_children,
        "elements_total": len(elements) if isinstance(elements, list) else None,
    }

    paths: List[str] = []
    if isinstance(elements, list):
        for e in elements[:max_children]:
            _collect_paths(e, base="", depth=depth, max_children=max_children, acc=paths)

    dt_ms = (time.perf_counter() - t0) * 1000.0
    return _with_meta(
        tool="describe_submodel",
        host=host,
        encode=True,
        dt_ms=dt_ms,
        cache_hit=hit,
        data={"outline": outline, "validPaths": paths},
    )


# -----------------------------
# MCP App
# -----------------------------

app = FastMCP(
    name="aas-mcp",
    instructions="""
This server provides READ-ONLY tools for navigating Asset Administration Shells (AAS)
using the Shellsmith Python SDK to interact with Eclipse BaSyx environments.

IMPORTANT (BaSyx encoding):
- Pass shell_id / submodel_id as the canonical ID (URL). Do NOT pass Base64 yourself.
- This server ALWAYS calls BaSyx endpoints with encode=True for all ID-based requests.

LLM-safe navigation pattern (recommended):
1) resolve_identifier() for any human-readable identifier (idShort).
2) describe_shell() to see which submodels are linked.
3) describe_submodel() to discover validPaths before calling get_submodel_element().
4) Prefer search() over get_shells/get_submodels for discovery.
""",
)

# -----------------------------
# Tools: Index + discovery
# -----------------------------

@app.tool()
@log_tool
async def build_index(host: str = config.host, refresh: bool = False) -> dict:
    """Builds/refreshes an in-memory index for idShort resolution and lightweight search."""
    key = _args_key("build_index", {"host": host})

    async def _do():
        return await _build_index_impl(host)

    if refresh:
        res = await _do()
        return {"meta": {"refreshed": True}, "data": res}

    res, hit = await _cached_call(key=key, ttl_s=30.0, fn=_do)
    return {"meta": {"cache_hit": hit, "refreshed": False}, "data": res}


@app.tool()
@log_tool
async def search(kind: str, query: str, host: str = config.host, limit: int = 10) -> dict:
    """Lightweight search for shells/submodels by idShort or id (URL). kind: 'shell' | 'submodel'."""
    return await _search_impl(kind=kind, query=query, host=host, limit=limit)


@app.tool()
@log_tool
async def resolve_identifier(value: str, host: str = config.host, kinds: list[str] = ["shell", "submodel"]) -> dict:
    """
    Resolves an identifier that might be:
    - URL (returns as-is)
    - idShort (uses index)
    If not found, returns candidates (via search).
    """
    return await _resolve_identifier_impl(value=value, host=host, kinds=list(kinds))


@app.tool()
@log_tool
async def get_tool_guide(goal: str = "navigate_aas") -> dict:
    """Operational playbook for safe navigation without guessing IDs/paths."""
    return {
        "goal": goal,
        "steps": [
            "1) resolve_identifier(value) for any human-readable identifier (idShort).",
            "2) describe_shell(shell) to list linked submodels.",
            "3) If you need element paths, call describe_submodel(submodel, depth=2) and use validPaths.",
            "4) Call get_submodel_element(submodel_id, id_short_path) only with known valid paths.",
            "5) Prefer search(kind, query) over get_shells/get_submodels for discovery.",
        ],
        "common_pitfalls": [
            "Do not pass idShort directly into get_shell/get_submodel; resolve first.",
            "Avoid repeating calls with same args; cache/dedupe will reuse results.",
            "If get_submodel_element returns NotFound, use suggestions or describe_submodel().",
        ],
    }


# -----------------------------
# Tools: Shell read operations
# -----------------------------

@app.tool()
@log_tool
async def get_shells(host: str = config.host) -> dict:
    """Get all shells from the BaSyx environment."""
    key = _args_key("get_shells", {"host": host})

    async def _do():
        async with AsyncClient(host=host) as client:
            return await client.get_shells()

    t0 = time.perf_counter()
    res, hit = await _cached_call(key=key, ttl_s=5.0, fn=_do)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return _with_meta(tool="get_shells", host=host, encode=None, dt_ms=dt_ms, cache_hit=hit, data=res)


@app.tool()
@log_tool
async def get_shell(shell_id: str, host: str = config.host) -> dict:
    """Get a specific shell by its full ID (URL)."""
    key = _args_key("get_shell", {"shell_id": shell_id, "host": host})

    async def _do():
        async with AsyncClient(host=host) as client:
            return await client.get_shell(shell_id, encode=True)

    t0 = time.perf_counter()
    try:
        res, hit = await _cached_call(key=key, ttl_s=30.0, fn=_do)
    except Exception as exc:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 404:
            return _with_meta(
                tool="get_shell", host=host, encode=True,
                dt_ms=dt_ms, cache_hit=False,
                data={
                    "error": "ShellNotFound",
                    "message": (
                        f"Shell '{shell_id}' does not exist in BaSyx. "
                        f"You probably used the idShort instead of the id. "
                        f"These can differ (e.g. idShort='TK-RECT-A' but id ends with 'TK-RECT-01'). "
                        f"Go back to your get_shells result and use the exact 'id' field."
                    ),
                    "shell_id_used": shell_id,
                },
            )
        raise
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return _with_meta(tool="get_shell", host=host, encode=True, dt_ms=dt_ms, cache_hit=hit, data=res)


# -----------------------------
# Tools: Submodel references (read-only)
# -----------------------------

@app.tool()
@log_tool
async def get_submodel_refs(shell_id: str, host: str = config.host) -> dict:
    """Get submodel references for a shell."""
    key = _args_key("get_submodel_refs", {"shell_id": shell_id, "host": host})

    async def _do():
        async with AsyncClient(host=host) as client:
            return await client.get_submodel_refs(shell_id, encode=True)

    t0 = time.perf_counter()
    try:
        res, hit = await _cached_call(key=key, ttl_s=30.0, fn=_do)
    except Exception as exc:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 404:
            return _with_meta(
                tool="get_submodel_refs", host=host, encode=True,
                dt_ms=dt_ms, cache_hit=False,
                data={
                    "error": "ShellNotFound",
                    "message": (
                        f"Shell '{shell_id}' does not exist in BaSyx. "
                        f"You probably used the idShort instead of the id. "
                        f"These can differ (e.g. idShort='TK-RECT-A' but id ends with 'TK-RECT-01'). "
                        f"Go back to your get_shells result and use the exact 'id' field."
                    ),
                    "shell_id_used": shell_id,
                },
            )
        raise
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return _with_meta(tool="get_submodel_refs", host=host, encode=True, dt_ms=dt_ms, cache_hit=hit, data=res)


# -----------------------------
# Tools: Submodel read operations
# -----------------------------

@app.tool()
@log_tool
async def get_submodels(host: str = config.host) -> dict:
    """Get all submodels from the BaSyx environment."""
    key = _args_key("get_submodels", {"host": host})

    async def _do():
        async with AsyncClient(host=host) as client:
            return await client.get_submodels()

    t0 = time.perf_counter()
    res, hit = await _cached_call(key=key, ttl_s=10.0, fn=_do)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return _with_meta(tool="get_submodels", host=host, encode=None, dt_ms=dt_ms, cache_hit=hit, data=res)


@app.tool()
@log_tool
async def get_submodel(submodel_id: str, host: str = config.host) -> dict:
    """Get a specific submodel by its full ID (URL)."""
    key = _args_key("get_submodel", {"submodel_id": submodel_id, "host": host})

    async def _do():
        async with AsyncClient(host=host) as client:
            return await client.get_submodel(submodel_id, encode=True)

    t0 = time.perf_counter()
    res, hit = await _cached_call(key=key, ttl_s=60.0, fn=_do)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return _with_meta(tool="get_submodel", host=host, encode=True, dt_ms=dt_ms, cache_hit=hit, data=res)


@app.tool()
@log_tool
async def get_submodel_value(submodel_id: str, host: str = config.host) -> dict:
    """Get the value of a submodel."""
    async with AsyncClient(host=host) as client:
        return await client.get_submodel_value(submodel_id, encode=True)


@app.tool()
@log_tool
async def get_submodel_metadata(submodel_id: str, host: str = config.host) -> dict:
    """Get metadata for a submodel."""
    key = _args_key("get_submodel_metadata", {"submodel_id": submodel_id, "host": host})

    async def _do():
        async with AsyncClient(host=host) as client:
            return await client.get_submodel_metadata(submodel_id, encode=True)

    t0 = time.perf_counter()
    res, hit = await _cached_call(key=key, ttl_s=120.0, fn=_do)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return _with_meta(tool="get_submodel_metadata", host=host, encode=True, dt_ms=dt_ms, cache_hit=hit, data=res)


# -----------------------------
# Tools: Describe (LLM navigation)
# -----------------------------

@app.tool()
@log_tool
async def describe_shell(shell: str, host: str = config.host) -> dict:
    """
    Compact shell description + linked submodels.
    Input can be idShort or URL (shell id).
    ALWAYS uses encode=True internally.
    """
    t0 = time.perf_counter()

    rid = await _resolve_identifier_impl(shell, host=host, kinds=["shell"])
    if rid.get("resolved") is None:
        return {"error": "ShellNotResolved", "details": rid}

    shell_id = rid["resolved"]["id"]

    key_shell = _args_key("client.get_shell", {"shell_id": shell_id, "host": host})
    async def _do_shell():
        async with AsyncClient(host=host) as client:
            return await client.get_shell(shell_id, encode=True)

    shell_obj, hit_shell = await _cached_call(key=key_shell, ttl_s=30.0, fn=_do_shell)

    key_refs = _args_key("client.get_submodel_refs", {"shell_id": shell_id, "host": host})
    async def _do_refs():
        async with AsyncClient(host=host) as client:
            return await client.get_submodel_refs(shell_id, encode=True)

    refs_obj, hit_refs = await _cached_call(key=key_refs, ttl_s=30.0, fn=_do_refs)

    refs_items = _unwrap_shellsmith_list(refs_obj)
    submodel_ids: List[str] = []
    if isinstance(refs_items, list):
        for r in refs_items:
            if not isinstance(r, dict):
                continue
            for k in (r.get("keys") or []):
                if isinstance(k, dict) and k.get("type") == "Submodel" and k.get("value"):
                    submodel_ids.append(k["value"])

    submodels: List[dict] = []
    for smid in submodel_ids[:200]:
        key_smmeta = _args_key("client.get_submodel_metadata", {"submodel_id": smid, "host": host})
        async def _do_smmeta(smid=smid):
            async with AsyncClient(host=host) as client:
                return await client.get_submodel_metadata(smid, encode=True)
        try:
            smmeta, _ = await _cached_call(key=key_smmeta, ttl_s=120.0, fn=_do_smmeta)
            if isinstance(smmeta, dict):
                submodels.append({
                    "id": smid,
                    "idShort": smmeta.get("idShort"),
                    "semanticId": smmeta.get("semanticId"),
                })
            else:
                submodels.append({"id": smid, "idShort": None, "semanticId": None})
        except Exception:
            submodels.append({"id": smid, "idShort": None, "semanticId": None})

    dt_ms = (time.perf_counter() - t0) * 1000.0
    return _with_meta(
        tool="describe_shell",
        host=host,
        encode=True,
        dt_ms=dt_ms,
        cache_hit=(hit_shell and hit_refs),
        data={
            "shell": {
                "id": shell_obj.get("id") if isinstance(shell_obj, dict) else shell_id,
                "idShort": shell_obj.get("idShort") if isinstance(shell_obj, dict) else rid["resolved"].get("idShort"),
                "assetInformation": shell_obj.get("assetInformation") if isinstance(shell_obj, dict) else None,
                "derivedFrom": shell_obj.get("derivedFrom") if isinstance(shell_obj, dict) else None,
            },
            "submodels": submodels,
            "submodelIds": submodel_ids,
        },
    )


@app.tool()
@log_tool
async def describe_submodel(
    submodel: str,
    host: str = config.host,
    depth: int = 2,
    max_children: int = 50,
) -> dict:
    """
    Submodel outline + validPaths for element navigation.
    Input can be idShort or URL (submodel id).
    """
    return await _describe_submodel_impl(
        submodel=submodel,
        host=host,
        depth=depth,
        max_children=max_children,
    )


# -----------------------------
# Tools: Submodel element read operations
# -----------------------------

@app.tool()
@log_tool
async def get_submodel_elements(submodel_id: str, host: str = config.host) -> dict:
    """Get all elements of a submodel."""
    async with AsyncClient(host=host) as client:
        return await client.get_submodel_elements(submodel_id, encode=True)


@app.tool()
@log_tool
async def get_submodel_element(
    submodel_id: str,
    id_short_path: str,
    host: str = config.host,
) -> dict:
    """
    Gets a submodel element by path.
    On NotFound, returns suggestions + hint (prevents LLM loops).
    ALWAYS uses encode=True internally.
    """
    id_short_path = _norm(id_short_path)
    id_short_path = id_short_path.replace("/", ".")
    key = _args_key(
        "client.get_submodel_element",
        {"submodel_id": submodel_id, "id_short_path": id_short_path, "host": host},
    )

    async def _do():
        async with AsyncClient(host=host) as client:
            try:
                return await client.get_submodel_element(submodel_id, id_short_path, encode=True)
            except Exception as e:
                if _is_http_404(e):
                    return {
                        "__not_found__": True,
                        "error": "NotFound",
                        "message": "Element path not found",
                        "path": id_short_path,
                        "details": _http_err_details(e),
                    }
                raise

    t0 = time.perf_counter()
    res, hit = await _cached_call(key=key, ttl_s=30.0, fn=_do)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    if isinstance(res, dict) and res.get("__not_found__"):
        counter_key = f"{submodel_id}::{id_short_path}"
        _NOTFOUND_COUNTER[counter_key] = _NOTFOUND_COUNTER.get(counter_key, 0) + 1

        if _NOTFOUND_COUNTER[counter_key] >= 3:
            payload = {
                "error": "RepeatedNotFound",
                "message": (
                    f"STOP: Path '{id_short_path}' has been tried {_NOTFOUND_COUNTER[counter_key]} times and does not exist. "
                    f"You MUST call describe_submodel('{submodel_id}') to discover valid paths. "
                    f"Do NOT retry this path."
                ),
                "path": id_short_path,
            }
            return _with_meta(tool="get_submodel_element", host=host, encode=True, dt_ms=dt_ms, cache_hit=hit, data=payload)


        suggestions: List[str] = []
        try:
            desc = await _describe_submodel_impl(submodel=submodel_id, host=host, depth=2, max_children=80)
            paths = (desc.get("data") or {}).get("validPaths") or []
            suggestions = _suggest_paths(paths, id_short_path, limit=10)
        except Exception:
            suggestions = []

        payload = dict(res)
        payload.pop("__not_found__", None)
        payload["suggestions"] = suggestions
        payload["hint"] = "Call describe_submodel() and use validPaths before guessing paths."
        return _with_meta(tool="get_submodel_element", host=host, encode=True, dt_ms=dt_ms, cache_hit=hit, data=payload)

    return _with_meta(tool="get_submodel_element", host=host, encode=True, dt_ms=dt_ms, cache_hit=hit, data=res)


@app.tool()
@log_tool
async def get_submodel_element_value(
    submodel_id: str,
    id_short_path: str,
    host: str = config.host,
) -> dict | list | str | int | float | bool | None:
    """Get the raw value of a submodel element."""
    id_short_path = id_short_path.replace("/", ".")
    async with AsyncClient(host=host) as client:
        return await client.get_submodel_element_value(submodel_id, id_short_path, encode=True)


# -----------------------------
# Tools: Health monitoring
# -----------------------------

@app.tool()
@log_tool
async def get_health_status(host: str = config.host, timeout: float = config.timeout) -> str:
    """Get the health status of the BaSyx environment."""
    async with AsyncClient(host=host, timeout=timeout) as client:
        return await client.get_health_status()


@app.tool()
@log_tool
async def is_healthy(host: str = config.host, timeout: float = config.timeout) -> bool:
    """Check if the BaSyx environment is healthy."""
    async with AsyncClient(host=host, timeout=timeout) as client:
        return await client.is_healthy()


# -----------------------------
# Entrypoint
# -----------------------------

def cli_main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger.info("Starting AAS MCP server (cli_main)")
    host = os.getenv("MCP_HTTP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_HTTP_PORT", "8000"))
    path = os.getenv("MCP_HTTP_PATH", "/mcp")

    logger.info("HTTP transport config host=%s port=%s path=%s default_host=%s", host, port, path, config.host)

    app.run(
        transport="streamable-http",
        host=host,
        port=port,
        path=path,
    )