"""
MCP server for Asset Administration Shell (AAS) BaSyx integration.

Adds precise, consistent logging for every tool call:
- TOOL_START / TOOL_END / TOOL_ERROR
- tool name, key arguments, timing (ms), and result summary
- stack traces on exceptions
- optional request correlation id per tool call
"""

import logging
import os
import time
import functools
from typing import Any, Optional

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
    """
    Log only the most useful parameters (and avoid dumping full payloads like 'shell' or 'submodel').
    """
    interesting = [
        "shell_id",
        "submodel_id",
        "id_short_path",
        "encode",
        "host",
        "timeout",
    ]
    picked = {k: kwargs.get(k) for k in interesting if k in kwargs}

    # Avoid huge payloads
    if "shell" in kwargs:
        picked["shell"] = "<dict>"
    if "submodel" in kwargs:
        picked["submodel"] = "<dict>"
    if "element" in kwargs:
        picked["element"] = "<dict>"
    if "value" in kwargs:
        v = kwargs["value"]
        picked["value"] = f"<{type(v).__name__}>"

    return ", ".join(f"{k}={picked[k]!r}" for k in picked.keys()) or "(no-key-args)"


def log_tool(fn):
    """
    Decorator for MCP tools that logs:
    - start/end/error
    - elapsed time
    - key args
    - result summary
    """
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
# MCP App
# -----------------------------

app = FastMCP(
    name="aas-mcp",
    instructions="""
This server provides tools for managing Asset Administration Shells (AAS)
using the Shellsmith Python SDK to interact with Eclipse BaSyx environments.

Available capabilities:
- Shell management: get_shells(), get_shell(), create_shell(),
  update_shell(), delete_shell()
- Submodel management: get_submodels(), get_submodel(), create_submodel(),
  update_submodel(), delete_submodel()
- Submodel element operations: get_submodel_elements(), get_submodel_element(),
  create_submodel_element(), update_submodel_element(),
  delete_submodel_element()
- Value operations: get_submodel_value(), update_submodel_value(),
  get_submodel_element_value(), update_submodel_element_value()
- Reference management: get_submodel_refs(), create_submodel_ref(),
  delete_submodel_ref()
- Health monitoring: get_health_status(), is_healthy()

All tools accept a 'host' parameter to override the default BaSyx server URL.
IDs are automatically Base64-encoded unless 'encode=False' is specified.
""",
)


# -----------------------------
# Tools: Shell management
# -----------------------------

@app.tool()
@log_tool
async def get_shells(host: str = config.host) -> dict:
    async with AsyncClient(host=host) as client:
        return await client.get_shells()


@app.tool()
@log_tool
async def get_shell(shell_id: str, encode: bool = True, host: str = config.host) -> dict:
    async with AsyncClient(host=host) as client:
        return await client.get_shell(shell_id, encode=encode)


@app.tool()
@log_tool
async def create_shell(shell: dict, host: str = config.host) -> dict:
    async with AsyncClient(host=host) as client:
        return await client.create_shell(shell)


@app.tool()
@log_tool
async def update_shell(shell_id: str, shell: dict, encode: bool = True, host: str = config.host) -> None:
    async with AsyncClient(host=host) as client:
        await client.update_shell(shell_id, shell, encode=encode)


@app.tool()
@log_tool
async def delete_shell(shell_id: str, encode: bool = True, host: str = config.host) -> None:
    async with AsyncClient(host=host) as client:
        await client.delete_shell(shell_id, encode=encode)


# -----------------------------
# Tools: Submodel references
# -----------------------------

@app.tool()
@log_tool
async def get_submodel_refs(shell_id: str, encode: bool = True, host: str = config.host) -> dict:
    async with AsyncClient(host=host) as client:
        return await client.get_submodel_refs(shell_id, encode=encode)


@app.tool()
@log_tool
async def create_submodel_ref(shell_id: str, submodel_ref: dict, encode: bool = True, host: str = config.host) -> None:
    async with AsyncClient(host=host) as client:
        await client.create_submodel_ref(shell_id, submodel_ref, encode=encode)


@app.tool()
@log_tool
async def delete_submodel_ref(shell_id: str, submodel_id: str, encode: bool = True, host: str = config.host) -> None:
    async with AsyncClient(host=host) as client:
        await client.delete_submodel_ref(shell_id, submodel_id, encode=encode)


# -----------------------------
# Tools: Submodel management
# -----------------------------

@app.tool()
@log_tool
async def get_submodels(host: str = config.host) -> dict:
    async with AsyncClient(host=host) as client:
        return await client.get_submodels()


@app.tool()
@log_tool
async def get_submodel(submodel_id: str, encode: bool = True, host: str = config.host) -> dict:
    async with AsyncClient(host=host) as client:
        return await client.get_submodel(submodel_id, encode=encode)


@app.tool()
@log_tool
async def create_submodel(submodel: dict, host: str = config.host) -> dict:
    async with AsyncClient(host=host) as client:
        return await client.create_submodel(submodel)


@app.tool()
@log_tool
async def update_submodel(submodel_id: str, submodel: dict, encode: bool = True, host: str = config.host) -> None:
    async with AsyncClient(host=host) as client:
        await client.update_submodel(submodel_id, submodel, encode=encode)


@app.tool()
@log_tool
async def delete_submodel(submodel_id: str, encode: bool = True, host: str = config.host) -> None:
    async with AsyncClient(host=host) as client:
        await client.delete_submodel(submodel_id, encode=encode)


# -----------------------------
# Tools: Value operations
# -----------------------------

@app.tool()
@log_tool
async def get_submodel_value(submodel_id: str, encode: bool = True, host: str = config.host) -> dict:
    async with AsyncClient(host=host) as client:
        return await client.get_submodel_value(submodel_id, encode=encode)


@app.tool()
@log_tool
async def update_submodel_value(submodel_id: str, value: list[dict], encode: bool = True, host: str = config.host) -> None:
    async with AsyncClient(host=host) as client:
        await client.update_submodel_value(submodel_id, value, encode=encode)


@app.tool()
@log_tool
async def get_submodel_metadata(submodel_id: str, encode: bool = True, host: str = config.host) -> dict:
    async with AsyncClient(host=host) as client:
        return await client.get_submodel_metadata(submodel_id, encode=encode)


# -----------------------------
# Tools: Submodel element operations
# -----------------------------

@app.tool()
@log_tool
async def get_submodel_elements(submodel_id: str, encode: bool = True, host: str = config.host) -> dict:
    async with AsyncClient(host=host) as client:
        return await client.get_submodel_elements(submodel_id, encode=encode)


@app.tool()
@log_tool
async def create_submodel_element(
    submodel_id: str,
    element: dict,
    id_short_path: Optional[str] = None,
    encode: bool = True,
    host: str = config.host,
) -> None:
    async with AsyncClient(host=host) as client:
        await client.create_submodel_element(submodel_id, element, id_short_path, encode=encode)


@app.tool()
@log_tool
async def get_submodel_element(
    submodel_id: str,
    id_short_path: str,
    encode: bool = True,
    host: str = config.host,
) -> dict:
    async with AsyncClient(host=host) as client:
        return await client.get_submodel_element(submodel_id, id_short_path, encode=encode)


@app.tool()
@log_tool
async def update_submodel_element(
    submodel_id: str,
    id_short_path: str,
    element: dict,
    encode: bool = True,
    host: str = config.host,
) -> None:
    async with AsyncClient(host=host) as client:
        await client.update_submodel_element(submodel_id, id_short_path, element, encode=encode)


@app.tool()
@log_tool
async def delete_submodel_element(
    submodel_id: str,
    id_short_path: str,
    encode: bool = True,
    host: str = config.host,
) -> None:
    async with AsyncClient(host=host) as client:
        await client.delete_submodel_element(submodel_id, id_short_path, encode=encode)


@app.tool()
@log_tool
async def get_submodel_element_value(
    submodel_id: str,
    id_short_path: str,
    encode: bool = True,
    host: str = config.host,
) -> dict | list | str | int | float | bool | None:
    async with AsyncClient(host=host) as client:
        return await client.get_submodel_element_value(submodel_id, id_short_path, encode=encode)


@app.tool()
@log_tool
async def update_submodel_element_value(
    submodel_id: str,
    id_short_path: str,
    value: str,
    encode: bool = True,
    host: str = config.host,
) -> None:
    async with AsyncClient(host=host) as client:
        await client.update_submodel_element_value(submodel_id, id_short_path, value, encode=encode)


# -----------------------------
# Tools: Health monitoring
# -----------------------------

@app.tool()
@log_tool
async def get_health_status(host: str = config.host, timeout: float = config.timeout) -> str:
    async with AsyncClient(host=host, timeout=timeout) as client:
        return await client.get_health_status()


@app.tool()
@log_tool
async def is_healthy(host: str = config.host, timeout: float = config.timeout) -> bool:
    async with AsyncClient(host=host, timeout=timeout) as client:
        return await client.is_healthy()


# -----------------------------
# Entrypoints
# -----------------------------

async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger.info("Starting AAS MCP server (async main)")
    host = os.getenv("MCP_HTTP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_HTTP_PORT", "8000"))
    path = os.getenv("MCP_HTTP_PATH", "/mcp")

    logger.info("HTTP transport config host=%s port=%s path=%s default_host=%s", host, port, path, config.host)

    await app.run(
        transport="streamable-http",
        host=host,
        port=port,
        path=path,
    )


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


if __name__ == "__main__":
    cli_main()
