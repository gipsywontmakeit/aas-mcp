"""
AAS MCP server — CUD operations for Asset Administration Shells.

READ operations have been moved to native LangChain tools (agents/tools/aas/)
for better performance (no MCP overhead, in-process caching, anti-loop).

This MCP server now handles only CUD (Create, Update, Delete) operations
that benefit from external governance, audit logging, and access control.

TODO: Add authentication middleware when ready.
"""

import logging
import os

from fastmcp import FastMCP
from shellsmith.clients import AsyncClient
from shellsmith.config import config

logger = logging.getLogger("aas_mcp")

app = FastMCP(
    name="aas-mcp",
    instructions="""
This server provides CUD (Create, Update, Delete) operations for Asset
Administration Shells (AAS) in Eclipse BaSyx environments.

READ operations (search, describe, get elements) are handled by native
LangChain tools and are NOT available here.

IMPORTANT (BaSyx encoding):
- Pass shell_id / submodel_id as the canonical ID (URL).
- This server ALWAYS calls BaSyx endpoints with encode=True.
""",
)


# =====================================================================
# Shell CUD operations
# =====================================================================

@app.tool()
async def create_shell(shell_data: dict, host: str = config.host) -> dict:
    """Create a new Asset Administration Shell."""
    async with AsyncClient(host=host) as client:
        return await client.create_shell(shell_data)


@app.tool()
async def update_shell(shell_id: str, shell_data: dict, host: str = config.host) -> dict:
    """Update an existing Asset Administration Shell."""
    async with AsyncClient(host=host) as client:
        return await client.update_shell(shell_id, shell_data, encode=True)


@app.tool()
async def delete_shell(shell_id: str, host: str = config.host) -> dict:
    """Delete an Asset Administration Shell."""
    async with AsyncClient(host=host) as client:
        return await client.delete_shell(shell_id, encode=True)
        return {"deleted": True, "shell_id": shell_id}


# =====================================================================
# Submodel CUD operations
# =====================================================================

@app.tool()
async def create_submodel(submodel_data: dict, host: str = config.host) -> dict:
    """Create a new Submodel."""
    async with AsyncClient(host=host) as client:
        return await client.create_submodel(submodel_data)


@app.tool()
async def update_submodel(submodel_id: str, submodel_data: dict, host: str = config.host) -> dict:
    """Update an existing Submodel."""
    async with AsyncClient(host=host) as client:
        return await client.update_submodel(submodel_id, submodel_data, encode=True)


@app.tool()
async def delete_submodel(submodel_id: str, host: str = config.host) -> dict:
    """Delete a Submodel."""
    async with AsyncClient(host=host) as client:
        return await client.delete_submodel(submodel_id, encode=True)
        return {"deleted": True, "submodel_id": submodel_id}


# =====================================================================
# Submodel element CUD operations
# =====================================================================

@app.tool()
async def create_submodel_element(
    submodel_id: str,
    element_data: dict,
    host: str = config.host,
) -> dict:
    """Create a new element in a Submodel."""
    async with AsyncClient(host=host) as client:
        return await client.create_submodel_element(submodel_id, element_data, encode=True)


@app.tool()
async def update_submodel_element(
    submodel_id: str,
    id_short_path: str,
    element_data: dict,
    host: str = config.host,
) -> dict:
    """Update an existing Submodel element by path."""
    id_short_path = id_short_path.replace(".", "/")
    async with AsyncClient(host=host) as client:
        return await client.update_submodel_element(
            submodel_id, id_short_path, element_data, encode=True
        )


@app.tool()
async def update_submodel_element_value(
    submodel_id: str,
    id_short_path: str,
    value,
    host: str = config.host,
) -> dict:
    """Update only the value of a Submodel element."""
    id_short_path = id_short_path.replace(".", "/")
    async with AsyncClient(host=host) as client:
        return await client.update_submodel_element_value(
            submodel_id, id_short_path, value, encode=True
        )


@app.tool()
async def delete_submodel_element(
    submodel_id: str,
    id_short_path: str,
    host: str = config.host,
) -> dict:
    """Delete a Submodel element by path."""
    id_short_path = id_short_path.replace(".", "/")
    async with AsyncClient(host=host) as client:
        return await client.delete_submodel_element(
            submodel_id, id_short_path, encode=True
        )
        return {"deleted": True, "submodel_id": submodel_id, "path": id_short_path}


# =====================================================================
# Submodel reference CUD operations
# =====================================================================

@app.tool()
async def add_submodel_reference(
    shell_id: str,
    submodel_id: str,
    host: str = config.host,
) -> dict:
    """Add a submodel reference to a shell (link a submodel to a shell)."""
    ref_data = {
        "type": "ModelReference",
        "keys": [{"type": "Submodel", "value": submodel_id}],
    }
    async with AsyncClient(host=host) as client:
        return await client.add_submodel_reference(shell_id, ref_data, encode=True)


@app.tool()
async def remove_submodel_reference(
    shell_id: str,
    submodel_id: str,
    host: str = config.host,
) -> dict:
    """Remove a submodel reference from a shell (unlink a submodel)."""
    async with AsyncClient(host=host) as client:
        return await client.remove_submodel_reference(
            shell_id, submodel_id, encode=True
        )
        return {"removed": True, "shell_id": shell_id, "submodel_id": submodel_id}


# =====================================================================
# Entrypoint
# =====================================================================

def cli_main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger.info("Starting AAS MCP server (CUD only)")
    host = os.getenv("MCP_HTTP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_HTTP_PORT", "8000"))
    path = os.getenv("MCP_HTTP_PATH", "/mcp")

    logger.info("HTTP transport: host=%s port=%s path=%s", host, port, path)

    app.run(
        transport="streamable-http",
        host=host,
        port=port,
        path=path,
    )