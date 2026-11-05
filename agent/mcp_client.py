"""MCP client: starts the server over stdio and forwards tool calls (used by the agent and the baseline)."""
from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class McpTools:
    def __init__(self, session: ClientSession):
        self._s = session
        self.tools: list[dict] = []

    async def load(self) -> None:
        listed = await self._s.list_tools()
        self.tools = [{"name": t.name, "description": t.description or "", "input_schema": t.inputSchema}
                      for t in listed.tools]

    async def call(self, name: str, args: dict) -> dict:
        """Call a tool; always returns a dict (server errors are structured dicts too)."""
        try:
            res = await self._s.call_tool(name, args)
        except Exception as e:
            return {"ok": False, "error": {"type": "tool_call_failed", "message": str(e)}}
        if res.structuredContent is not None:
            return res.structuredContent
        text = res.content[0].text if res.content else ""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"ok": False, "error": {"type": "tool_error", "message": text}}


@asynccontextmanager
async def open_tools():
    params = StdioServerParameters(command=sys.executable, args=["-m", "server.server"])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = McpTools(session)
            await tools.load()
            yield tools
