"""Convenience launcher for the LangGraph MCP agent."""

import asyncio

from mcp_agentic.client import main


if __name__ == "__main__":
    asyncio.run(main())
