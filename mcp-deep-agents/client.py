"""Launcher for the deep-agents MCP client."""

import asyncio

from mcp_deep_agents.client import main


if __name__ == "__main__":
    asyncio.run(main())
