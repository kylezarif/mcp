"""Thin wrapper to run the packaged MCP client when invoked as a script."""

import asyncio

from mcp_client.client import main


if __name__ == "__main__":
    asyncio.run(main())
