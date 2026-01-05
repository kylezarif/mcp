# MCP Agentic (LangGraph Client)

A LangGraph-based MCP client that connects to the existing MCP servers under `mcp/mcp-servers` (no servers duplicated here). It uses OpenAI for reasoning and the MCP stdio protocol for tool calls.

## Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) installed
- OpenAI API key in `mcp/mcp-agentic/.env` (`OPENAI_API_KEY`, optionally `OPENAI_BASE_URL`, `OPENAI_MODEL`)
- `~/.mcp/config.json` already pointing to your MCP servers (e.g., sec-filings, gdelt-news, macro-data, stooq-prices, fhfa-hpi)

## Setup (from repo root)
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e mcp/mcp-agentic
```

## Run (from repo root)
```bash
uv run --directory mcp/mcp-agentic python client.py
```
Or from inside `mcp/mcp-agentic`:
```bash
uv run python client.py
```

## How it works
- Reads `~/.mcp/config.json`, spawns each server via stdio, and registers their tools with the LLM using `server__tool` names (OpenAI-safe).
- Builds a LangGraph with a simple loop: LLM → (tool call, if any) → LLM → output.
- Maintains short-term memory in-process; `reset` clears it; restart clears all history.

## Sample prompts
- “List recent filings for CIK 0000320193.”
- “Search news about Tesla stock in the last week.”
- “Get recent Treasury yields.”
- “Fetch daily OHLCV for AAPL.US for the last 10 rows.”
- “Show FHFA HPI state sample rows.”

## Add a new server (reuse existing servers repo)
1) Build your server in `mcp/mcp-servers/<my-server>/` (see existing servers as templates).
2) Update `~/.mcp/config.json`:
   ```json
   "my-server": {
     "command": "uv",
     "args": [
       "--directory",
       "~/mcp/mcp-servers/my-server",
       "run",
       "my_server.py"
     ]
   }
   ```
3) Restart the agent: `uv run --directory mcp/mcp-agentic python client.py`.
