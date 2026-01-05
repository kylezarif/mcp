# MCP Deep Agents

A LangGraph-based “deep agent” client that adds planning (TODOs), scratch files, and MCP tool execution. It connects to the existing MCP servers under `mcp/mcp-servers` and uses OpenAI for reasoning.

## Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) installed
- OpenAI API key in `.env` (place in `mcp/mcp-deep-agents/.env` or `mcp/.env`): `OPENAI_API_KEY`, optionally `OPENAI_BASE_URL`, `OPENAI_MODEL`
- `~/.mcp/config.json` pointing to your MCP servers (sec-filings, gdelt-news, macro-data, stooq-prices, fhfa-hpi)

## Install (from repo root)
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e mcp/mcp-deep-agents
```

## Run
From repo root:
```bash
uv run --directory mcp/mcp-deep-agents python client.py
```
From inside `mcp/mcp-deep-agents`:
```bash
uv run python client.py
```

## How it works
- Connects to MCP servers from `~/.mcp/config.json` (stdio).
- Builds a LangGraph with planning + tool loop:
  - `plan` node: generate TODOs for the user request, save to `runs/<timestamp>/plan.json`.
  - `llm` node: reason with available MCP tools (named `server__tool`).
  - `tool` node: execute MCP tools, heuristically fill missing args (CIK, tickers, dates, news queries).
  - `normalize_tools` node: wraps tool outputs as JSON strings for stability; tool outputs are also saved to `runs/<timestamp>/`.
- Memory: in-process rolling message history (bounded by `MCP_MAX_HISTORY`) that includes the system prompt; `reset` clears state and starts a new run directory.

## Why use this client
- Structured workflow (plan → act → summarize) rather than a single loop.
- Filesystem scratchpad: tool outputs and plans are written to `runs/<timestamp>/`.
- Heuristics for common args (tickers, dates, CIKs, ECB series patterns, news queries).
- Normalized tool outputs reduce formatting issues with the LLM.

## Notes on MCP servers
The client reuses the existing finance/keyless servers in `mcp/mcp-servers`:
- `sec-filings`: SEC EDGAR submissions/facts.
- `gdelt-news`: GDELT news search.
- `macro-data`: Treasury (debt, yields), World Bank, ECB.
- `stooq-prices`: Historical OHLCV (Stooq).
- `fhfa-hpi`: FHFA House Price Index CSVs.

Ensure `~/.mcp/config.json` points to those server script paths, as in the main `mcp/README.md`.
