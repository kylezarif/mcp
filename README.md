# MCP Workspace

This workspace includes two MCP clients and five finance-focused MCP servers that use free/keyless APIs. Both clients talk to OpenAI and can call tools exposed by the servers discovered from `~/.mcp/config.json`.

- `mcp/mcp-client`: simple single-loop client.
- `mcp/mcp-agentic`: LangGraph-based client with explicit graph/state handling and tool normalization.

## Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- OpenAI API key in `mcp/mcp-client/.env` (set `OPENAI_API_KEY`, optionally `OPENAI_BASE_URL` and `OPENAI_MODEL`)

## One-time setup
From repo root:
```bash
# Create a virtual env with uv and activate it
uv venv .venv
source .venv/bin/activate

# Install the classic client dependencies
uv pip install -e mcp/mcp-client
```
If you are already inside `mcp/mcp-client`, install with:
```bash
uv pip install -e .
```

Agentic client:
```bash
uv pip install -e mcp/mcp-agentic
```
If you are already inside `mcp/mcp-agentic`, install with:
```bash
uv pip install -e .
```

## Configure servers
`~/.mcp/config.json` should point to the servers in this repo, for example:
```json
{
  "mcpServers": {
    "sec-filings": {
      "command": "uv",
      "args": [
        "--directory",
        "~/mcp/mcp-servers/sec-filings",
        "run",
        "sec_filings.py"
      ]
    },
    "gdelt-news": {
      "command": "uv",
      "args": [
        "--directory",
        "~/mcp/mcp-servers/gdelt-news",
        "run",
        "gdelt_news.py"
      ]
    },
    "macro-data": {
      "command": "uv",
      "args": [
        "--directory",
        "~/mcp/mcp-servers/macro-data",
        "run",
        "macro_data.py"
      ]
    },
    "stooq-prices": {
      "command": "uv",
      "args": [
        "--directory",
        "~/mcp/mcp-servers/stooq-prices",
        "run",
        "stooq_prices.py"
      ]
    },
    "fhfa-hpi": {
      "command": "uv",
      "args": [
        "--directory",
        "~/mcp/mcp-servers/fhfa-hpi",
        "run",
        "fhfa_hpi.py"
      ]
    }
  }
}
```

## Included servers and sample validation questions
- **sec-filings** (SEC EDGAR submissions/facts, keyless):  
  - “List recent filings for CIK 0000320193.”  
  - “Fetch company facts keys for CIK 0000789019.”  
  - “Summarize submissions for CIK 0001652044.”
- **gdelt-news** (GDELT DOC API, keyless):  
  - “Search news about Tesla stock in the last week.”  
  - “Company news for NVDA.”  
  - “Topic trends for inflation.”
- **macro-data** (Treasury Fiscal Data, World Bank, ECB, keyless):  
  - “Get recent debt outstanding.”  
  - “Get recent Treasury yields.”  
  - “World Bank indicator FP.CPI.TOTL for USA, 2015:2023.”  
  - “ECB series EXR/M.USD.EUR.SP00.A.”
- **stooq-prices** (Stooq historical OHLCV, keyless):  
  - “Get daily OHLCV for AAPL.US for the last 10 rows.”  
  - “Fetch daily OHLCV for spy.us between 2024-01-01 and 2024-02-01.”
- **fhfa-hpi** (FHFA House Price Index CSVs, keyless):  
  - “Show FHFA HPI state sample rows.”  
  - “Show FHFA HPI metro sample rows.”

## Architecture (high level)
```
Classic client (mcp-client)
┌────────────────────────┐   messages (system prompt + memory window)   ┌────────────────────┐
│      OpenAI LLM        │ <------------------------------------------> │  MCP Client        │
│  (chat + tool calling) │   tools advertised via server__tool schemas  │  (mcp-client/)     │
└────────────────────────┘                                              └────────────┬───────┘
                                                                                     │
                                           ~/.mcp/config.json declares servers       │ stdio
                                                                                     │
     ┌────────────────┬─────────────┬──────────────┬───────────────┬──────────────┐
     │ sec-filings    │ gdelt-news  │ macro-data   │ stooq-prices  │ fhfa-hpi     │
     │ (SEC EDGAR)    │ (GDELT DOC) │ (Treasury,   │ (Stooq CSV)   │ (FHFA HPI)   │
     │                │             │  WB, ECB)    │               │              │
     └────────────────┴─────────────┴──────────────┴───────────────┴──────────────┘

Agentic client (mcp-agentic / LangGraph)
┌────────────────────────┐   messages (system prompt + memory window)   ┌──────────────────────────┐
│      OpenAI LLM        │ <------------------------------------------> │  LangGraph MCP Agent     │
│  (chat + tool calling) │   tools advertised via server__tool schemas  │  (llm → tool → normalize │
└────────────────────────┘                                              │   → llm)                 │
                                                                       └──────────────┬───────────┘
                                                                                      │
                                           ~/.mcp/config.json declares servers        │ stdio
                                                                                      │
     ┌────────────────┬─────────────┬──────────────┬───────────────┬──────────────┐
     │ sec-filings    │ gdelt-news  │ macro-data   │ stooq-prices  │ fhfa-hpi     │
     │ (SEC EDGAR)    │ (GDELT DOC) │ (Treasury,   │ (Stooq CSV)   │ (FHFA HPI)   │
     │                │             │  WB, ECB)    │               │              │
     └────────────────┴─────────────┴──────────────┴───────────────┴──────────────┘

Memory/prompt: both clients keep a rolling history (configurable via `MCP_MAX_HISTORY`) that includes the system prompt. `reset` clears in-memory history; restart clears all state. Tools are normalized as `server__tool` for OpenAI compatibility. The agentic client also normalizes tool outputs into JSON strings to stabilize formatting between turns.
```

## Add a new server (step by step)
1) Create the server folder under `mcp/mcp-servers/<server-name>/`.
2) Add a `pyproject.toml` (use an existing server as a template) and a `<server_name>.py` with `FastMCP` tools.
3) (Optional) `uv pip install -e .` inside that server folder to ensure dependencies resolve.
4) Update `~/.mcp/config.json` to include the new server entry:
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
   Make sure the path and script name match your server.
5) Restart the MCP client (`uv run --directory mcp/mcp-client python client.py`). It will auto-discover tools from the new server.

## Run the MCP client (no `cd` required)
```bash
uv run --directory mcp/mcp-client python client.py
```
If you are inside `mcp/mcp-client`, the equivalent is:
```bash
uv run python client.py
```
This will connect to the configured MCP servers, let you chat, and route tool calls as needed.

## Run the LangGraph agentic client
From repo root:
```bash
uv run --directory mcp/mcp-agentic python client.py
```
From inside `mcp/mcp-agentic`:
```bash
uv run python client.py
```
It uses the same servers from `~/.mcp/config.json`, applies the same prompt/memory window, and drives tool calls through a LangGraph loop (LLM → tools → LLM).

## Benefits at a glance
- **mcp-client (classic):** minimal dependencies, straightforward control flow, good for quick CLI use and debugging tool availability.
- **mcp-agentic (LangGraph):** explicit graph (LLM → tool → normalization → LLM), safer tool name handling, heuristics to infer missing args (tickers, dates, CIKs), and tool output normalization to stabilize responses.
