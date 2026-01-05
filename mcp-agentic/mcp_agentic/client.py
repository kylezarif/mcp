import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load env (.env preferred in this directory; fall back to repo-level mcp/.env and mcp-client/.env)
here = Path(__file__).resolve().parent
load_dotenv(here / ".env", override=True)
load_dotenv(here.parent / ".env", override=True)
load_dotenv(here.parent / "mcp-client" / ".env", override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
SYSTEM_PROMPT = os.getenv(
    "MCP_SYSTEM_PROMPT",
    "You are an AI assistant integrated with MCP tool servers. Use tools when they help answer the user's queries.",
)
MAX_HISTORY = int(os.getenv("MCP_MAX_HISTORY", "40"))
TOOL_NAME_SEP = "__"  # safe for OpenAI tool name schema


class AgentState(dict):
    """Simple state container for LangGraph."""

    messages: List[Any]
    available_tools: List[Dict[str, Any]]
    tool_sessions: Dict[str, Any]


class MCPAgent:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.tool_sessions: Dict[str, Dict[str, Any]] = {}
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0,
        )
        self._graph = None

    def _tool_result_to_text(self, result) -> str:
        c = getattr(result, "content", result)
        if c is None:
            return ""
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = []
            for item in c:
                if isinstance(item, dict):
                    parts.append(item.get("text") or json.dumps(item, ensure_ascii=False))
                else:
                    txt = getattr(item, "text", None)
                    parts.append(txt if isinstance(txt, str) else str(item))
            return "\n".join(parts)
        return str(c)

    async def connect_servers(self):
        """Load servers from ~/.mcp/config.json and initialize sessions."""
        config_path = Path.home() / ".mcp" / "config.json"
        if not config_path.exists():
            logger.warning("No ~/.mcp/config.json found — starting without servers.")
            return
        with open(config_path) as f:
            config = json.load(f)
        servers = config.get("mcpServers", {})
        if not servers:
            logger.warning("No servers defined in ~/.mcp/config.json")
            return

        for name, entry in servers.items():
            try:
                cmd, args = entry["command"], entry["args"]
                params = StdioServerParameters(command=cmd, args=args, env=None)
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
                read, write = stdio_transport
                session = await self.exit_stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                response = await session.list_tools()
                tools = response.tools
                logger.info("Connected to %s with tools: %s", name, [t.name for t in tools])
                self.tool_sessions[name] = {"session": session, "tools": tools}
            except Exception as exc:
                logger.error("Failed to connect to server %s: %s", name, exc)

    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        for server_name, server in self.tool_sessions.items():
            for tool in server["tools"]:
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": f"{server_name}{TOOL_NAME_SEP}{tool.name}",
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )
        return tools

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph agent graph."""
        workflow = StateGraph(AgentState)

        async def call_llm(state: AgentState, config: Optional[RunnableConfig] = None):
            messages = state["messages"][-(MAX_HISTORY + 1) :]
            available_tools = state["available_tools"]
            response = await self.llm.ainvoke(
                messages,
                tools=available_tools if available_tools else None,
                tool_choice="auto",
                config=config,
            )
            tc = getattr(response, "tool_calls", None)
            if tc:
                tool_names = []
                for call in tc:
                    name = None
                    fn = getattr(call, "function", None)
                    if fn is not None:
                        name = getattr(fn, "name", None)
                        if name is None and isinstance(fn, dict):
                            name = fn.get("name")
                    if name is None:
                        name = getattr(call, "name", None)
                    if name is None and hasattr(call, "get"):
                        name = (call.get("function") or {}).get("name") or call.get("name")
                    tool_names.append(name)
                logger.info("LLM requested tools: %s", tool_names)
            new_messages = state["messages"] + [response]
            return {"messages": new_messages, "available_tools": available_tools, "tool_sessions": state["tool_sessions"]}

        async def maybe_tool(state: AgentState, config: Optional[RunnableConfig] = None):
            last = state["messages"][-1]
            tool_calls = getattr(last, "tool_calls", None)
            if not tool_calls:
                return {"messages": state["messages"], "available_tools": state["available_tools"], "tool_sessions": state["tool_sessions"]}
            tool_msgs: List[Any] = []
            for call in tool_calls:
                # Support both dict-like and object-like tool calls
                fn = getattr(call, "function", None)
                if fn is None and hasattr(call, "get"):
                    fn = call.get("function")
                # Some SDKs surface name/arguments directly on the call
                full_name = (
                    getattr(fn, "name", None)
                    or (fn.get("name") if isinstance(fn, dict) else None)
                    or getattr(call, "name", None)
                    or (call.get("name") if hasattr(call, "get") else None)
                )
                call_id = getattr(call, "id", None) or (call.get("id") if hasattr(call, "get") else "unknown")
                if not full_name:
                    tool_msgs.append(
                        ToolMessage(
                            content="Tool call missing function name.",
                            name="unknown",
                            tool_call_id=call_id,
                        )
                    )
                    continue
                server_name, tool_name = full_name.split(TOOL_NAME_SEP, 1)
                args_raw = getattr(fn, "arguments", None) if fn else None
                if args_raw is None and isinstance(fn, dict):
                    args_raw = fn.get("arguments")
                if args_raw is None:
                    # Fallback to call.args / call.arguments if present
                    args_raw = getattr(call, "arguments", None) or getattr(call, "args", None)
                args = json.loads(args_raw or "{}")
                # Heuristic: if ECB series is missing args, try to parse last user message with flow/key pattern
                if tool_name == "get_ecb_series" and not args:
                    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
                    if last_user:
                        txt = (last_user.content or "").strip()
                        if "/" in txt:
                            flow_ref, key_val = txt.split("/", 1)
                            args = {"flow_ref": flow_ref.strip(), "key": key_val.strip().rstrip(".")}
                            logger.info("Heuristically filled get_ecb_series args=%s from user text", args)
                if tool_name in {"company_news", "search_news"} and not args:
                    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
                    if last_user:
                        q = (last_user.content or "").strip().rstrip(".")
                        # If the user said "search ... about X", focus on the segment after "about"
                        if " about " in q.lower():
                            try:
                                q = q.split(" about ", 1)[1]
                            except Exception:
                                pass
                        if tool_name == "company_news":
                            args = {"name_or_ticker": q}
                        else:
                            args = {"query": q}
                        logger.info("Heuristically filled news args=%s from user text", args)
                if tool_name == "get_daily_ohlcv" and not args:
                    # Try to extract a ticker like AAPL.US from the last user message, plus optional dates
                    import re

                    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
                    if last_user:
                        txt = (last_user.content or "").strip()
                        match = re.search(r"\b([A-Za-z]{1,10}\.[A-Za-z]{2,4})\b", txt)
                        dates = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", txt)
                        start_date = dates[0] if len(dates) >= 1 else None
                        end_date = dates[1] if len(dates) >= 2 else None
                        if match:
                            args = {"symbol": match.group(1).lower()}
                            if start_date:
                                args["start_date"] = start_date
                            if end_date:
                                args["end_date"] = end_date
                            logger.info("Heuristically filled get_daily_ohlcv args=%s from user text", args)
                if tool_name in {"get_recent_filings", "get_company_submissions", "get_company_facts"} and not args:
                    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
                    if last_user:
                        import re

                        digits = re.findall(r"\b(\d{5,10})\b", last_user.content or "")
                        if digits:
                            cik = digits[0].zfill(10)
                            args = {"cik": cik}
                            logger.info("Heuristically filled SEC args=%s from user text", args)
                server_entry = state["tool_sessions"].get(server_name)
                if not server_entry:
                    tool_msgs.append(
                        ToolMessage(
                            content=f"Server {server_name} not connected.",
                            name=full_name,
                            tool_call_id=call_id,
                        )
                    )
                    continue
                try:
                    logger.info("Calling tool %s args=%s", full_name, args)
                    result = await server_entry["session"].call_tool(tool_name, args)
                    logger.info(
                        "Tool %s returned isError=%s content=%r",
                        full_name,
                        getattr(result, "isError", None),
                        getattr(result, "content", None),
                    )
                    is_error = bool(getattr(result, "isError", False))
                    text = self._tool_result_to_text(result)
                    if (
                        "No debt data returned." in text
                        or "No FHFA HPI data returned" in text
                        or "No results or failed to fetch news." in text
                    ):
                        is_error = True
                    if is_error:
                        text = f"[TOOL ERROR] {text}"
                    tool_msgs.append(
                        ToolMessage(
                            content=text,
                            name=full_name,
                            tool_call_id=call_id,
                        )
                    )
                except Exception as exc:
                    tool_msgs.append(
                        ToolMessage(
                            content=f"Error calling {full_name}: {exc}",
                            name=full_name,
                            tool_call_id=call_id,
                        )
                    )
            new_messages = state["messages"] + tool_msgs
            return {"messages": new_messages, "available_tools": state["available_tools"], "tool_sessions": state["tool_sessions"]}

        async def normalize_tools(state: AgentState, config: Optional[RunnableConfig] = None):
            """Ensure tool messages carry structured JSON strings to stabilize LLM formatting."""
            normalized: List[Any] = []
            for msg in state["messages"]:
                if isinstance(msg, ToolMessage):
                    content = msg.content
                    if not isinstance(content, str):
                        try:
                            content = json.dumps(content, ensure_ascii=False)
                        except Exception:
                            content = str(content)
                    else:
                        # If not valid JSON, wrap in a JSON envelope
                        try:
                            json.loads(content)
                        except Exception:
                            content = json.dumps({"tool": msg.name, "content": content}, ensure_ascii=False)
                    normalized.append(
                        ToolMessage(
                            content=content,
                            name=msg.name,
                            tool_call_id=msg.tool_call_id,
                        )
                    )
                else:
                    normalized.append(msg)
            return {"messages": normalized, "available_tools": state["available_tools"], "tool_sessions": state["tool_sessions"]}

        workflow.add_node("llm", call_llm)
        workflow.add_node("tool", maybe_tool)
        workflow.add_node("normalize_tools", normalize_tools)

        def route_llm(state: AgentState):
            last = state["messages"][-1]
            return "tool" if getattr(last, "tool_calls", None) else "end"

        workflow.add_conditional_edges("llm", route_llm, {"tool": "tool", "end": END})
        workflow.add_edge("tool", "normalize_tools")
        workflow.add_edge("normalize_tools", "llm")
        workflow.set_entry_point("llm")
        return workflow

    async def run(self):
        await self.connect_servers()
        tools = self._build_tool_schemas()
        # initial state with system prompt
        messages: List[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
        state = AgentState(messages=messages, available_tools=tools, tool_sessions=self.tool_sessions)
        self._graph = self._build_graph().compile()

        print("\nLangGraph MCP Agent started. Type 'quit' to exit.\n")
        while True:
            user_text = input("You: ").strip()
            if user_text.lower() in {"quit", "exit"}:
                break
            if user_text.lower() == "reset":
                messages = [SystemMessage(content=SYSTEM_PROMPT)]
                state = AgentState(messages=messages, available_tools=tools, tool_sessions=self.tool_sessions)
                print("Memory reset.\n")
                continue

            state["messages"].append(HumanMessage(content=user_text))
            # Run the graph to completion and capture the resulting state
            state = await self._graph.ainvoke(state)

            # Show the latest AI message if present
            # Prefer the newest AIMessage after the last user turn
            ai_msgs = [m for m in state["messages"] if isinstance(m, AIMessage)]
            if ai_msgs:
                last_ai = ai_msgs[-1]
                content = last_ai.content
                if content:
                    print(f"\nAssistant: {content}\n")

    async def close(self):
        await self.exit_stack.aclose()


async def main():
    agent = MCPAgent()
    try:
        await agent.run()
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
