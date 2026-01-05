import os
import json
import asyncio
import logging
from contextlib import AsyncExitStack
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Environment Setup
load_dotenv(
    dotenv_path=Path(".env").expanduser(),
    override=True,
    verbose=True,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env file")


# MCP Client Class
TOOL_NAME_SEP = "__"

class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.client = None
        self.servers = {}
        # Persistent chat history (system + conversation messages)
        system_prompt = os.getenv(
            "MCP_SYSTEM_PROMPT",
            "You are an AI assistant integrated with MCP tool servers. Use tools when they help answer the user's queries."
        )
        self.messages = [{"role": "system", "content": system_prompt}]
        # Max number of non-system messages to keep (simple cap)
        self.max_history = int(os.getenv("MCP_MAX_HISTORY", "40"))

    def _trim_history(self):
        # Keep system message + last N other messages
        if len(self.messages) <= self.max_history + 1:
            return
        system = self.messages[0]
        tail = self.messages[-self.max_history:]
        self.messages = [system] + tail

    async def connect_to_all_servers(self):
        """Auto-load all MCP servers from ~/.mcp/config.json."""
        config_path = Path.home() / ".mcp" / "config.json"
        if not config_path.exists():
            logger.warning("No ~/.mcp/config.json found — starting in standalone mode.")
            return

        with open(config_path) as f:
            config = json.load(f)

        servers = config.get("mcpServers", {})
        if not servers:
            logger.warning("No servers defined in ~/.mcp/config.json")
            return

        for name, entry in servers.items():
            try:
                command = entry["command"]
                args = entry["args"]
                server_params = StdioServerParameters(command=command, args=args, env=None)

                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                read, write = stdio_transport
                session = await self.exit_stack.enter_async_context(ClientSession(read, write))
                await session.initialize()

                response = await session.list_tools()
                tool_names = [t.name for t in response.tools]
                logger.info(f" Connected to MCP server '{name}' with tools: {tool_names}")

                self.servers[name] = {
                    "session": session,
                    "tools": response.tools,
                }

            except Exception as e:
                logger.error(f" Failed to connect to server '{name}': {e}")

    async def process_query(self, query: str) -> str:
        """Send query to OpenAI and handle tool calls across all MCP servers with memory."""
        # Append user query to persistent history
        self.messages.append({"role": "user", "content": query})
        self._trim_history()
        available_tools = []

        # Collect all tools from connected servers
        for name, server in self.servers.items():
            for tool in server["tools"]:
                tool_call_name = f"{name}{TOOL_NAME_SEP}{tool.name}"
                available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool_call_name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                })

        # Ask OpenAI which tools to call
        completion = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=self.messages,
            tools=available_tools or None,
        )

        assistant_message = completion.choices[0].message
        final_text = assistant_message.content or ""

        # Keep the assistant’s message (with tool_calls if any) in persistent history
        self.messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": getattr(assistant_message, "tool_calls", None),
        })
        self._trim_history()

        # Handle tool calls
        if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
            for call in assistant_message.tool_calls:
                full_tool_name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                logger.info(f"Calling tool {full_tool_name} with args {args}")

                # Determine which server and tool
                server_name, tool_name = None, full_tool_name
                if TOOL_NAME_SEP in full_tool_name:
                    server_name, tool_name = full_tool_name.split(TOOL_NAME_SEP, 1)
                elif len(self.servers) == 1:
                    server_name = next(iter(self.servers))
                else:
                    logger.warning(f"Could not determine server for tool {tool_name}")
                    continue

                server_entry = self.servers.get(server_name)
                if not server_entry:
                    logger.error(f"No connected MCP server found for '{server_name}'")
                    continue

                try:
                    result = await server_entry["session"].call_tool(tool_name, args)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": f"{server_name}.{tool_name}",
                        "content": result.content,
                    })
                except Exception as e:
                    logger.error(f"Tool call failed for {server_name}.{tool_name}: {e}")
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": f"{server_name}.{tool_name}",
                        "content": f"Error calling {server_name}.{tool_name}: {e}",
                    })
            self._trim_history()

            # Ask OpenAI for a final completion including tool outputs
            completion = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=self.messages,
                tools=available_tools,
            )
            final_assistant = completion.choices[0].message
            final_text = final_assistant.content or final_text
            # Append final assistant response (without duplicating tool calls)
            self.messages.append({"role": "assistant", "content": final_assistant.content})
            self._trim_history()

        return final_text

    async def chat_loop(self):
        """Run an interactive CLI chat."""
        print("\n OpenAI MCP Client Started!")
        print("Type your queries or 'quit' to exit.\n")

        while True:
            try:
                query = input("You: ").strip()
                if query.lower() in {"quit", "exit"}:
                    break
                if query.lower() == "reset":
                    # Preserve system prompt only
                    system = self.messages[0] if self.messages and self.messages[0]["role"] == "system" else None
                    self.messages = [system] if system else []
                    print("\nMemory cleared.\n")
                    continue
                response = await self.process_query(query)
                print(f"\nAssistant: {response}\n")
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()


# Main Entry Point
async def main():
    client = MCPClient()
    openai_kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        openai_kwargs["base_url"] = OPENAI_BASE_URL
    client.client = OpenAI(**openai_kwargs)

    print(" Connecting to all MCP servers from ~/.mcp/config.json...")
    await client.connect_to_all_servers()

    # Summarize connection results
    if client.servers:
        connected_names = ", ".join(client.servers.keys())
        print(
            f"\n Retrieved context from MCP servers successfully"
            f"and integrated with the LLM.\n"
            f" Using LLM-powered chatbot client: OpenAI MCP Client\n"
            f" Servers available to MCP Client: {connected_names}\n"
        )
    else:
        print(
            "\n No MCP servers connected. Running in standalone chat mode "
            "with OpenAI.\n"
        )

    print("Ready! Type queries below.\n")

    try:
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
