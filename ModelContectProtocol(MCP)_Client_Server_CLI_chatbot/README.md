## MCP Client + Server Chatbot (MCP + Amazon Bedrock)

Production-style CLI chatbot demonstrating a complete MCP ecosystem: a first-class MCP server and an MCP-aware client, wired into Amazon Bedrock for model reasoning and tool-use. It showcases real-world patterns like document resources, model-invoked tools, user-invoked prompts, @mentions, and slash-command UX.

### Why this project matters
- Built both sides: client and server — understand the full protocol and message flow
- Implements the three MCP primitives end-to-end: tools (model-controlled), resources (app-controlled), prompts (user-controlled)
- Real UX patterns: @document mentions, Tab-completed slash commands, autosuggestions
- Bedrock integration: routes tool-augmented messages to Anthropic models via `converse`

---

## Features

- **Amazon Bedrock chat orchestration**: Robust message loop with tool-use handling
- **MCP Server (FastMCP)** exposing:
  - Tools: `read_doc_contents`, `edit_document`
  - Resources: `docs://documents`, `docs://documents/{doc_id}`
  - Prompts: `/format` (Markdown rewriter)
- **CLI UX** powered by `prompt_toolkit`:
  - `@mentions` to inject document content
  - `/<command>` slash prompts with Tab completion and inline suggestions
  - Smooth keyboard bindings for `/` and `@`
- **Extensible multi-server**: Spin up additional MCP servers and auto-aggregate their tools

---

## Architecture

- `mcp_server.py`: FastMCP server defining tools, resources, and prompts
- `mcp_client.py`: Async client wrapper that launches and speaks MCP over stdio
- `core/bedrock.py`: Thin Bedrock runtime client; converts MCP types to Bedrock-compatible schemas
- `core/chat.py`: Model conversation loop including tool-use detection and execution
- `core/tools.py`: Tool discovery and execution across multiple MCP clients
- `core/cli_chat.py`: CLI agent; handles @mentions, `/command` prompts, and message shaping
- `core/cli.py`: Rich CLI with completion, suggestions, and keybindings
- `main.py`: App bootstrap; loads env, starts MCP servers, launches CLI

Data flow at a glance:
1) CLI reads input → 2) CLI agent enriches with docs/prompts → 3) Bedrock model responds (possibly with tool calls) → 4) Tool manager executes against MCP server(s) → 5) Responses fed back to the model → 6) Final answer printed.

---

## Demo Commands

- **Ask a question**
  - Type a plain question and press Enter

- **Mention documents**
  - Example: `Tell me about @deposition.md`
  - The CLI injects the document contents into the prompt automatically

- **Use a prompt (slash command)**
  - Example: `/format deposition.md`
  - Tab completion shows available commands and required arguments

---

## Prerequisites

- Python 3.8+
- AWS account with Bedrock access and credentials configured
  - Ensure your default credentials/profile can call Bedrock Runtime in your region

---

## Setup

### 1) Configure environment

Create a `.env` file in the project root:

```
BEDROCK_REGION="us-west-2"
BEDROCK_MODEL_ID="us.anthropic.claude-3-7-sonnet-20250219-v1:0"
# Optional: set to 1 to prefer uv for subprocess servers
USE_UV="0"
```

Make sure the region and model are valid for your AWS account.

### 2) Install dependencies

Option A — with uv (recommended)

```bash
pip install uv
uv venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
uv pip install boto3==1.37.38 python-dotenv prompt-toolkit "mcp[cli]==1.6.0"
```

Option B — standard pip

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install boto3==1.37.38 python-dotenv prompt-toolkit "mcp[cli]==1.6.0"
```

### 3) Run

```bash
# Using uv if USE_UV=1 in .env
uv run main.py

# Or with Python directly
python main.py
```

The app will start the MCP server, connect the client, and launch the interactive CLI.

---

## Usage Tips

- Press `/` at the start of the line to see available commands via completion
- Type `@` to autocomplete document ids
- After `/format `, press Tab to autocomplete the required `doc_id`

Example session:

```
> What does @report.pdf say about the condenser tower?
> /format deposition.md
```

---

## Extending the System

- Add or edit documents: update the `docs` dictionary in `mcp_server.py`
- Create new tools: decorate functions with `@mcp.tool(...)`
- Expose new resources: use `@mcp.resource("scheme://path")`
- Add prompts (slash commands): add `@mcp.prompt(...)` handlers
- Add more MCP servers: run with extra script names `python main.py another_mcp.py` (tools are auto-aggregated)

Note: A `TODO` in `mcp_server.py` hints at adding a `/summarize`-style prompt.

---

## Project Structure

```
core/
  bedrock.py        # Bedrock chat client + conversions
  chat.py           # Chat loop with tool-use handling
  cli.py            # CLI shell: completion, suggestions, keybindings
  cli_chat.py       # CLI agent: prompts, mentions, message shaping
  tools.py          # Aggregated tool execution
mcp_client.py       # MCP client wrapper (stdio)
mcp_server.py       # FastMCP server: tools/resources/prompts
main.py             # Entry: wiring + startup
```

---

## Troubleshooting

- Bedrock errors: verify AWS creds, `BEDROCK_REGION`, `BEDROCK_MODEL_ID`
- Command not found: ensure virtualenv is activated and dependencies installed
- Windows stdio issues: the app sets an async policy for Windows automatically
- No tools found: confirm the MCP server started (it runs in-process via stdio)

---

## Tech Stack

- Python, `prompt_toolkit`, `pydantic`
- MCP: `mcp[cli]==1.6.0`, `fastmcp`
- AWS: `boto3` (Bedrock Runtime)

---

## License

MIT — feel free to use, modify, and share.

---

## What I built and learned

- Implemented a full MCP client-server system and wired it into Bedrock
- Used tools for model-controlled actions, resources for app-controlled data, prompts for user-controlled flows
- Designed a CLI with @mentions and /commands to mirror modern chat UX
- Practiced protocol-level debugging using an MCP mindset and structured tracing
