from pathlib import Path

OPENAI_API_KEY = "API_KEY"
KEY_OWNER = "NAME"


DEBUG = False

MAX_CHUNK_SIZE = 4

LLM_VERS = "gpt-4o-mini"

# Which backend answers LLM calls:
#   "openai"     -- OpenAI API with OPENAI_API_KEY and LLM_VERS (original setup)
#   "claude_cli" -- Claude subscription through the `claude` CLI (Claude Code)
#   "codex_cli"  -- ChatGPT subscription through the `codex` CLI (Codex)
LLM_BACKEND = "openai"
CLAUDE_MODEL = None  # e.g. "sonnet" or "haiku"; None uses the CLI's default
CODEX_MODEL = None   # None uses the CLI's default
CLI_TIMEOUT = 300    # seconds per CLI call

# Embeddings for memory retrieval: "openai" (text-embedding-3-small) or 
# "local" (sentence-transformers, no API key). Stored agent embeddings must 
# come from the same backend; see reembed_agents.py.
EMBEDDING_BACKEND = "openai"
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BASE_DIR = f"{Path(__file__).resolve().parent.parent}"

## To do: Are the following needed in the new structure? Ideally Populations_Dir is for the user to define.
POPULATIONS_DIR = f"{BASE_DIR}/agent_bank/populations" 
LLM_PROMPT_DIR = f"{BASE_DIR}/simulation_engine/prompt_template"