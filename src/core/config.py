"""Application settings, read once from the environment (.env)."""

import os

from dotenv import load_dotenv

load_dotenv()

APP_NAME = "XOR Chat API"

# Redis: stores the last N messages per conversation.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MAX_CHAT_MESSAGES = int(os.getenv("MAX_CHAT_MESSAGES", "20"))

# LLM (Bedrock via LiteLLM; must support tool calling for the deep agent).
LLM_MODEL = os.getenv("LLM_MODEL", "bedrock/qwen.qwen3-vl-235b-a22b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip().strip('"').strip("'")
LLM_API_BASE = os.getenv("LLM_API_BASE", "").strip().strip('"').strip("'")
LLM_TOOLS_ENABLED = os.getenv("LLM_TOOLS_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
}
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

if AWS_REGION:
    os.environ.setdefault("AWS_REGION", AWS_REGION)
    os.environ.setdefault("AWS_REGION_NAME", AWS_REGION)

# Tavily web search tool.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Supabase (storage + Postgres metadata for uploads).
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().strip('"').strip("'")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip().strip('"').strip("'")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip().strip('"').strip("'")
DIRECT_URL = os.getenv("DIRECT_URL", "").strip().strip('"').strip("'")
STORAGE_BUCKET_COMPRESSED = os.getenv("STORAGE_BUCKET_1", "compressed_uploads").strip().strip('"').strip("'")
STORAGE_BUCKET_ORIGINAL = os.getenv("STORAGE_BUCKET_2", "original_uploads").strip().strip('"').strip("'")

# Supabase Auth (email/password + OAuth). Redirect lands on the frontend callback.
AUTH_REDIRECT_URL = os.getenv(
    "AUTH_REDIRECT_URL", "http://localhost:5173/auth/callback"
).strip().strip('"').strip("'")
