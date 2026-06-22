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

# Embedding model (Bedrock via LiteLLM) for pgvector RAG over image/file chunks.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bedrock/amazon.titan-embed-text-v2:0").strip().strip('"').strip("'")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "").strip().strip('"').strip("'")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "").strip().strip('"').strip("'")
EMBEDDING_DIM = int((os.getenv("EMBEDDING_DIM", "1024").strip().strip('"').strip("'")) or "1024")
EMBEDDING_SEND_DIMENSIONS = os.getenv("EMBEDDING_SEND_DIMENSIONS", "true").strip().strip('"').strip("'").lower() in {
    "1",
    "true",
    "yes",
}

# Knowledge-base recompute: minimum cosine similarity between a new insight and
# the project's existing accumulated analysis for the insight to be folded in.
# Lenient by design — we only want to skip clearly-unrelated uploads (e.g. a
# stray photo that has nothing to do with the product). Set to 0 to disable.
KB_RELATEDNESS_THRESHOLD = float(
    (os.getenv("KB_RELATEDNESS_THRESHOLD", "0.15").strip().strip('"').strip("'")) or "0.15"
)

# Tavily web search tool.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# DigiKey API (component MPN resolution + pricing for should-cost reports).
DIGIKEY_CLIENT_ID = os.getenv("DIGIKEY_CLIENT_ID", "").strip().strip('"').strip("'")
DIGIKEY_CLIENT_SECRET = os.getenv("DIGIKEY_CLIENT_SECRET", "").strip().strip('"').strip("'")
DIGIKEY_API_BASE = os.getenv("DIGIKEY_API_BASE", "https://api.digikey.com").strip().strip('"').strip("'")
DIGIKEY_LOCALE_SITE = os.getenv("DIGIKEY_LOCALE_SITE", "US").strip().strip('"').strip("'")
DIGIKEY_LOCALE_LANGUAGE = os.getenv("DIGIKEY_LOCALE_LANGUAGE", "en").strip().strip('"').strip("'")
DIGIKEY_LOCALE_CURRENCY = os.getenv("DIGIKEY_LOCALE_CURRENCY", "USD").strip().strip('"').strip("'")

# Report generation: default assumed production volume (units) for should-cost
# when the user has not stated one, and the cap on how many HILT questions the
# report tool may ask in one round (we keep this small to avoid annoying users).
REPORT_DEFAULT_VOLUME = int(
    (os.getenv("REPORT_DEFAULT_VOLUME", "1000").strip().strip('"').strip("'")) or "1000"
)
REPORT_MAX_QUESTIONS = int(
    (os.getenv("REPORT_MAX_QUESTIONS", "4").strip().strip('"').strip("'")) or "4"
)
# Fallback USD→INR rate when the FX API is unavailable.
REPORT_USD_INR_FALLBACK = float(
    (os.getenv("REPORT_USD_INR_FALLBACK", "85.0").strip().strip('"').strip("'")) or "85.0"
)

# The four order quantities the entire cost model is run at (the volume curve).
REPORT_VOLUME_CURVE = [
    int(x)
    for x in (
        os.getenv("REPORT_VOLUME_CURVE", "1,100,1000,10000")
        .strip()
        .strip('"')
        .strip("'")
        .split(",")
    )
    if x.strip()
] or [1, 100, 1000, 10000]

# --------------------------------------------------------------------------- #
# Component pricing — Mouser (INR list prices for the BOM).
# --------------------------------------------------------------------------- #
MOUSER_API_KEY = os.getenv("MOUSER_API_KEY", "").strip().strip('"').strip("'")
MOUSER_API_BASE = os.getenv(
    "MOUSER_API_BASE", "https://api.mouser.com/api/v2"
).strip().strip('"').strip("'")

# --------------------------------------------------------------------------- #
# PCB fabrication quote — PCBWay (USD prices, converted to INR via Frankfurter).
# --------------------------------------------------------------------------- #
PCBWAY_API_KEY = os.getenv("PCBWAY_API_KEY", "").strip().strip('"').strip("'")
PCBWAY_API_BASE = os.getenv(
    "PCBWAY_API_BASE", "https://api-partner.pcbway.com"
).strip().strip('"').strip("'")

# --------------------------------------------------------------------------- #
# FX — Frankfurter (no key). Used ONLY to convert the PCBWay USD quote to INR.
# --------------------------------------------------------------------------- #
FRANKFURTER_API_BASE = os.getenv(
    "FRANKFURTER_API_BASE", "https://api.frankfurter.dev/v1"
).strip().strip('"').strip("'")

# --------------------------------------------------------------------------- #
# Assembly model — Indian EMS rate card (no API). Stencil + setup are one-time
# NRE; the per-joint rate is per-unit. All INR.
# --------------------------------------------------------------------------- #
ASSEMBLY_SETUP_FEE_INR = float(
    (os.getenv("ASSEMBLY_SETUP_FEE_INR", "1000.0").strip().strip('"').strip("'")) or "1000.0"
)
ASSEMBLY_STENCIL_FEE_INR = float(
    (os.getenv("ASSEMBLY_STENCIL_FEE_INR", "500.0").strip().strip('"').strip("'")) or "500.0"
)
ASSEMBLY_RATE_PER_JOINT_INR = float(
    (os.getenv("ASSEMBLY_RATE_PER_JOINT_INR", "0.15").strip().strip('"').strip("'")) or "0.15"
)

# Supabase storage bucket for rendered report PDFs (path/URL stored, not base64).
REPORTS_BUCKET = os.getenv("REPORTS_BUCKET", "reports").strip().strip('"').strip("'")

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
