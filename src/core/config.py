"""Application settings, read once from the environment (.env)."""

import os

from dotenv import load_dotenv

load_dotenv()

APP_NAME = "XOR Chat API"

# Redis: stores the last N messages per conversation.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MAX_CHAT_MESSAGES = int(os.getenv("MAX_CHAT_MESSAGES", "20"))

# Background processing: number of parallel worker processes to run from
# worker.py. Each is an independent process consuming the same queue, so N
# workers analyze N files concurrently. Scale this with available CPU/RAM.
WORKER_COUNT = int((os.getenv("WORKER_COUNT", "1").strip().strip('"').strip("'")) or "1")

# LLM (Bedrock via LiteLLM; must support tool calling for the deep agent).
LLM_MODEL = os.getenv("LLM_MODEL", "bedrock/qwen.qwen3-vl-235b-a22b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip().strip('"').strip("'")
LLM_API_BASE = os.getenv("LLM_API_BASE", "").strip().strip('"').strip("'")

# A small/cheap model used ONLY for fast intent routing (classifying a chat
# message into generate/edit/fetch/chat). It does no reasoning over report
# content — just intent — so the smallest available model is fine. Falls back to
# LLM_MODEL when unset so the app works out of the box; set this to a cheaper
# Bedrock model (e.g. a Nova/Haiku-class id) to cut routing cost/latency.
LLM_INTENT_MODEL = (
    os.getenv("LLM_INTENT_MODEL", "").strip().strip('"').strip("'") or LLM_MODEL
)
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
    (os.getenv("REPORT_DEFAULT_VOLUME", "1").strip().strip('"').strip("'")) or "1"
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
# PCB fabrication quote — JLCPCB OpenAPI (USD prices, converted to INR via
# Frankfurter). Requests are signed with HMAC-SHA256 (see services/jlcpcb.py).
# The credentials below are env-overridable; move them to .env in production.
# --------------------------------------------------------------------------- #
JLCPCB_API_BASE = os.getenv(
    "JLCPCB_API_BASE", "https://open.jlcpcb.com"
).strip().strip('"').strip("'")
JLCPCB_CALCULATE_PATH = os.getenv(
    "JLCPCB_CALCULATE_PATH", "/overseas/openapi/pcb/calculate"
).strip().strip('"').strip("'")
JLCPCB_APP_ID = os.getenv("JLCPCB_APP_ID", "541563789036167170").strip().strip('"').strip("'")
JLCPCB_ACCESS_KEY = os.getenv(
    "JLCPCB_ACCESS_KEY", "56e210adc361476e97f0f39a6f0be274"
).strip().strip('"').strip("'")
JLCPCB_SECRET_KEY = os.getenv(
    "JLCPCB_SECRET_KEY", "usxT59LkUuuY7BJyGaM5t5oxNxmzyGim"
).strip().strip('"').strip("'")
# Destination country for the quote (affects shipping options, not board cost).
JLCPCB_COUNTRY = os.getenv("JLCPCB_COUNTRY", "IN").strip().strip('"').strip("'")

# Legacy PCBWay settings (no longer used for quoting; kept for reference).
PCBWAY_API_KEY = os.getenv("PCBWAY_API_KEY", "").strip().strip('"').strip("'")
PCBWAY_API_BASE = os.getenv(
    "PCBWAY_API_BASE", "https://api-partner.pcbway.com"
).strip().strip('"').strip("'")

# --------------------------------------------------------------------------- #
# FX — Frankfurter (no key). Used ONLY to convert the JLCPCB USD fab quote to INR.
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

# Folder for per-failure debug JSON files (one timestamped file per failed
# external call / fallback during report generation). Best-effort; never blocks.
REPORT_FAILURE_LOG_DIR = os.getenv(
    "REPORT_FAILURE_LOG_DIR", "logs/report_failures"
).strip().strip('"').strip("'")
# Master switch: when false, no failure breadcrumb files are written at all.
REPORT_FAILURE_LOG_ENABLED = os.getenv(
    "REPORT_FAILURE_LOG_ENABLED", "true"
).strip().strip('"').strip("'").lower() in {"1", "true", "yes"}

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

# External parts database (read-only): a separate Postgres holding the JLCPCB
# component dataset (MPN, stock, quantity-break pricing). Used for exact-MPN
# existence + pricing lookups. Kept fully separate from the app's own DB above.
PG_HOST = os.getenv("PG_HOST", "").strip().strip('"').strip("'")
PG_PORT = int((os.getenv("PG_PORT", "5432").strip().strip('"').strip("'")) or "5432")
PG_USER = os.getenv("PG_USER", "").strip().strip('"').strip("'")
PG_PASSWORD = os.getenv("PG_PASSWORD", "").strip().strip('"').strip("'")
PG_DATABASE = os.getenv("PG_DATABASE", "").strip().strip('"').strip("'")
PG_SCHEMA = os.getenv("PG_SCHEMA", "public").strip().strip('"').strip("'")
PG_TABLE = os.getenv("PG_TABLE", "components").strip().strip('"').strip("'")
PG_ENABLED = os.getenv("PG_ENABLED", "false").strip().strip('"').strip("'").lower() in {
    "1",
    "true",
    "yes",
}
