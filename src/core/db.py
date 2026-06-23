"""Postgres schema management (DDL) executed via the Supabase direct connection."""

from __future__ import annotations

from src.core.config import DIRECT_URL, EMBEDDING_DIM

# Base tables that predate the insight pipeline.
SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    context TEXT,
    structured_context TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id);

CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    title TEXT NOT NULL DEFAULT 'New conversation',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_conversations_project_id ON conversations(project_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    seq BIGSERIAL,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    file_ids JSONB,
    tools_used JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS uploaded_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    original_url TEXT NOT NULL,
    compressed_url TEXT,
    file_type TEXT NOT NULL,
    mime_type TEXT,
    size_bytes BIGINT,
    processing_status TEXT,
    user_id UUID,
    project_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# pgvector + the insight pipeline tables. EMBEDDING_DIM is interpolated from .env so
# the vector columns match the configured embedding model's dimensionality.
VECTOR_DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS image_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL,
    user_id UUID,
    file_id UUID NOT NULL,
    content TEXT NOT NULL,
    chunk_index INT,
    embedding VECTOR({EMBEDDING_DIM}),
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_image_chunks_project_id ON image_chunks(project_id);
CREATE INDEX IF NOT EXISTS idx_image_chunks_file_id ON image_chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_image_chunks_embedding
    ON image_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS file_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL,
    user_id UUID,
    file_id UUID NOT NULL,
    content TEXT NOT NULL,
    chunk_index INT,
    embedding VECTOR({EMBEDDING_DIM}),
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_file_chunks_project_id ON file_chunks(project_id);
CREATE INDEX IF NOT EXISTS idx_file_chunks_file_id ON file_chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_file_chunks_embedding
    ON file_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS project_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL,
    user_id UUID,
    file_id UUID NOT NULL,
    media_kind TEXT,
    theory_context TEXT,
    structured_context JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_project_insights_project_id ON project_insights(project_id);
CREATE INDEX IF NOT EXISTS idx_project_insights_project_file
    ON project_insights(project_id, file_id);

CREATE TABLE IF NOT EXISTS project_knowledge_base (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL UNIQUE,
    user_id UUID,
    theory_context TEXT,
    structured_context JSONB,
    insights_total INT NOT NULL DEFAULT 0,
    insights_processed INT NOT NULL DEFAULT 0,
    status TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_project_knowledge_base_project_id
    ON project_knowledge_base(project_id);

-- Generated should-cost reports. The aggregated structured JSON (report_json)
-- is the source of truth for edits; the rendered PDF lives in the `reports`
-- storage bucket (pdf_path/pdf_url), and `html` backs the live preview panel.
-- Legacy markdown/costs columns are kept nullable for back-compat.
CREATE TABLE IF NOT EXISTS reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL,
    conversation_id UUID,
    user_id UUID,
    title TEXT,
    volume INT,
    report_json JSONB,
    html TEXT,
    pdf_path TEXT,
    pdf_url TEXT,
    markdown TEXT,
    costs JSONB,
    status TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_reports_project_id ON reports(project_id);
CREATE INDEX IF NOT EXISTS idx_reports_conversation_id ON reports(conversation_id);

-- Human-in-the-loop questions the report tool asked the user, with their
-- answers. Insights extracted from answered questions are folded back into the
-- project's insight pipeline in the background.
CREATE TABLE IF NOT EXISTS report_questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL,
    conversation_id UUID,
    user_id UUID,
    report_id UUID,
    question TEXT NOT NULL,
    kind TEXT,
    answer TEXT,
    file_ids JSONB,
    status TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    answered_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_report_questions_project_id
    ON report_questions(project_id);
CREATE INDEX IF NOT EXISTS idx_report_questions_conversation_id
    ON report_questions(conversation_id);
"""

MIGRATIONS_DDL = """
-- Legacy columns kept for backward-compat (no longer written to).
ALTER TABLE uploaded_files ADD COLUMN IF NOT EXISTS image_analysis TEXT;
ALTER TABLE uploaded_files ADD COLUMN IF NOT EXISTS image_analysis_status TEXT;
ALTER TABLE uploaded_files ADD COLUMN IF NOT EXISTS user_id UUID;
ALTER TABLE uploaded_files ADD COLUMN IF NOT EXISTS project_id UUID;
-- Unified processing status for both images and documents.
ALTER TABLE uploaded_files ADD COLUMN IF NOT EXISTS processing_status TEXT;
CREATE INDEX IF NOT EXISTS idx_uploaded_files_project_id ON uploaded_files(project_id);
CREATE INDEX IF NOT EXISTS idx_uploaded_files_user_id ON uploaded_files(user_id);
ALTER TABLE messages ADD COLUMN IF NOT EXISTS seq BIGSERIAL;
CREATE INDEX IF NOT EXISTS idx_messages_conversation_seq ON messages(conversation_id, seq);
-- Generated-report metadata attached to an assistant message (report_id, title, volume).
ALTER TABLE messages ADD COLUMN IF NOT EXISTS report JSONB;

-- Should-cost reports: store the aggregated structured JSON + rendered HTML, and
-- the PDF in the `reports` storage bucket (path/URL only). The old inline base64
-- column is dropped — the PDF is only ever served from the bucket now.
ALTER TABLE reports ADD COLUMN IF NOT EXISTS report_json JSONB;
ALTER TABLE reports ADD COLUMN IF NOT EXISTS html TEXT;
ALTER TABLE reports ADD COLUMN IF NOT EXISTS pdf_path TEXT;
ALTER TABLE reports ADD COLUMN IF NOT EXISTS pdf_url TEXT;
ALTER TABLE reports DROP COLUMN IF EXISTS pdf_base64;
"""


def ensure_schema() -> None:
    """Create all application tables and run idempotent migrations."""
    if not DIRECT_URL:
        return

    import psycopg2

    conn = psycopg2.connect(DIRECT_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_DDL)
            cur.execute(VECTOR_DDL)
            cur.execute(MIGRATIONS_DDL)
        conn.commit()
    finally:
        conn.close()
