"""Postgres schema management (DDL) executed via the Supabase direct connection."""

from __future__ import annotations

from src.core.config import DIRECT_URL

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
    image_analysis TEXT,
    image_analysis_status TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

MIGRATIONS_DDL = """
ALTER TABLE uploaded_files ADD COLUMN IF NOT EXISTS image_analysis TEXT;
ALTER TABLE uploaded_files ADD COLUMN IF NOT EXISTS image_analysis_status TEXT;
ALTER TABLE uploaded_files ADD COLUMN IF NOT EXISTS user_id UUID;
ALTER TABLE uploaded_files ADD COLUMN IF NOT EXISTS project_id UUID;
CREATE INDEX IF NOT EXISTS idx_uploaded_files_project_id ON uploaded_files(project_id);
CREATE INDEX IF NOT EXISTS idx_uploaded_files_user_id ON uploaded_files(user_id);
ALTER TABLE messages ADD COLUMN IF NOT EXISTS seq BIGSERIAL;
CREATE INDEX IF NOT EXISTS idx_messages_conversation_seq ON messages(conversation_id, seq);
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
            cur.execute(MIGRATIONS_DDL)
        conn.commit()
    finally:
        conn.close()
