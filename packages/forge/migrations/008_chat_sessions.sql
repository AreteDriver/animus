-- Migration: 008_chat_sessions
-- Description: Chat sessions and messages for conversational AI interface
-- Created: 2026-01-30

-- Chat sessions table - tracks conversation threads
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'New Chat',
    project_path TEXT,
    mode TEXT NOT NULL DEFAULT 'assistant',  -- 'assistant', 'self_improve'
    status TEXT NOT NULL DEFAULT 'active',   -- 'active', 'archived'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT  -- JSON for extensibility
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_status ON chat_sessions(status);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated ON chat_sessions(updated_at DESC);

-- Chat messages table - stores conversation history
CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,                      -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    agent TEXT,                              -- which agent authored (supervisor, planner, etc.)
    job_id TEXT,                             -- linked job if any
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT,                           -- JSON for additional data
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_messages_job ON chat_messages(job_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_agent ON chat_messages(agent);

-- Chat session jobs - links sessions to executed jobs
CREATE TABLE IF NOT EXISTS chat_session_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    UNIQUE(session_id, job_id)
);

CREATE INDEX IF NOT EXISTS idx_chat_session_jobs_session ON chat_session_jobs(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_session_jobs_job ON chat_session_jobs(job_id);
