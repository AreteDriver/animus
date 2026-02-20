-- Migration 009: Filesystem tools for chat sessions
-- Enables agents to propose file edits and tracks file access for audit

-- Edit proposals table
-- Stores proposed changes that require user approval before applying
CREATE TABLE IF NOT EXISTS edit_proposals (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    old_content TEXT,  -- NULL for new files
    new_content TEXT NOT NULL,
    description TEXT DEFAULT '',
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'applied', 'failed')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    applied_at TIMESTAMP,
    error_message TEXT
);

-- File access audit log
-- Tracks all file access for security auditing
CREATE TABLE IF NOT EXISTS file_access_log (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    tool TEXT NOT NULL,  -- read_file, list_files, search_code, etc.
    file_path TEXT NOT NULL,
    operation TEXT NOT NULL,  -- read, list, search
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_edit_proposals_session ON edit_proposals(session_id);
CREATE INDEX IF NOT EXISTS idx_edit_proposals_status ON edit_proposals(status);
CREATE INDEX IF NOT EXISTS idx_edit_proposals_session_status ON edit_proposals(session_id, status);

CREATE INDEX IF NOT EXISTS idx_file_access_session ON file_access_log(session_id);
CREATE INDEX IF NOT EXISTS idx_file_access_timestamp ON file_access_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_file_access_session_timestamp ON file_access_log(session_id, timestamp);
