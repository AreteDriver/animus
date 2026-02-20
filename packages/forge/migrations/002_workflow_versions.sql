-- Migration: 002_workflow_versions
-- Description: Add workflow versioning support
-- Created: 2026-01-19

-- Workflow versions table - stores version history for each workflow
CREATE TABLE IF NOT EXISTS workflow_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_name TEXT NOT NULL,
    version TEXT NOT NULL,
    version_major INTEGER NOT NULL,
    version_minor INTEGER NOT NULL,
    version_patch INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    description TEXT,
    author TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE,
    metadata TEXT
);

-- Unique constraint on workflow name + version
CREATE UNIQUE INDEX IF NOT EXISTS idx_workflow_versions_name_version
    ON workflow_versions(workflow_name, version);

-- Index for finding active version
CREATE INDEX IF NOT EXISTS idx_workflow_versions_active
    ON workflow_versions(workflow_name, is_active) WHERE is_active = TRUE;

-- Index for version history queries (ordered by version components)
CREATE INDEX IF NOT EXISTS idx_workflow_versions_history
    ON workflow_versions(workflow_name, version_major DESC, version_minor DESC, version_patch DESC);

-- Index for content hash lookups (deduplication)
CREATE INDEX IF NOT EXISTS idx_workflow_versions_hash
    ON workflow_versions(workflow_name, content_hash);

-- Index for timestamp-based queries
CREATE INDEX IF NOT EXISTS idx_workflow_versions_created
    ON workflow_versions(workflow_name, created_at DESC);
