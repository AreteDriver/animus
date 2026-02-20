-- Migration: 005_user_settings
-- Description: User preferences and API key storage
-- Created: 2026-01-29

-- User preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id TEXT PRIMARY KEY,
    theme TEXT DEFAULT 'system',
    compact_view INTEGER DEFAULT 0,
    show_costs INTEGER DEFAULT 1,
    default_page_size INTEGER DEFAULT 20,
    notify_execution_complete INTEGER DEFAULT 1,
    notify_execution_failed INTEGER DEFAULT 1,
    notify_budget_alert INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API keys table - stores encrypted API key metadata
-- Raw keys are encrypted, we only return masked versions
CREATE TABLE IF NOT EXISTS user_api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    encrypted_key TEXT NOT NULL,
    key_prefix TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, provider)
);

CREATE INDEX IF NOT EXISTS idx_user_api_keys_user ON user_api_keys(user_id);
