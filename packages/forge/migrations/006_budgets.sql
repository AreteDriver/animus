-- Migration: 006_budgets
-- Description: Persistent budget tracking and management
-- Created: 2026-01-29

-- Budgets table - tracks budget allocations and usage
CREATE TABLE IF NOT EXISTS budgets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    total_amount REAL NOT NULL DEFAULT 0,
    used_amount REAL NOT NULL DEFAULT 0,
    period TEXT NOT NULL DEFAULT 'monthly',
    agent_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_budgets_period ON budgets(period);
CREATE INDEX IF NOT EXISTS idx_budgets_agent_id ON budgets(agent_id);
CREATE INDEX IF NOT EXISTS idx_budgets_name ON budgets(name);
