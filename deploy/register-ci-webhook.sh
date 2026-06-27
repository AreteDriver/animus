#!/usr/bin/env bash
# Register the GitHub Actions CI webhook in Animus
# Usage: ./register-ci-webhook.sh <ANIMUS_HOST> <AUTH_TOKEN>
#
# Requires: curl, jq
# Prerequisites:
#   1. Animus Forge is running with webhook endpoints exposed.
#   2. GITHUB_WEBHOOK_SECRET is exported (must match GitHub webhook secret).

set -euo pipefail

ANIMUS_HOST="${1:-http://localhost:8000}"
AUTH_TOKEN="${2:-}"          # Bearer token for Animus API auth
WEBHOOK_SECRET="${GITHUB_WEBHOOK_SECRET:-$(openssl rand -hex 32)}"

PAYLOAD=$(cat <<JSON
{
  "id": "github-ci-failure",
  "name": "GitHub Actions CI Failure",
  "description": "Triggers ci-diagnosis workflow when a GitHub Actions workflow run fails",
  "workflow_id": "ci-diagnosis",
  "secret": "${WEBHOOK_SECRET}",
  "payload_mappings": [
    {"source_path": "repository.owner.login", "target_variable": "owner", "default": "unknown"},
    {"source_path": "repository.name",         "target_variable": "repo",  "default": "unknown"},
    {"source_path": "workflow_run.id",        "target_variable": "run_id", "default": "0"},
    {"source_path": "workflow_run.conclusion","target_variable": "conclusion", "default": "failure"},
    {"source_path": "workflow_run.head_branch","target_variable": "branch", "default": "main"},
    {"source_path": "workflow_run.head_sha",  "target_variable": "commit_sha", "default": ""},
    {"source_path": "workflow_run.name",      "target_variable": "workflow_name", "default": ""},
    {"source_path": "sender.login",           "target_variable": "actor", "default": "github-actions"}
  ],
  "static_variables": {"remediate": "false"},
  "status": "active"
}
JSON
)

echo "Registering webhook 'github-ci-failure'..."
HEADERS=(-H "Content-Type: application/json")
if [ -n "${AUTH_TOKEN}" ]; then
  HEADERS+=(-H "Authorization: Bearer ${AUTH_TOKEN}")
fi

RESPONSE=$(curl -s -w "\n%{http_code}" \
  "${HEADERS[@]}" \
  -d "${PAYLOAD}" \
  "${ANIMUS_HOST}/v1/webhooks")

HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ]; then
  echo "✅ Webhook registered successfully"
  echo "$BODY" | jq .
  echo ""
  TRIGGER_URL=$(echo "$BODY" | jq -r '.trigger_url // empty')
  if [ -n "$TRIGGER_URL" ]; then
    echo "📡 Trigger URL: ${ANIMUS_HOST}${TRIGGER_URL}"
    echo "   Configure GitHub webhook with:"
    echo "     Payload URL: ${ANIMUS_HOST}${TRIGGER_URL}"
    echo "     Content type: application/json"
    echo "     Secret: ${WEBHOOK_SECRET}"
    echo "     Events: workflow_run (completed)"
  fi
else
  echo "❌ Registration failed (HTTP ${HTTP_CODE}):"
  echo "$BODY" | jq . 2>/dev/null || echo "$BODY"
  exit 1
fi
