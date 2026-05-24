#!/usr/bin/env bash
set -euo pipefail
REMOTE_HOST="${REMOTE_HOST:-192.168.2.21}"
usage() { echo "Usage: $0 -p PORT [-h HOST] [-m MODEL] [PROMPT]"; exit 0; }
TARGET_PORT=""; TARGET_MODEL=""; PROMPT=""
while [[ $# -gt 0 ]]; do case $1 in -h|--host) REMOTE_HOST="$2"; shift 2;; -p|--port) TARGET_PORT="$2"; shift 2;; -m|--model) TARGET_MODEL="$2"; shift 2;; --help) usage;; *) [ -z "$PROMPT" ] && PROMPT="$1" || PROMPT="$PROMPT $1"; shift;; esac; done
PROMPT="${PROMPT:-ping}"
[ -z "$TARGET_PORT" ] && { echo "Error: --port required"; usage; }
BASE_URL="http://${REMOTE_HOST}:${TARGET_PORT}"
if [ -z "$TARGET_MODEL" ]; then
  TARGET_MODEL=$(curl -s "${BASE_URL}/v1/models" 2>/dev/null | jq -r '.data[0].id' 2>/dev/null || true)
  [ -z "$TARGET_MODEL" ] && TARGET_MODEL="default"
fi
echo "=== Host: $REMOTE_HOST | Port: $TARGET_PORT | Model: $TARGET_MODEL ==="
payload=$(jq -n --arg prompt "$PROMPT" --arg model "$TARGET_MODEL" '{model: $model, messages: [{role: "user", content: $prompt}], max_tokens: 50}')
response=$(curl -s -w "\n%{http_code}" -X POST -H "Content-Type: application/json" -d "$payload" "${BASE_URL}/v1/chat/completions" 2>/dev/null)
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
if [ "$http_code" = "200" ]; then
  echo "OK ($http_code):"
  content=$(echo "$body" | jq -r '.choices[0].message.content' 2>/dev/null || true)
  [ -n "$content" ] && echo "$content" || { reasoning=$(echo "$body" | jq -r '.choices[0].message.reasoning // empty' 2>/dev/null || true); [ -n "$reasoning" ] && echo "[reasoning] $reasoning" || echo "$body" | jq .; }
else
  echo "HTTP $http_code"; echo "$body" | head -c 500
fi
