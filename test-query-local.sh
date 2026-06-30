#!/usr/bin/env bash
#
# test-query-local.sh - Send a test prompt to a vLLM server on a remote host.
#
# Unlike test-query.sh (which auto-discovers local Docker containers), this
# script targets a single OpenAI-compatible endpoint on a remote machine and
# POSTs an OpenAI-style /v1/chat/completions request, printing the response
# content (falling back to the reasoning field, then the raw body).
#
# Usage:
#   ./test-query-local.sh [OPTIONS] [PROMPT]
#
# Arguments:
#   PROMPT              Prompt text to send. Multiple positional args are
#                       joined with spaces. Default: "ping"
#
# Options:
#   -p, --port PORT     Target host port. Default: 8002
#   -h, --host HOST     Remote host or IP. Default: $REMOTE_HOST or `ai`
#       --HOST HOST     Same as --host (overrides the `ai` default).
#   -m, --model MODEL   Model name in the request payload. Default:
#                       auto-detected from the host's /v1/models endpoint
#                       (first model id), falling back to "default".
#       --help          Show usage and exit.
#
# Environment:
#   REMOTE_HOST         Default remote host when -h is not passed
#                       (default: ai).
#   CONNECT_TIMEOUT     Curl --connect-timeout in seconds (default: 5).
#   MAX_TIME            Curl --max-time in seconds (default: 120).
#   MAX_TOKENS          max_tokens sent in the request payload (default: 2048).
# Requires `curl` and `jq`.
#
# Examples:
#   ./test-query-local.sh                         # ping ai:8002
#   ./test-query-local.sh 'hello world'           # custom prompt, ai:8002
#   ./test-query-local.sh -p 8002 'hello world'   # explicit port
#   ./test-query-local.sh -h 10.0.0.5 -p 8001 -m qwen3 hi
#
set -uo pipefail
REMOTE_HOST="${REMOTE_HOST:-ai}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-5}"
MAX_TIME="${MAX_TIME:-120}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
usage() { echo "Usage: $0 [-p PORT] [-h HOST] [-m MODEL] [PROMPT]"; exit 0; }
TARGET_PORT=""; TARGET_MODEL=""; PROMPT=""
while [[ $# -gt 0 ]]; do case $1 in -h|--host|--HOST) REMOTE_HOST="$2"; shift 2;; -p|--port) TARGET_PORT="$2"; shift 2;; -m|--model) TARGET_MODEL="$2"; shift 2;; --help) usage;; *) [ -z "$PROMPT" ] && PROMPT="$1" || PROMPT="$PROMPT $1"; shift;; esac; done
PROMPT="${PROMPT:-ping}"
TARGET_PORT="${TARGET_PORT:-8002}"
BASE_URL="http://${REMOTE_HOST}:${TARGET_PORT}"

# curl error code -> human message
curl_err() {
  case "$1" in
    6)  echo "could not resolve host '${REMOTE_HOST}' (DNS failure)";;
    7)  echo "connection refused — nothing listening on ${REMOTE_HOST}:${TARGET_PORT}, or firewall blocking it";;
    28) echo "timed out after ${MAX_TIME}s — host reachable but no response";;
    35) echo "TLS/SSL handshake failed";;
    52) echo "server returned empty reply (crashed or not an HTTP server?)";;
    56) echo "receive error — connection dropped mid-response";;
    *)  echo "curl exit code $1";;
  esac
}

# Run curl, capture body+http_code, report errors verbosely.
# $1=method $2=url [$3=json payload]
# Sets: HTTP_CODE, BODY, CURL_EXIT
do_curl() {
  local method="$1" url="$2" payload="${3:-}"
  local err_file
  err_file="$(mktemp)"
  if [ -n "$payload" ]; then
    response=$(curl -sS --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" \
      -w "\n%{http_code}" -X "$method" -H "Content-Type: application/json" \
      -d "$payload" "$url" 2>"$err_file")
  else
    response=$(curl -sS --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" \
      -w "\n%{http_code}" -X "$method" "$url" 2>"$err_file")
  fi
  CURL_EXIT=$?
  CURL_ERR="$(cat "$err_file")"
  rm -f "$err_file"
  if [ "$CURL_EXIT" -ne 0 ]; then
    HTTP_CODE=""; BODY=""
    return 1
  fi
  HTTP_CODE="$(printf '%s' "$response" | tail -n1)"
  BODY="$(printf '%s' "$response" | sed '$d')"
  return 0
}

# --- model detection ---
if [ -z "$TARGET_MODEL" ]; then
  if ! do_curl GET "${BASE_URL}/v1/models"; then
    echo "WARN: /v1/models probe failed ($(curl_err "$CURL_EXIT")); using model='default'."
    [ -n "$CURL_ERR" ] && echo "       curl: $CURL_ERR"
    TARGET_MODEL="default"
  else
    if [ "$HTTP_CODE" = "200" ]; then
      TARGET_MODEL="$(printf '%s' "$BODY" | jq -r '.data[0].id // empty' 2>/dev/null || true)"
      [ -z "$TARGET_MODEL" ] && { echo "WARN: /v1/models returned 200 but no model id found; using 'default'."; TARGET_MODEL="default"; }
    else
      echo "WARN: /v1/models returned HTTP $HTTP_CODE; using model='default'."
      printf '%s' "$BODY" | head -c 300 >&2; echo >&2
      TARGET_MODEL="default"
    fi
  fi
fi

echo "=== Host: $REMOTE_HOST | Port: $TARGET_PORT | Model: $TARGET_MODEL ==="
echo "    POST ${BASE_URL}/v1/chat/completions  (prompt: ${PROMPT:0:60})"

payload=$(jq -n --arg prompt "$PROMPT" --arg model "$TARGET_MODEL" --argjson max_tokens "$MAX_TOKENS" \
  '{model: $model, messages: [{role: "user", content: $prompt}], max_tokens: $max_tokens}')

if ! do_curl POST "${BASE_URL}/v1/chat/completions" "$payload"; then
  echo "ERROR: request failed — $(curl_err "$CURL_EXIT")"
  [ -n "$CURL_ERR" ] && echo "       curl: $CURL_ERR"
  exit 1
fi

if [ "$HTTP_CODE" != "200" ]; then
  echo "ERROR: HTTP $HTTP_CODE from ${BASE_URL}/v1/chat/completions"
  # try to extract server error message
  err_msg="$(printf '%s' "$BODY" | jq -r '.error.message // .error // .detail // empty' 2>/dev/null || true)"
  [ -n "$err_msg" ] && echo "       server: $err_msg"
  echo "       body (first 500 bytes):"
  printf '%s' "$BODY" | head -c 500
  echo
  exit 1
fi

echo "OK (200):"
content="$(printf '%s' "$BODY" | jq -r '.choices[0].message.content // empty' 2>/dev/null || true)"
if [ -n "$content" ]; then
  echo "$content"
else
  reasoning="$(printf '%s' "$BODY" | jq -r '.choices[0].message.reasoning // empty' 2>/dev/null || true)"
  if [ -n "$reasoning" ]; then
    echo "[reasoning] $reasoning"
  else
    echo "WARN: 200 but no .choices[0].message.content; raw body:"
    printf '%s' "$BODY" | jq . 2>/dev/null || printf '%s' "$BODY"
  fi
fi
