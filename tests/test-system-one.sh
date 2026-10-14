#!/usr/bin/env bash
#
# test-system-one.sh - Smoke-test a running System One scorer.
#
# The scorer (scripts/model-system-one.sh -> libexec/system-one-server.py) is not an
# OpenAI chat endpoint, so test-query.sh can't exercise it. This script hits
# its own API instead: GET /health, GET /v1/models, then POST /score with a
# sample routing decision and validates the response shape (per-option
# probabilities that sum to ~1 and a valid answer_index).
#
# Usage:
#   ./tests/test-system-one.sh [OPTIONS]
#
# Options:
#   -p, --port PORT     Target host port. Default: 8007
#   -h, --host HOST     Host or IP. Default: $REMOTE_HOST or `ai`
#       --HOST HOST     Same as --host (overrides the `ai` default).
#       --help          Show usage and exit.
#
# Environment:
#   REMOTE_HOST         Default host when -h is not passed (default: ai).
#   CONNECT_TIMEOUT     Curl --connect-timeout in seconds (default: 5).
#   MAX_TIME            Curl --max-time in seconds (default: 120).
#
# Requires `curl`; `jq` is used when available to validate the response
# (without jq the test only checks for HTTP 200 and prints the raw body).
#
# Examples:
#   ./tests/test-system-one.sh              # ping ai:8007
#   ./tests/test-system-one.sh -h localhost # scorer running on this machine
#   ./tests/test-system-one.sh -p 9007      # custom port
#
set -uo pipefail
REMOTE_HOST="${REMOTE_HOST:-ai}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-5}"
MAX_TIME="${MAX_TIME:-120}"
usage() { echo "Usage: $0 [-p PORT] [-h HOST]"; exit 0; }
TARGET_PORT=""
while [[ $# -gt 0 ]]; do case $1 in -h|--host|--HOST) REMOTE_HOST="$2"; shift 2;; -p|--port) TARGET_PORT="$2"; shift 2;; --help) usage;; *) echo "unknown arg: $1" >&2; exit 2;; esac; done
TARGET_PORT="${TARGET_PORT:-8007}"
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

fail() { echo "FAIL: $1"; exit 1; }

echo "=== Host: $REMOTE_HOST | Port: $TARGET_PORT ==="

# --- liveness ---
if ! do_curl GET "${BASE_URL}/health"; then
  fail "GET /health failed — $(curl_err "$CURL_EXIT")
       curl: $CURL_ERR"
fi
[ "$HTTP_CODE" = "200" ] || fail "GET /health returned HTTP $HTTP_CODE: ${BODY:0:300}"
echo "health: $BODY"

# --- model listing ---
do_curl GET "${BASE_URL}/v1/models" && [ "$HTTP_CODE" = "200" ] \
  && echo "models: $BODY" \
  || echo "WARN: /v1/models probe failed — continuing anyway"

# --- score ---
echo "    POST ${BASE_URL}/score  (3-option routing decision)"
payload='{"state":"Customer: my shoes arrived in the wrong size and I would like an exchange.","question":"Which team should handle this ticket?","options":["returns","shipping","billing"]}'

if ! do_curl POST "${BASE_URL}/score" "$payload"; then
  fail "POST /score failed — $(curl_err "$CURL_EXIT")
       curl: $CURL_ERR"
fi
[ "$HTTP_CODE" = "200" ] || fail "POST /score returned HTTP $HTTP_CODE: ${BODY:0:500}"

if command -v jq >/dev/null 2>&1; then
  n_options=$(printf '%s' "$BODY" | jq '.options | length' 2>/dev/null)
  answer_index=$(printf '%s' "$BODY" | jq '.answer_index // -1' 2>/dev/null)
  prob_sum=$(printf '%s' "$BODY" | jq '[.probabilities[]] | add // 0' 2>/dev/null)
  answer=$(printf '%s' "$BODY" | jq -r '.answer // empty' 2>/dev/null)
  [ "$n_options" = "3" ] || fail "expected 3 options echoed, got: $n_options"
  [ "$answer_index" -ge 0 ] && [ "$answer_index" -lt 3 ] \
    || fail "answer_index out of range: $answer_index"
  awk "BEGIN { exit !($prob_sum > 0.99 && $prob_sum < 1.01) }" \
    || fail "probabilities do not sum to ~1: $prob_sum"
  echo "OK (200): answer='$answer' (index $answer_index), probabilities sum to $prob_sum"
  printf '%s' "$BODY" | jq '{options, probabilities, answer_index, answer, temperature}'
else
  echo "OK (200) — jq not installed, raw body:"
  printf '%s\n' "$BODY"
fi
