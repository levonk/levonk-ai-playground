#!/usr/bin/env bash
#
# test-query.sh - Send a test prompt to locally-running vLLM containers.
#
# Auto-discovers running Docker containers whose image name contains "llm"
# (case-insensitive) and which publish a TCP port, then POSTs an OpenAI-style
# /v1/chat/completions request to each one and prints the response content
# (falling back to the reasoning field, then the raw body). Exits non-zero if
# no container returned HTTP 200.
#
# Usage:
#   ./test-query.sh [OPTIONS] [PROMPT]
#
# Arguments:
#   PROMPT              Prompt text to send. Multiple positional args are
#                       joined with spaces. Default: "ping"
#
# Options:
#   -p, --port PORT     Target a single host port instead of auto-discovering
#                       containers. When set, Docker discovery is skipped.
#       --HOST HOST     Host to use for the manual/fallback target instead of
#                       the `ai` default (overrides FALLBACK_HOST). Only
#                       relevant when -p is given or discovery finds nothing.
#   -m, --model MODEL   Model name to send in the request payload. Default:
#                       auto-detected from the container's /v1/models
#                       endpoint (first model id), falling back to the
#                       container name.
#   -h, --help          Show usage and exit.
#
# Fallback:
#   When no -p is given AND no running LLM containers are discovered, the
#   script falls back to host `ai` port 8002 (overridable via the
#   FALLBACK_HOST and FALLBACK_PORT env vars).
#
# Environment:
#   FALLBACK_HOST       Host to use when discovery finds nothing
#                       (default: ai).
#   FALLBACK_PORT       Port to use when discovery finds nothing
#                       (default: 8002).
#   CONNECT_TIMEOUT     Curl --connect-timeout in seconds (default: 5).
#   MAX_TIME            Curl --max-time in seconds (default: 120).
#   MAX_TOKENS          max_tokens sent in the request payload (default: 2048).
#   Requires `docker` and `curl`. `jq` is used when available for JSON
#   building/parsing; falls back to sed-based escaping otherwise.
#
# Examples:
#   ./test-query.sh                          # ping every running LLM container
#   ./test-query.sh 'hello world'            # custom prompt, all containers
#   ./test-query.sh -p 8002 -m qwen3 'hi'    # single port, explicit model
#   ./test-query.sh --port 8002 ping
#
set -uo pipefail
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-5}"
MAX_TIME="${MAX_TIME:-120}"
MAX_TOKENS="${MAX_TOKENS:-2048}"

# curl exit code -> human message (uses $ERR_HOST/$ERR_PORT for context)
curl_err() {
  case "$1" in
    6)  echo "could not resolve host '${ERR_HOST}' (DNS failure)";;
    7)  echo "connection refused — nothing listening on ${ERR_HOST}:${ERR_PORT}, or firewall blocking it";;
    28) echo "timed out after ${MAX_TIME}s — host reachable but no response";;
    35) echo "TLS/SSL handshake failed";;
    52) echo "server returned empty reply (crashed or not an HTTP server?)";;
    56) echo "receive error — connection dropped mid-response";;
    *)  echo "curl exit code $1";;
  esac
}

# Run curl, capture body+http_code into HTTP_CODE/BODY/CURL_EXIT/CURL_ERR.
# $1=method $2=url [$3=json payload]
do_curl() {
  local method="$1" url="$2" payload="${3:-}" err_file
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
    HTTP_CODE=""; BODY=""; return 1
  fi
  HTTP_CODE="$(printf '%s' "$response" | tail -n1)"
  BODY="$(printf '%s' "$response" | sed '$d')"
  return 0
}

# Extract a server error message from a JSON body (best-effort, jq-optional).
extract_err_msg() {
  if command -v jq &> /dev/null; then
    printf '%s' "$1" | jq -r '.error.message // .error // .detail // empty' 2>/dev/null || true
  fi
}

usage() {
    echo "Usage: $0 [OPTIONS] [PROMPT]"
    echo
    echo "Options:"
    echo "  -p, --port PORT      Target port (default: auto-discover from docker)"
    echo "      --HOST HOST      Host for manual/fallback target (default: ai)"
    echo "  -m, --model MODEL    Model name (default: auto-detect from /v1/models or container name)"
    echo "  -h, --help           Show this help"
    echo
    echo "Examples:"
    echo "  $0 'hello world'"
    echo "  $0 -p 8002 -m 'my-model' 'hello world'"
    echo "  $0 --port 8002 ping"
    exit 0
}

TARGET_PORT=""
TARGET_MODEL=""
PROMPT=""
FALLBACK_HOST="${FALLBACK_HOST:-ai}"
FALLBACK_PORT="${FALLBACK_PORT:-8002}"
HOST_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            TARGET_PORT="$2"
            shift 2
            ;;
        --HOST)
            HOST_OVERRIDE="$2"
            FALLBACK_HOST="$2"
            shift 2
            ;;
        -m|--model)
            TARGET_MODEL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            if [ -z "$PROMPT" ]; then
                PROMPT="$1"
            else
                PROMPT="$PROMPT $1"
            fi
            shift
            ;;
    esac
done

PROMPT="${PROMPT:-ping}"

if ! command -v docker &> /dev/null; then
    echo "Error: docker not found"
    exit 1
fi

discover_containers() {
    docker ps --format '{{.Names}}|{{.Image}}|{{.Ports}}' | grep -i 'llm' | grep -E '[0-9]+->[0-9]+/tcp' || true
}

mapfile -t lines < <(discover_containers)

FALLBACK_ACTIVE=0
if [ ${#lines[@]} -eq 0 ] && [ -z "$TARGET_PORT" ]; then
    echo "No running LLM containers found (image must contain 'llm', case-insensitive); falling back to ${FALLBACK_HOST}:${FALLBACK_PORT}."
    TARGET_PORT="$FALLBACK_PORT"
    FALLBACK_ACTIVE=1
fi

found=false

process_port() {
    local name="$1"
    local image="$2"
    local host="$3"
    local host_port="$4"
    local model_name="$5"

    ERR_HOST="$host"; ERR_PORT="$host_port"
    local url="http://${host}:${host_port}/v1/chat/completions"
    echo "=== Container: $name | Image: $image | Host: $host | Port: $host_port | Model: $model_name ==="
    echo "    POST $url  (prompt: ${PROMPT:0:60})"

    if command -v jq &> /dev/null; then
        payload=$(jq -n \
            --arg prompt "$PROMPT" \
            --arg model "$model_name" \
            --argjson max_tokens "$MAX_TOKENS" \
            '{model: $model, messages: [{role: "user", content: $prompt}], max_tokens: $max_tokens}')
    else
        escaped_prompt=$(printf '%s' "$PROMPT" | sed 's/"/\\"/g')
        payload="{\"model\":\"$model_name\",\"messages\":[{\"role\":\"user\",\"content\":\"$escaped_prompt\"}],\"max_tokens\":$MAX_TOKENS}"
    fi

    if ! do_curl POST "$url" "$payload"; then
        echo "ERROR: request failed — $(curl_err "$CURL_EXIT")"
        [ -n "$CURL_ERR" ] && echo "       curl: $CURL_ERR"
        echo
        return
    fi

    if [ "$HTTP_CODE" != "200" ]; then
        echo "ERROR: HTTP $HTTP_CODE from $url"
        local err_msg; err_msg="$(extract_err_msg "$BODY")"
        [ -n "$err_msg" ] && echo "       server: $err_msg"
        echo "       body (first 500 bytes):"
        printf '%s' "$BODY" | head -c 500
        echo; echo
        return
    fi

    echo "OK (200):"
    if command -v jq &> /dev/null; then
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
    else
        printf '%s' "$BODY"
    fi
    found=true
    echo
}

# Probe /v1/models on host:port, return model id via stdout (empty on failure).
detect_model() {
    local host="$1" host_port="$2"
    ERR_HOST="$host"; ERR_PORT="$host_port"
    if ! do_curl GET "http://${host}:${host_port}/v1/models"; then
        echo "WARN: /v1/models probe failed ($(curl_err "$CURL_EXIT")) on ${host}:${host_port}." >&2
        [ -n "$CURL_ERR" ] && echo "       curl: $CURL_ERR" >&2
        return 1
    fi
    if [ "$HTTP_CODE" != "200" ]; then
        echo "WARN: /v1/models returned HTTP $HTTP_CODE on ${host}:${host_port}." >&2
        return 1
    fi
    if command -v jq &> /dev/null; then
        printf '%s' "$BODY" | jq -r '.data[0].id // empty' 2>/dev/null || true
    fi
}

if [ -n "$TARGET_PORT" ]; then
    manual_host="localhost"
    [ "$FALLBACK_ACTIVE" = 1 ] && manual_host="$FALLBACK_HOST"
    [ -n "$HOST_OVERRIDE" ] && manual_host="$HOST_OVERRIDE"
    if [ -z "$TARGET_MODEL" ]; then
        TARGET_MODEL="$(detect_model "$manual_host" "$TARGET_PORT")"
        [ -z "$TARGET_MODEL" ] && { echo "INFO: no model detected; using 'default'."; TARGET_MODEL="default"; }
    fi
    process_port "manual" "manual" "$manual_host" "$TARGET_PORT" "$TARGET_MODEL"
else
    for line in "${lines[@]}"; do
        IFS='|' read -r name image ports <<< "$line"
        host_port=$(echo "$ports" | grep -oE '[0-9]+->' | sed 's/->//' | head -1)

        if [ -z "$host_port" ]; then
            echo "WARN: container '$name' has no published TCP port, skipping."
            continue
        fi

        if [ -z "$TARGET_MODEL" ]; then
            model_name="$(detect_model "localhost" "$host_port")"
            [ -z "$model_name" ] && { echo "INFO: using container name '$name' as model."; model_name="$name"; }
        else
            model_name="$TARGET_MODEL"
        fi

        process_port "$name" "$image" "localhost" "$host_port" "$model_name"
    done
fi

if [ "$found" = false ]; then
    echo "No successful LLM responses found."
    exit 1
fi
