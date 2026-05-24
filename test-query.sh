#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [OPTIONS] [PROMPT]"
    echo
    echo "Options:"
    echo "  -p, --port PORT      Target port (default: auto-discover from docker)"
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

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            TARGET_PORT="$2"
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

if [ ${#lines[@]} -eq 0 ] && [ -z "$TARGET_PORT" ]; then
    echo "No running LLM containers found (image must contain 'llm', case-insensitive)."
    exit 1
fi

found=false

process_port() {
    local name="$1"
    local image="$2"
    local host_port="$3"
    local model_name="$4"

    url="http://localhost:${host_port}/v1/chat/completions"
    echo "=== Container: $name | Image: $image | Port: $host_port | Model: $model_name ==="

    if command -v jq &> /dev/null; then
        payload=$(jq -n \
            --arg prompt "$PROMPT" \
            --arg model "$model_name" \
            '{
                model: $model,
                messages: [{role: "user", content: $prompt}],
                max_tokens: 50
            }')
    else
        escaped_prompt=$(printf '%s' "$PROMPT" | sed 's/"/\\"/g')
        payload="{\"model\":\"$model_name\",\"messages\":[{\"role\":\"user\",\"content\":\"$escaped_prompt\"}],\"max_tokens\":50}"
    fi

    if response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$url" 2>/dev/null); then

        http_code=$(echo "$response" | tail -n1)
        body=$(echo "$response" | sed '$d')

        if [ "$http_code" = "200" ]; then
            echo "OK ($http_code):"
            if command -v jq &> /dev/null; then
                content=$(echo "$body" | jq -r '.choices[0].message.content' 2>/dev/null || true)
                if [ -n "$content" ]; then
                    echo "$content"
                else
                    reasoning=$(echo "$body" | jq -r '.choices[0].message.reasoning // empty' 2>/dev/null || true)
                    if [ -n "$reasoning" ]; then
                        echo "[reasoning] $reasoning"
                    else
                        echo "$body" | jq .
                    fi
                fi
            else
                echo "$body"
            fi
            found=true
        else
            echo "HTTP $http_code"
            echo "$body" | head -c 500
        fi
    else
        echo "Connection failed"
    fi
    echo
}

if [ -n "$TARGET_PORT" ]; then
    if [ -z "$TARGET_MODEL" ]; then
        detected_model=$(curl -s "http://localhost:${TARGET_PORT}/v1/models" 2>/dev/null | jq -r '.data[0].id' 2>/dev/null || true)
        if [ -n "$detected_model" ]; then
            TARGET_MODEL="$detected_model"
        else
            TARGET_MODEL="default"
        fi
    fi
    process_port "manual" "manual" "$TARGET_PORT" "$TARGET_MODEL"
else
    for line in "${lines[@]}"; do
        IFS='|' read -r name image ports <<< "$line"
        host_port=$(echo "$ports" | grep -oE '[0-9]+->' | sed 's/->//' | head -1)

        if [ -z "$host_port" ]; then
            continue
        fi

        if [ -z "$TARGET_MODEL" ]; then
            detected_model=$(curl -s "http://localhost:${host_port}/v1/models" 2>/dev/null | jq -r '.data[0].id' 2>/dev/null || true)
            if [ -n "$detected_model" ]; then
                model_name="$detected_model"
            else
                model_name="$name"
            fi
        else
            model_name="$TARGET_MODEL"
        fi

        process_port "$name" "$image" "$host_port" "$model_name"
    done
fi

if [ "$found" = false ]; then
    echo "No successful LLM responses found."
    exit 1
fi
