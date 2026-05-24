#!/usr/bin/env bash
set -euo pipefail

PROMPT="${1:-ping}"

if ! command -v docker &> /dev/null; then
    echo "Error: docker not found"
    exit 1
fi

mapfile -t lines < <(docker ps --format '{{.Names}}|{{.Ports}}' | grep -E '[0-9]+->[0-9]+/tcp' || true)

if [ ${#lines[@]} -eq 0 ]; then
    echo "No running containers with port mappings found."
    exit 1
fi

found=false

for line in "${lines[@]}"; do
    IFS='|' read -r name ports <<< "$line"
    host_port=$(echo "$ports" | grep -oE '[0-9]+->' | sed 's/->//' | head -1)
    
    if [ -z "$host_port" ]; then
        continue
    fi
    
    url="http://localhost:${host_port}/v1/chat/completions"
    echo "=== Container: $name | Port: $host_port ==="
    
    if command -v jq &> /dev/null; then
        payload=$(jq -n \
            --arg prompt "$PROMPT" \
            '{
                model: "default",
                messages: [{role: "user", content: $prompt}],
                max_tokens: 50
            }')
    else
        escaped_prompt=$(printf '%s' "$PROMPT" | sed 's/"/\\"/g')
        payload="{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"$escaped_prompt\"}],\"max_tokens\":50}"
    fi
    
    if response=$(curl -s -w "\\n%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$url" 2>/dev/null); then
        
        http_code=$(echo "$response" | tail -n1)
        body=$(echo "$response" | sed '$d')
        
        if [ "$http_code" = "200" ]; then
            echo "OK ($http_code):"
            if command -v jq &> /dev/null; then
                echo "$body" | jq -r '.choices[0].message.content // .' 2>/dev/null || echo "$body"
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
done

if [ "$found" = false ]; then
    echo "No successful LLM responses found."
    exit 1
fi
