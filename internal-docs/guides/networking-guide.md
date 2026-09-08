# Networking Guide

> How to expose your local model server to other devices (phone, laptop,
> desktop) securely using Tailscale, without exposing ports to the public
> internet.
>
> Source: [lws.io — My local model setup](https://lws.io/blog/my-local-model-setup/)

## Why Tailscale

Your model server listens on a port (e.g., 8000). By default, it's only
accessible from localhost. To access it from your phone, laptop, or other
devices, you need a network path.

Options:
1. **Open a firewall port** — exposes your server to the public internet.
   Not recommended for a model server with no authentication.
2. **VPN** — works but complex to set up and maintain.
3. **Tailscale** — creates a private mesh network between your devices.
   Zero config, end-to-end encrypted, no ports exposed. **Recommended.**

## Setting up Tailscale

### On the model server (GPU host or Mac)

```bash
# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# Authenticate
sudo tailscale up

# Note your tailnet IP
tailscale ip -4
# e.g., 100.64.0.1
```

### On your other devices

- **iPhone/iPad**: Install Tailscale from the App Store, log in with the
  same account.
- **MacBook**: Install Tailscale from the App Store or `brew install tailscale`.
- **Linux**: Same install script as above.

All devices on the same Tailscale account are on the same private network
(tailnet). They can reach each other by tailnet IP or MagicDNS hostname.

## Connecting clients to the model server

Once Tailscale is set up, point your clients at the model server's tailnet
IP and port:

```bash
# Instead of:
curl http://localhost:8000/v1/chat/completions ...

# Use the tailnet IP:
curl http://100.64.0.1:8000/v1/chat/completions ...

# Or the MagicDNS hostname (if enabled):
curl http://gpu-host:8000/v1/chat/completions ...
```

### Client-specific setup

| Client | Platform | How to connect |
|--------|----------|---------------|
| Apollo | iOS | Settings > Custom Endpoint > `http://gpu-host:8000/v1` |
| Raycast AI | macOS | Configure custom OpenAI-compatible endpoint |
| Hermes desktop | macOS | Shell settings > Backend URL |
| Open WebUI | Any | Settings > OpenAI API URL |
| Cursor / Continue | macOS | Config file > API base URL |
| Custom scripts | Any | `curl http://gpu-host:8000/v1/...` |

All clients hit the same endpoint. No configuration drift between devices.

## Multiple models on the tailnet

Since each model script uses a unique port, you can expose all of them:

```
http://gpu-host:8000  → chat model
http://gpu-host:8001  → code model
http://gpu-host:8002  → reasoning model
http://gpu-host:8003  → general/thinking model
http://gpu-host:8004  → lite model
http://gpu-host:8005  → TTS
http://gpu-host:8006  → STT
```

Each device can connect to whichever model it needs. For example:
- Phone uses the lite model (8004) for quick chats
- Laptop uses the code model (8001) for coding
- Desktop uses the reasoning model (8002) for analysis

## Security considerations

- **No public exposure**: Tailscale uses WireGuard encryption. Your model
  server is not exposed to the public internet.
- **Device authentication**: Only devices on your tailnet can connect.
  Tailscale authenticates devices via your account (Google, GitHub, etc.).
- **No model server auth**: The model server (vLLM, SGLang, MLX) typically
  has no authentication. This is fine on a tailnet because only your devices
  can reach it. If you need API key auth, add a reverse proxy (nginx,
  Caddy) with auth in front.
- **Tailnet ACLs**: For additional security, configure Tailscale ACLs to
  restrict which devices can reach which ports.

## Alternative: SSH tunnel

If you don't want to use Tailscale, an SSH tunnel works for a single client:

```bash
# From your laptop, create a tunnel to the GPU host:
ssh -L 8000:localhost:8000 user@gpu-host

# Now localhost:8000 on your laptop forwards to gpu-host:8000
```

This is simpler but only works for one client at a time and requires SSH
access to the host.

## Using test-query-local.sh with Tailscale

The repo's `test-query-local.sh` targets a remote host. With Tailscale:

```bash
# Test the chat model via tailnet
./test-query-local.sh -h 100.64.0.1 -p 8000 'hello'

# Or with MagicDNS:
./test-query-local.sh -h gpu-host -p 8000 'hello'
```

## KV cache persistence on the tailnet

oMLX's KV cache SSD persistence is particularly valuable on a tailnet
setup. Coding agents and other agent workflows repeatedly revisit earlier
context. oMLX caches each KV block to SSD, so when the agent returns to a
previous prefix, it's restored from disk in milliseconds instead of being
recomputed. This makes the local setup practical for agent work over the
network, where latency would otherwise compound.

If using vllm-mlx on Apple Silicon, check whether its SSD-tiered cache
provides similar benefits for your workload.
