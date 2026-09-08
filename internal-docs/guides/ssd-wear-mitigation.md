# SSD Wear Mitigation

> How to protect soldered, non-replaceable Mac SSDs (and extend the life of
> any NVMe drive) by offloading model storage and high-write workloads to
> external SSDs.
>
> Sources: [coreguard.app Mac SSD endurance research](https://coreguard.app/insights/mac-ssd-endurance-disk-writes/), [Eclectic Light Company SSD longevity analysis](https://eclecticlight.co/2026/02/26/how-long-will-my-macs-ssd-last/)

## Why this matters

Apple Silicon Macs have soldered SSDs that cannot be replaced. When the SSD
wears out, the entire logic board needs replacement. SSD wear is caused by
write operations — each NAND cell can only be erased and rewritten a finite
number of times (~3,000 cycles for recent Apple SSDs).

Running local LLMs generates significant write traffic:
- Model downloads: 2-80 GB per model
- Model conversions (safetensors to MLX): rewrites the entire model
- KV cache SSD tiering: repeated writes during inference
- Docker VM images: layer pulls and container filesystem writes
- Swap: when unified memory is exhausted, macOS writes to SSD

## Write workload classification

Not all writes are equal. The key distinction is **write frequency**:

| Workload | Write pattern | Wear impact | Should go on external SSD? |
|----------|-------------|-------------|---------------------------|
| Model downloads | Write-once, read-many | Low per-download, but adds up with frequent model swaps | **Yes** |
| Model storage (HuggingFace cache) | Write-once, read-many | Low (static after download) | **Yes** |
| MLX auto-conversion | Write-once per model | Low (happens once per model) | **Yes** |
| KV cache SSD tiering | Write-many, read-many, overwrite | **High** (repeated writes during inference) | Trade-off (see below) |
| Docker VM image | Write-once per pull, then read | Low (static after pull) | **Yes** (Docker Desktop setting) |
| Docker container logs | Write-many, append | Medium | **Yes** (if using Docker) |
| Swap | Write-many, random | Medium-High | No (internal SSD is faster) |
| Temp files / build artifacts | Write-many, delete | Medium | **Yes** |

## The strategy: external SSD for storage, internal for speed

### What goes on the external SSD

1. **Model weights (HuggingFace cache)** — This is the biggest win. Models
   are 2-80 GB each. You might download dozens over time. Store them on an
   external NVMe SSD. They're read sequentially at inference time, which
   external SSDs handle well.

2. **Docker volumes** — Model caches, Docker images, container filesystems.
   Docker Desktop on Mac lets you move the VM disk image to external storage.

3. **Download staging** — Download new models to the external SSD first,
   then they're already in the right place.

4. **Build artifacts / temp files** — Anything transient that gets written
   and deleted.

### What stays on the internal SSD

1. **Swap** — macOS swap needs the internal SSD's speed. If you're hitting
   swap, reduce memory usage (smaller model, shorter context) rather than
   redirecting swap.

2. **KV cache SSD tiering** — This is the trade-off. See below.

3. **OS and applications** — System files, installed apps.

### The KV cache trade-off

vllm-mlx and oMLX can tier KV cache to SSD. This caches KV blocks to disk
so that when an agent revisits earlier context, the cache is restored from
disk in milliseconds instead of being recomputed.

**Option A: KV cache on internal SSD (faster, more wear)**
- Pro: Fast restore (internal NVMe is faster than external)
- Con: Repeated writes wear the internal SSD
- Best when: You run agent workflows frequently and need the speed

**Option B: KV cache on external SSD (slower, less wear)**
- Pro: Protects internal SSD from write wear
- Con: Slower KV cache restore (USB-C NVMe is still fast, ~1000-3000 MB/s)
- Best when: You want to maximize internal SSD lifespan

**Option C: Disable KV cache SSD tiering (no wear, no speed benefit)**
- Pro: Zero additional SSD writes
- Con: Agent workflows that revisit context must recompute KV cache
- Best when: You have enough RAM for the full KV cache in memory

**Recommendation**: For most users, Option A is fine — the internal SSD is
fast and the wear from KV cache is manageable. If you run agents 24/7, consider
Option B. If you never use agent workflows, Option C.

## How to set up external storage

### Prerequisites

- External NVMe SSD in a USB-C (USB 3.2 Gen 2x2 or USB4) or Thunderbolt 4/5
  enclosure. SATA SSDs via FireWire/eSATA are too slow for model loading.
- Format as APFS (enables TRIM, which is critical for SSD longevity)
- Verify TRIM is enabled: `system_profiler SPSerialATADataType | grep "TRIM"`
  or check System Information > Hardware > NVMExpress > your drive

### Interface speed comparison

| Interface | Real-world speed | Good for model storage? |
|-----------|-----------------|------------------------|
| Thunderbolt 5 | ~6,000 MB/s | Excellent |
| Thunderbolt 4 / USB4 | ~3,000 MB/s | Excellent |
| USB-C 3.2 Gen 2x2 | ~2,000 MB/s | Good |
| USB-C 3.2 Gen 2 | ~1,000 MB/s | Acceptable |
| FireWire 800 | ~100 MB/s | Too slow |
| eSATA | ~600 MB/s | Marginal (OK for cold storage, slow for loading) |

Model loading reads the full model file sequentially. A 45 GB model loads in:
- Thunderbolt 4: ~15 seconds
- USB-C 3.2 Gen 2: ~45 seconds
- eSATA: ~75 seconds (usable but slow)

### Step 1: Move HuggingFace cache to external SSD

```bash
# Create model directory on external SSD
mkdir -p /Volumes/ExternalSSD/huggingface

# Set HF_HOME to the external path
# Add to .env or your shell profile:
export HF_HOME=/Volumes/ExternalSSD/huggingface

# For this repo, set LOCAL_CACHE_DIR:
export LOCAL_CACHE_DIR=/Volumes/ExternalSSD/huggingface
```

For this repo's scripts, add to `.env`:
```bash
# Use external SSD for model storage (protects internal SSD)
LOCAL_CACHE_DIR=/Volumes/ExternalSSD/huggingface
```

### Step 2: Move Docker VM to external SSD (Mac)

In Docker Desktop:
1. Settings > Resources > Advanced
2. Change "Disk image location" to your external SSD

Or via CLI:
```bash
# Stop Docker
osascript -e 'quit app "Docker"'

# Move the Docker VM disk image
mv ~/Library/Containers/com.docker.docker/Data/vms/0 /Volumes/ExternalSSD/DockerData

# Create a symlink
ln -s /Volumes/ExternalSSD/DockerData ~/Library/Containers/com.docker.docker/Data/vms/0

# Restart Docker
open -a Docker
```

### Step 3: Configure MLX model directory

For Apple Silicon MLX runtime:
```bash
# Point vllm-mlx to models on external SSD
export MLX_MODEL_DIR=/Volumes/ExternalSSD/models
```

### Step 4: Configure KV cache location (optional)

For vllm-mlx, check if KV cache SSD tiering can be redirected:
```bash
# If vllm-mlx supports a cache directory env var:
export MLX_CACHE_DIR=/Volumes/ExternalSSD/kv-cache
```

For oMLX, configure the cache directory in its settings.

## Monitoring SSD health

### Check write totals

```bash
# Install smartmontools
brew install smartmontools

# Check SSD health and total bytes written
sudo smartctl -a /dev/disk0 | grep "Data Units Written"
```

### Monitor write rate over time

```bash
# Record current bytes written
sudo smartctl -a /dev/disk0 | grep "Data Units Written" > ssd-baseline.txt

# ... run models for a day ...

# Check again and compare
sudo smartctl -a /dev/disk0 | grep "Data Units Written"
```

### Activity Monitor fallback

If smartctl doesn't work on your Mac, use Activity Monitor:
1. Open Activity Monitor
2. View > Columns > Bytes Written
3. Sort by Bytes Written to see which processes write the most

## Cold storage for infrequently used models

If you have models you rarely use but don't want to re-download:

```bash
# Move to cold storage (external SSD or even external HDD)
mv ~/.cache/huggingface/hub/models--Qwen--Qwen3-Next-80B-A3B /Volumes/ColdStorage/models/

# When needed, move back or symlink:
ln -s /Volumes/ColdStorage/models/models--Qwen--Qwen3-Next-80B-A3B ~/.cache/huggingface/hub/
```

For true cold storage (models you might use someday but not now), an
external HDD is fine — you only pay the loading time once when you need it.

## Summary

| What | Where | Why |
|------|-------|-----|
| Model weights | External NVMe SSD | Large, write-once, read-many |
| Docker images/volumes | External NVMe SSD | Large, mostly static |
| Downloads | External NVMe SSD | Avoid writing to internal |
| KV cache SSD tiering | Internal SSD (speed) or external (longevity) | Trade-off |
| Swap | Internal SSD | Speed-critical |
| OS + apps | Internal SSD | System files |

The single biggest win: **move the HuggingFace cache to an external SSD**.
This eliminates 80% of the write traffic from model downloads and storage.
