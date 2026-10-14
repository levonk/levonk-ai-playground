#!/usr/bin/env python3
"""System One scorer server.

Serves pngwn/system-one-qwen3.5-4b-scorer-v2b — a LoRA adapter on
Qwen/Qwen3.5-4B-Base with a scalar sequence-classification head. Each option
of a typed question gets a scalar score in a single forward pass; the scores
are then softmaxed per question (with a fitted calibration temperature).
There is no autoregressive generation.

The encode/pad/score logic here mirrors the upstream system_one.py in
pngwn/system-one-qwen3.5-4b-scorer: the prompt tail (question + option) is
never truncated, the state is truncated to make room, and the head pools the
last non-pad token (right padding).

API:
    GET  /health     -> {"status": "ok", "model": ...}
    GET  /v1/models  -> OpenAI-style model list (for test-query.sh probes)
    POST /score      -> {"state": str, "question": str, "options": [str, ...],
                         "temperature": float (optional, default S1_TEMPERATURE)}
                     <- {"scores": [...], "probabilities": [...],
                         "answer_index": int, "answer": str, ...}

Environment variables:
    S1_BASE_MODEL    base checkpoint (default: Qwen/Qwen3.5-4B-Base)
    S1_ADAPTER       PEFT adapter repo (default: pngwn/system-one-qwen3.5-4b-scorer-v2b)
    S1_MAX_LEN       max sequence length (default: 384, matches training)
    S1_TEMPERATURE   calibration temperature (default: 2.35, fitted on val;
                     pass "temperature": 0 in the request for raw scores)
    S1_BATCH_SIZE    options per forward pass (default: 8)
    S1_MAX_OPTIONS   max options per request (default: 64)
    S1_DEVICE        cuda|mps|cpu (default: auto)
    S1_DTYPE         bf16|fp16|fp32 (default: bf16 on cuda/mps, fp32 on cpu)
    PORT             listen port (default: 8007; --port overrides)
"""

import argparse
import json
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

BASE_MODEL = os.environ.get("S1_BASE_MODEL", "Qwen/Qwen3.5-4B-Base")
ADAPTER = os.environ.get("S1_ADAPTER", "pngwn/system-one-qwen3.5-4b-scorer-v2b")
MAX_LEN = int(os.environ.get("S1_MAX_LEN", "384"))
TEMPERATURE = float(os.environ.get("S1_TEMPERATURE", "2.35"))
BATCH_SIZE = int(os.environ.get("S1_BATCH_SIZE", "8"))
MAX_OPTIONS = int(os.environ.get("S1_MAX_OPTIONS", "64"))


def pick_device():
    want = os.environ.get("S1_DEVICE", "auto").lower()
    if want != "auto":
        return want
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device):
    want = os.environ.get("S1_DTYPE", "").lower()
    if want in ("bf16", "bfloat16"):
        return torch.bfloat16
    if want in ("fp16", "float16", "half"):
        return torch.float16
    if want in ("fp32", "float32"):
        return torch.float32
    return torch.float32 if device == "cpu" else torch.bfloat16


def load_model(device, dtype):
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # The adapter repo ships a complete tokenizer; fall back to the base
    # checkpoint's tokenizer if it is missing.
    try:
        tok = AutoTokenizer.from_pretrained(ADAPTER)
    except OSError:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    try:
        from transformers import Qwen3_5TextForSequenceClassification
        cls = Qwen3_5TextForSequenceClassification
    except ImportError:
        cls = AutoModelForSequenceClassification
    model = cls.from_pretrained(BASE_MODEL, num_labels=1, dtype=dtype)

    # pad/eos are unset on the base checkpoint; force the tokenizer's ids onto
    # the top config and the text sub-config or padding ids go unset.
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    tc = model.config.get_text_config()
    tc.pad_token_id = tok.pad_token_id
    tc.eos_token_id = tok.eos_token_id
    tc.num_labels = 1
    model.num_labels = 1

    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval().to(device)
    return tok, model


def encode(tok, state, question, option, max_len):
    """Token ids for one scoring instance.

    The tail (question + option) is kept intact and the state is truncated
    from its end, because the head pools the last non-pad token.
    """
    tail = tok("\n\nQuestion:\n" + question + "\n\nOption:\n" + option,
               add_special_tokens=False)["input_ids"]
    if len(tail) >= max_len:
        return tail[-max_len:]
    head = tok("State:\n" + state, add_special_tokens=False)["input_ids"]
    return head[: max_len - len(tail)] + tail


def pad_batch(seqs, pad_id, max_len):
    """Right padding: the head pools the last non-pad token."""
    L = min(max(len(s) for s in seqs), max_len)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[:L]
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, : len(s)] = 1
    return ids, attn


@torch.no_grad()
def score_options(model, tok, device, state, question, options):
    seqs = [encode(tok, state, question, o, MAX_LEN) for o in options]
    logits = []
    for i in range(0, len(seqs), BATCH_SIZE):
        ids, attn = pad_batch(seqs[i: i + BATCH_SIZE], tok.pad_token_id, MAX_LEN)
        out = model(input_ids=ids.to(device), attention_mask=attn.to(device))
        logits.extend(out.logits.squeeze(-1).float().cpu().tolist())
    return logits


def softmax(xs, temperature=1.0):
    xs = [x / temperature for x in xs]
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


class Handler(BaseHTTPRequestHandler):
    server_version = "system-one-scorer/1.0"

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "model": ADAPTER,
                             "base_model": BASE_MODEL})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{
                "id": ADAPTER, "object": "model",
                "owned_by": ADAPTER.split("/")[0]}]})
        else:
            self._json(404, {"error": "not found",
                             "hint": "POST /score with {state, question, options}"})

    def do_POST(self):
        if self.path not in ("/score", "/v1/score"):
            self._json(404, {"error": "not found",
                             "hint": "POST /score with {state, question, options}"})
            return
        try:
            n = min(int(self.headers.get("Content-Length", "0")), 1 << 20)
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._json(400, {"error": "invalid JSON body"})
            return

        options = body.get("options")
        if not isinstance(options, list) or not options:
            self._json(400, {"error": "'options' must be a non-empty list of strings"})
            return
        options = [str(o) for o in options]
        if len(options) > MAX_OPTIONS:
            self._json(400, {"error": f"too many options (max {MAX_OPTIONS})"})
            return

        state = str(body.get("state", ""))
        question = str(body.get("question", ""))
        temperature = float(body.get("temperature", TEMPERATURE))

        with LOCK:
            logits = score_options(MODEL, TOK, DEVICE, state, question, options)
        probs = softmax(logits, temperature) if temperature > 0 else softmax(logits)
        best = max(range(len(probs)), key=probs.__getitem__)
        self._json(200, {
            "model": ADAPTER,
            "options": options,
            "scores": logits,
            "probabilities": probs,
            "answer_index": best,
            "answer": options[best],
            "temperature": temperature,
        })


def main():
    ap = argparse.ArgumentParser(description="System One scorer server")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("PORT", "8007")))
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    global TOK, MODEL, DEVICE, LOCK
    DEVICE = pick_device()
    dtype = pick_dtype(DEVICE)
    print(f"Loading {BASE_MODEL} + adapter {ADAPTER} "
          f"on {DEVICE} ({dtype}) ...", flush=True)
    TOK, MODEL = load_model(DEVICE, dtype)
    LOCK = threading.Lock()

    print(f"Serving on http://{args.host}:{args.port}  (POST /score)", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
