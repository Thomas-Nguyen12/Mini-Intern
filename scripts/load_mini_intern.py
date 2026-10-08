# -*- coding: utf-8 -*-
"""
load_mini_intern.py - load a trained Mini-Intern checkpoint for inference only.

No training data, meta.json, optimizer or Drive mounting needed. Just the
checkpoint file written by Mini-Intern.py (best_model_params.pt).

Usage:
    CHECKPOINT=/path/to/best_model_params.pt MODEL_SIZE=base python -i load_mini_intern.py

Environment variables (all optional):
    CHECKPOINT    path to best_model_params.pt   (default: best_model_params.pt)
    MODEL_SIZE    "small" or "base"; must match the size the checkpoint was trained with (default: base)

Import from another script / app:
    from load_mini_intern import model, enc, complete, device
"""

import os
import math
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

CHECKPOINT = os.environ.get("CHECKPOINT", "best_model_params.pt")
MODEL_SIZE = os.environ.get("MODEL_SIZE", "base")
TOKENIZER_NAME = "cl100k_base"   # same tokenizer used when the token files were built

# ----------------------------------------------------------------------------
# Model architecture (identical to Mini-Intern.py, so the weights load exactly)
# ----------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                               dropout_p=self.attn_dropout.p if self.training else 0.0,
                                               is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

MODEL_SIZES = {
    "small": dict(n_layer=8,  n_head=8,  n_embd=512),
    "base":  dict(n_layer=12, n_head=12, n_embd=768),
}
assert MODEL_SIZE in MODEL_SIZES, f"MODEL_SIZE must be one of {list(MODEL_SIZES)}"

# ----------------------------------------------------------------------------
# Pick whatever device this machine has
# ----------------------------------------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"        # Apple Silicon GPU
else:
    device = "cpu"

# ----------------------------------------------------------------------------
# Build the model and load the weights
# ----------------------------------------------------------------------------
enc = tiktoken.get_encoding(TOKENIZER_NAME)

config = GPTConfig(
    vocab_size=100352,    # must match training (cl100k_base padded to a multiple of 128)
    block_size=1024,
    dropout=0.0,          # inference only
    bias=True,
    **MODEL_SIZES[MODEL_SIZE],
)
assert enc.n_vocab <= config.vocab_size, "tokenizer vocabulary is larger than config.vocab_size"

if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT} (set CHECKPOINT=/path/to/best_model_params.pt)")

model = GPT(config)
state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)   # load on CPU first, then move
state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}      # in case it was saved from a compiled model
model.load_state_dict(state)
model = model.to(device).eval()

print(f"Loaded {MODEL_SIZE} model ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params) "
      f"from {CHECKPOINT} onto {device}", flush=True)

# ----------------------------------------------------------------------------
# Text completion helper
# ----------------------------------------------------------------------------
def complete(text, max_new_tokens=200, temperature=0.8, top_k=50):
    context = torch.tensor(enc.encode_ordinary(text), dtype=torch.long).unsqueeze(0).to(device)
    with torch.inference_mode():
        y = model.generate(context, max_new_tokens, temperature=temperature, top_k=top_k)
    return enc.decode(y.squeeze(0).tolist())

if __name__ == "__main__":
    print(complete("How to Stay Healthy\n\nStaying healthy doesn't have to be complicated. Here are a few simple steps:"))