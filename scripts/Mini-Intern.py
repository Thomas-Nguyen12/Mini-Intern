# -*- coding: utf-8 -*-
"""
Mini-Intern.py - build and train the small GPT language model on your own PC.

This script is standalone: it never downloads or tokenizes data. It only needs the
three files that load_data.py wrote (possibly on another machine, at an earlier
time). Copy that whole folder to this PC, e.g.:

    data_code/
        train.bin          uint32 token ids
        validation.bin
        meta.json          tokenizer name, dtype, token counts

Then run:

    python -u Mini-Intern.py

Outputs go to RUN_DIR (default: runs/<MODEL_SIZE>/):
    ckpt.pt                  latest full checkpoint (model + optimizer + step), used to resume
    best_model_params.pt     weights with the best validation loss (use this for fine-tuning)
    loss_log.csv             step, train loss, val loss, lr, elapsed seconds
    loss_curve.png

Long runs on a PC get interrupted (sleep, power cuts, Ctrl-C). Training saves a
checkpoint every eval_interval steps and resumes automatically when you run the
same command again. Press Ctrl-C ONCE to stop cleanly: the current step finishes,
a checkpoint is saved, and the script exits. Press it twice to force quit.

Requirements:
    pip install numpy tqdm matplotlib torch      (see below for NVIDIA RTX 50-series)
    pip install tiktoken                         (optional: only for the text samples at the end)
  * RTX 50-series cards (e.g. 5070) need a PyTorch build for CUDA 12.8 or newer,
    e.g. pip install torch --index-url https://download.pytorch.org/whl/cu128
    The script prints a warning if your build doesn't support the GPU.
  * tiktoken downloads the tokenizer vocabulary on first use. If the PC is offline,
    training still works; only the final sample generation is skipped.

Environment variables (all optional):
    DATA_DIR      folder with train.bin / validation.bin / meta.json  (default: data_code)
    RUN_DIR       where checkpoints and logs go                       (default: runs/<MODEL_SIZE>)
    MODEL_SIZE    "base" (~163M) or "small" (~77M)                    (default: base)
    MAX_STEPS     total optimizer steps                               (default: 4000)
    BATCH_SIZE    micro-batch size; default is chosen from GPU memory (4 on a 12 GB card)
    RESUME        set to 0 to ignore an existing ckpt.pt and start over
    COMPILE       set to 1 to use torch.compile (CUDA only)

Examples:
    MAX_STEPS=20 MODEL_SIZE=small python -u Mini-Intern.py        # quick speed test
    MODEL_SIZE=small MAX_STEPS=1900 python -u Mini-Intern.py      # ~1B tokens
    DATA_DIR=/mnt/usb/data_code python -u Mini-Intern.py

Keep MAX_STEPS the same when resuming (it defines the learning-rate schedule).
"""

import os
import sys
import json
import math
import time
import signal
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from contextlib import nullcontext
from tqdm.auto import tqdm

# ----------------------------------------------------------------------------
# Step 1: Locate and validate the data made by load_data.py
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_DIR = f"{BASE_DIR}/data_code/" 

TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "validation.bin")
META_PATH = os.path.join(DATA_DIR, "meta.json")

missing = [p for p in (TRAIN_BIN, VAL_BIN, META_PATH) if not os.path.exists(p)]
if missing:
    sys.exit("Missing data files:\n  " + "\n  ".join(missing) +
             "\n\nRun load_data.py (on any machine) and copy the whole data folder here, "
             "then point DATA_DIR at it (default: ./data_code).")

with open(META_PATH) as f:
    meta = json.load(f)
TOKENIZER_NAME = meta["tokenizer"]
TOKEN_DTYPE = np.dtype(meta["dtype"])
N_VOCAB = meta.get("n_vocab", 100277)   # cl100k_base

# Catch half-copied files: sizes must match the token counts recorded by load_data.py
for path, key in ((TRAIN_BIN, "train_tokens"), (VAL_BIN, "val_tokens")):
    expected = meta[key] * TOKEN_DTYPE.itemsize
    actual = os.path.getsize(path)
    if actual != expected:
        sys.exit(f"{path} is {actual:,} bytes but meta.json says {expected:,}. "
                 f"The copy is probably incomplete - copy it again.")

# The tokenizer is only needed for the sample text at the end, so it is optional.
try:
    import tiktoken
    enc = tiktoken.get_encoding(TOKENIZER_NAME)
except Exception as e:
    enc = None
    print(f"Note: tokenizer '{TOKENIZER_NAME}' unavailable ({type(e).__name__}); training will "
          f"work, but the final text samples will be skipped.", flush=True)

print(f"Data: {DATA_DIR} | {meta['train_tokens']:,} train / {meta['val_tokens']:,} val tokens "
      f"({TOKENIZER_NAME}, {TOKEN_DTYPE.name})", flush=True)

# ----------------------------------------------------------------------------
# Step 2: Settings, device and run folder
#
# Everything is counted in OPTIMIZER STEPS (one step = one full gradient-
# accumulation cycle). Tokens per step is fixed at TOKENS_PER_STEP (524,288); the
# micro-batch size only changes how it is split, so a smaller BATCH_SIZE on a
# small GPU does not change the training. 4000 steps is ~2.1B tokens.
# ----------------------------------------------------------------------------
MODEL_SIZES = {
    "small": dict(n_layer=8,  n_head=8,  n_embd=512),    # ~77M parameters
    "base":  dict(n_layer=12, n_head=12, n_embd=768),    # ~163M parameters
}
MODEL_SIZE = os.environ.get("MODEL_SIZE", "base")
assert MODEL_SIZE in MODEL_SIZES, f"MODEL_SIZE must be one of {list(MODEL_SIZES)}"

RUN_DIR = os.path.abspath(os.environ.get("RUN_DIR") or os.path.join("runs", MODEL_SIZE))
os.makedirs(RUN_DIR, exist_ok=True)
CKPT_PATH = os.path.join(RUN_DIR, "ckpt.pt")
BEST_MODEL_PATH = os.path.join(RUN_DIR, "best_model_params.pt")
LOG_CSV = os.path.join(RUN_DIR, "loss_log.csv")
LOSS_PLOT_PATH = os.path.join(RUN_DIR, "loss_curve.png")

learning_rate = 6e-4
min_lr = 6e-5
max_steps = int(os.environ.get("MAX_STEPS", 4000))   # optimizer steps
warmup_steps = min(200, max_steps // 10)             # optimizer steps
eval_interval = min(250, max(1, max_steps // 4))     # optimizer steps between evaluations + checkpoints
eval_iters = 100          # batches per split per evaluation
block_size = 1024         # context window
TOKENS_PER_STEP = 524_288

# --- device ---
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"        # Apple Silicon GPU
else:
    device = "cpu"
device_type = 'cuda' if device == 'cuda' else 'cpu'   # used for autocast / pinned memory

if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"device: cuda ({props.name}, {props.total_memory / 1e9:.1f} GB)", flush=True)
    cap = f"{props.major}{props.minor}"
    if not any(a in (f"sm_{cap}", f"compute_{cap}") for a in torch.cuda.get_arch_list()):
        print(f"WARNING: this PyTorch build ({torch.__version__}) lists no kernels for your GPU "
              f"(sm_{cap}). If you see 'no kernel image' errors, install a newer PyTorch build "
              f"(RTX 50-series need CUDA 12.8+).", flush=True)
    torch.set_float32_matmul_precision("high")
else:
    msg = f"device: {device}"
    if device == "cpu":
        msg += " - training the base model on a CPU would take weeks."
        if shutil.which("nvidia-smi"):
            msg += (" An NVIDIA GPU was detected but PyTorch can't use it: install a CUDA build of "
                    "PyTorch matching your driver (RTX 50-series need CUDA 12.8+).")
    print(msg, flush=True)

# --- micro-batch size: default chosen from GPU memory (the 100k-token vocabulary makes logits big) ---
def default_batch_size():
    if device == "cuda":
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        return 16 if gb >= 30 else 8 if gb >= 20 else 4
    return 4

batch_size = int(os.environ.get("BATCH_SIZE", default_batch_size()))
gradient_accumulation_steps = max(1, TOKENS_PER_STEP // (batch_size * block_size))

if device == "cuda":
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
else:
    dtype = 'float32'     # mixed precision is unreliable on MPS/CPU
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype) if device == 'cuda' else nullcontext()

torch.set_default_device(device)

tokens_per_step = batch_size * gradient_accumulation_steps * block_size
train_tokens = meta["train_tokens"]
print(f"batch_size {batch_size} x {gradient_accumulation_steps} accumulation x {block_size} tokens = "
      f"{tokens_per_step:,} tokens/step | {max_steps} steps = {tokens_per_step * max_steps / 1e9:.2f}B tokens "
      f"({tokens_per_step * max_steps / train_tokens:.2f} epochs over the train file)", flush=True)

# ----------------------------------------------------------------------------
# Step 3: Input-output batches
# ----------------------------------------------------------------------------
def get_batch(split_name):
    # Recreate np.memmap every batch to avoid a memory leak
    path = TRAIN_BIN if split_name == 'train' else VAL_BIN
    data = np.memmap(path, dtype=TOKEN_DTYPE, mode='r')
    # Build indices on the CPU so the default CUDA device doesn't force a sync
    # for every data[i:...] slice.
    ix = torch.randint(len(data) - block_size, (batch_size,), device="cpu")
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix.tolist()])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix.tolist()])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# ----------------------------------------------------------------------------
# Step 4: Model architecture
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

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

def estimate_loss(model):
    out = {}
    model.eval()
    with torch.inference_mode():
        for split_name in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split_name)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split_name] = losses.mean().item()  # plain float
    model.train()
    return out

# ----------------------------------------------------------------------------
# Step 5: Build model + optimizer, and resume from a checkpoint if there is one
# ----------------------------------------------------------------------------
config = GPTConfig(
    vocab_size=100352,    # cl100k_base has 100,277 tokens; padded up to a multiple of 128
    block_size=block_size,
    dropout=0.1,
    bias=True,
    **MODEL_SIZES[MODEL_SIZE],
)
assert N_VOCAB <= config.vocab_size, "tokenizer vocabulary is larger than config.vocab_size"

from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

raw_model = GPT(config).to(device)   # checkpoints are always saved from the uncompiled model
print(f"Model ({MODEL_SIZE}) parameters: {sum(p.numel() for p in raw_model.parameters()) / 1e6:.1f}M", flush=True)

# No weight decay on biases / LayerNorm weights (1-D params), as in nanoGPT
decay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() >= 2]
nodecay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay_params, "weight_decay": 0.1},
     {"params": nodecay_params, "weight_decay": 0.0}],
    lr=learning_rate, betas=(0.9, 0.95), eps=1e-9,
)

# start_factor near 0 gives a real warmup (default is 1/3)
scheduler_warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_steps)
scheduler_decay = CosineAnnealingLR(optimizer, T_max=max(1, max_steps - warmup_steps), eta_min=min_lr)
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])

# Only needed for fp16 on CUDA
scaler = torch.amp.GradScaler("cuda", enabled=(device == 'cuda' and dtype == 'float16'))

start_step = 0
best_val_loss = float('inf')
eval_steps, train_loss_list, validation_loss_list = [], [], []

if os.path.exists(CKPT_PATH) and os.environ.get("RESUME", "1") != "0":
    # weights_only=False: this is our own checkpoint (it holds optimizer/scheduler state)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    if ckpt["config"] != asdict(config):
        sys.exit(f"{CKPT_PATH} was made with a different model config. Use a different RUN_DIR "
                 f"or set RESUME=0 to start over.")
    if ckpt["max_steps"] != max_steps:
        sys.exit(f"{CKPT_PATH} was made with MAX_STEPS={ckpt['max_steps']} (you set {max_steps}). "
                 f"Use the same value to resume, or set RESUME=0 to start over.")
    raw_model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start_step = ckpt["step"]
    best_val_loss = ckpt["best_val_loss"]
    eval_steps, train_loss_list, validation_loss_list = ckpt["eval_steps"], ckpt["train_loss"], ckpt["val_loss"]
    del ckpt
    print(f"Resuming from step {start_step} (best val loss so far {best_val_loss:.4f})", flush=True)
elif os.path.exists(CKPT_PATH):
    print(f"RESUME=0: ignoring existing {CKPT_PATH}; it will be overwritten.", flush=True)

torch.manual_seed(42 + start_step)   # different random batches after a resume

train_model = raw_model
if os.environ.get("COMPILE") == "1" and device == "cuda":
    train_model = torch.compile(raw_model)   # first step is slower while it compiles

def save_checkpoint(step):
    """step = number of completed optimizer steps. Written atomically."""
    tmp = CKPT_PATH + ".tmp"
    torch.save({
        "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "step": step, "best_val_loss": best_val_loss,
        "eval_steps": eval_steps, "train_loss": train_loss_list, "val_loss": validation_loss_list,
        "config": asdict(config), "max_steps": max_steps,
    }, tmp)
    os.replace(tmp, CKPT_PATH)

def log_eval(step, train_loss, val_loss, lr):
    new_file = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a") as f:
        if new_file:
            f.write("step,train_loss,val_loss,lr,elapsed_s\n")
        f.write(f"{step},{train_loss:.5f},{val_loss:.5f},{lr:.8f},{time.time() - t0:.0f}\n")

# Ctrl-C (or kill) asks for a clean stop: finish the step, save, exit.
stop_requested = False
def _request_stop(signum, frame):
    global stop_requested
    stop_requested = True
    print("\nStop requested: finishing the current step, saving a checkpoint, then exiting. "
          "(Press Ctrl-C again to force quit.)", flush=True)
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, signal.SIG_DFL)
        except (ValueError, OSError):
            pass

for s in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(s, _request_stop)
    except (ValueError, OSError):
        pass

# ----------------------------------------------------------------------------
# Step 6: Train
# ----------------------------------------------------------------------------
t0 = time.time()
stopped_early = False
raw_model.train()

for step in tqdm(range(start_step, max_steps), initial=start_step, total=max_steps):
    # one iteration = one optimizer step = gradient_accumulation_steps micro-batches
    for micro_step in range(gradient_accumulation_steps):
        X, y = get_batch("train")
        with ctx:
            logits, loss = train_model(X, y)
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

    # unscale before clipping so the clip threshold applies to the real
    # gradients (no-op when the scaler is disabled, e.g. bf16)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=0.5)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()  # once per optimizer step, after optimizer.step()

    done = step + 1
    just_saved = False
    if done % eval_interval == 0 or done == max_steps:
        losses = estimate_loss(train_model)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Step {done}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
              f"lr {lr_now:.6f}", flush=True)
        if not (math.isfinite(losses['train']) and math.isfinite(losses['val'])):
            sys.exit("Loss is NaN/inf - training has diverged. Your last good checkpoint is "
                     f"{CKPT_PATH}; try a lower learning rate or RESUME=0 with a fresh start.")
        eval_steps.append(done)
        train_loss_list.append(losses['train'])
        validation_loss_list.append(losses['val'])
        log_eval(done, losses['train'], losses['val'], lr_now)

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(raw_model.state_dict(), BEST_MODEL_PATH)
        save_checkpoint(done)
        just_saved = True

    if stop_requested:
        if not just_saved:
            save_checkpoint(done)
        print(f"Checkpoint saved at step {done}. Run the same command again to resume.", flush=True)
        stopped_early = True
        break

# ----------------------------------------------------------------------------
# Step 7: Plot the loss curve (saved to a PNG so no display is needed)
# ----------------------------------------------------------------------------
if eval_steps:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.plot(eval_steps, train_loss_list, 'g', label='train_loss')
    plt.plot(eval_steps, validation_loss_list, 'r', label='validation_loss')
    plt.xlabel("Optimizer step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(LOSS_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()

if stopped_early:
    sys.exit(0)

print(f"\nTraining finished. Best validation loss {best_val_loss:.4f}.\n"
      f"Best weights: {BEST_MODEL_PATH}", flush=True)

# ----------------------------------------------------------------------------
# Step 8: Sample from the trained model
#
# The model was trained on raw text and code (no instruction template), so we
# prompt it with the beginning of a document and let it continue. It will not
# follow instructions until it has been fine-tuned on instruction data.
# ----------------------------------------------------------------------------
if enc is None:
    print("Skipping text samples (tokenizer unavailable).")
    sys.exit(0)

model = GPT(config)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=torch.device(device)))
model = model.to(device)
model.eval()

def complete(text, max_new_tokens=200, temperature=0.8, top_k=50):
    context = torch.tensor(enc.encode_ordinary(text)).unsqueeze(dim=0).to(device)
    y = model.generate(context, max_new_tokens, temperature=temperature, top_k=top_k)
    return enc.decode(y.squeeze().tolist())

# Code completion (lower temperature keeps code more coherent)
print(complete('def fibonacci(n):\n    """Return the nth Fibonacci number."""\n', temperature=0.4))
print("\n" + "=" * 80 + "\n")

# Script-style opener
print(complete("# Read a CSV file and print the number of rows\nimport csv\n", temperature=0.4))
print("\n" + "=" * 80 + "\n")

# Educational-text opener
print(complete("How to Stay Healthy\n\nStaying healthy doesn't have to be complicated. Here are a few simple steps:"))
