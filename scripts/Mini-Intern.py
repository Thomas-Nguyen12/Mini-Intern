# -*- coding: utf-8 -*-
"""
Small Language Model from Scratch - trained on a pickled Cosmopedia subset

We build a Small Language Model (SLM) from scratch, keeping the parameter
count at roughly 50-60 million.

The dataset comes from a pickled DatasetDict (default: ds.pkl, written by
load_data.py) with "train" and "validation" splits and a "text" column. Each
document is tokenized with the GPT-2 BPE tokenizer and an end-of-text token is
appended, so the model learns where one document ends and the next begins.

Usage:
    python Mini-Intern.py                          # reads ./ds.pkl
    DATASET_PKL=/path/to/other.pkl python Mini-Intern.py

Note: tiktoken downloads the GPT-2 vocabulary on first use. If your compute
node has no internet, run `python -c "import tiktoken; tiktoken.get_encoding('gpt2')"`
once on a login node (same TIKTOKEN_CACHE_DIR / home dir) beforehand.

## Step 1: Import the Dataset
"""

import os
import pickle
from datasets import DatasetDict

# Path to the pickled DatasetDict
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PKL_PATH = os.environ.get(f"{BASE_DIR}/data/", "ds.pkl")

# Everything produced by this script (token files, checkpoint, plot) lives here
# so it can never be confused with files from an earlier Alpaca/TinyStories run.
DATA_DIR = "data_cosmopedia"
os.makedirs(DATA_DIR, exist_ok=True)
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "validation.bin")
BEST_MODEL_PATH = os.path.join(DATA_DIR, "best_model_params.pt")
LOSS_PLOT_PATH = os.path.join(DATA_DIR, "loss_curve.png")

# Use the CPUs Slurm gave us, if any
NUM_PROC = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

# The pickle is only needed to build the .bin files. If train.bin and
# validation.bin already exist (from a previous run), we skip loading it.
NEED_TOKENIZE = not (os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN))

if NEED_TOKENIZE:
    print(f"Loading dataset from {PKL_PATH} ...")
    with open(PKL_PATH, "rb") as f:
        ds = pickle.load(f)
    for name in ("train", "validation"):
        assert name in ds, f"pickled DatasetDict has no '{name}' split"
        assert "text" in ds[name].column_names, f"'{name}' split has no 'text' column"
    # Keep only the text column so tokenization sees nothing else
    ds = DatasetDict({k: v.select_columns(["text"]) for k, v in ds.items()})
    print({k: f"{len(v):,} rows" for k, v in ds.items()})
else:
    print(f"{TRAIN_BIN} and {VAL_BIN} found - skipping dataset loading and tokenization. "
          f"Delete them if you want to rebuild from {PKL_PATH}.")

"""## Step 2: Tokenize the Dataset

In this step, we will do the following:

(1) Tokenize each document's `text` into token IDs with the GPT-2 BPE
    tokenizer, and append the end-of-text token after every document.

(2) Create "train.bin" and "validation.bin" where we store the tokenIDs from
    the entire dataset.

(3) We make sure the tokenIDs are stored on disk, rather than in RAM, for
    efficient computation.
"""

import tiktoken
import numpy as np
from tqdm.auto import tqdm

enc = tiktoken.get_encoding("gpt2")

# Some functions from https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py

def process(example):
    ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)                    # mark the end of this document
    out = {'ids': ids, 'len': len(ids)}
    return out

if NEED_TOKENIZE:
    tokenized = ds.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=NUM_PROC,
    )
    # concatenate all the ids in each dataset into one large file we can use for training
    for split_name, dset in tokenized.items():
        arr_len = int(np.sum(dset['len'], dtype=np.uint64))
        filename = os.path.join(DATA_DIR, f'{split_name}.bin')
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        # num_shards can't exceed the dataset's length
        total_batches = min(1024, len(dset))

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"{filename}: {arr_len:,} tokens")

"""## Step 3: Create Input-Output batches for the dataset"""

import torch

# Some functions from https://github.com/karpathy/nanoGPT/blob/master/train.py with slight modifications
# block size = context window
def get_batch(split_name):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split_name == 'train':
        data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode='r')
    else:
        data = np.memmap(VAL_BIN, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

"""## Step 4: Define the SLM Model Architecture

(unchanged - the transformer itself doesn't care what dataset it was trained on)
"""

import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from contextlib import nullcontext

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
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
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
        """
        Generate tokens given a conditioning sequence.
        idx: Tensor of shape (B, T)
        """
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

# Cosmopedia documents are long, but we train on a packed token stream
# (train.bin), so block_size is just the context window and can be changed
# freely. Raise it (e.g. 512) if you have the GPU memory; lower it if you OOM.
config = GPTConfig(
    vocab_size=50257,     # use the tokenizer's vocab size
    block_size=256,       # context window
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True
)

model = GPT(config)

"""## Step 5: Define the loss function (unchanged)"""

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
            out[split_name] = losses.mean()
    model.train()
    return out

"""## Step 6: Define SLM Training Configuration Part 1"""

from contextlib import nullcontext

# NOTE: the original had learning_rate=1e-4 and min_lr=5e-4. A minimum LR
# higher than the starting LR makes the cosine schedule *increase* the LR
# over training, which is almost certainly unintended. Using the usual
# nanoGPT-style values for a model this size: peak 6e-4, decaying to 6e-5.
learning_rate = 6e-4
max_iters = 20000
warmup_steps = 1000
min_lr = 6e-5
eval_iters = 500
batch_size = 32
block_size = 256  # keep in sync with config.block_size above

gradient_accumulation_steps = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu'

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.set_default_device(device)
torch.manual_seed(42)

"""## Step 7: Define SLM Training Configuration Part 2 (unchanged)"""

from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9)

scheduler_warmup = LinearLR(optimizer, total_iters=warmup_steps)
scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr)
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

"""## Step 8: Pre-train the SLM"""

best_val_loss = float('inf')
train_loss_list, validation_loss_list = [], []

model = model.to(device)

for epoch in tqdm(range(max_iters)):
    if epoch % eval_iters == 0 and epoch != 0:
        losses = estimate_loss(model)
        print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"The current learning rate: {optimizer.param_groups[0]['lr']:.5f}")
        train_loss_list += [losses['train']]
        validation_loss_list += [losses['val']]

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    X, y = get_batch("train")
    X, y = X.to(device), y.to(device)

    with ctx:
        logits, loss = model(X, y)
        loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

    if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    scheduler.step()

"""## Step 9: Plot the SLM Loss Function

Saved to a PNG because compute nodes have no display.
"""

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

train_loss_list_converted = [i.cpu().detach() for i in train_loss_list]
validation_loss_list_converted = [i.cpu().detach() for i in validation_loss_list]

plt.plot(train_loss_list_converted, 'g', label='train_loss')
plt.plot(validation_loss_list_converted, 'r', label='validation_loss')
plt.xlabel("Steps - Every eval_iters epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(LOSS_PLOT_PATH, dpi=150, bbox_inches="tight")
plt.close()

"""## Step 10: Run SLM Inference on our trained model

The model was trained on raw Cosmopedia text (no instruction template), so we
prompt it with the beginning of a document and let it continue.
"""

model = GPT(config)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=torch.device(device)))
model = model.to(device)
model.eval()

def complete(text, max_new_tokens=200, temperature=0.8, top_k=50):
    context = torch.tensor(enc.encode_ordinary(text)).unsqueeze(dim=0).to(device)
    y = model.generate(context, max_new_tokens, temperature=temperature, top_k=top_k)
    return enc.decode(y.squeeze().tolist())

# Story-style opener (matches the "stories" config)
print(complete("Once upon a time, in a small village by the sea, there lived a young girl named"))
print("\n" + "=" * 80 + "\n")

# WikiHow-style opener (matches the "wikihow" config)
print(complete("How to Stay Healthy\n\nStaying healthy doesn't have to be complicated. Here are a few simple steps:"))

# from google.colab import runtime
# runtime.unassign()