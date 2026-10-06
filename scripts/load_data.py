# -*- coding: utf-8 -*-
"""
load_data.py - stream, tokenize and save the preprocessed training data.

Streams a mix of Python code, educational web text and synthetic textbooks from
the Hugging Face Hub, tokenizes it with tiktoken "cl100k_base", and writes:

    OUTPUT_DIR/train.bin        uint32 token ids (documents separated by <|endoftext|>)
    OUTPUT_DIR/validation.bin   ~1% of documents, held out
    OUTPUT_DIR/meta.json        tokenizer name, dtype and token counts

Run this first, then run Mini-Intern.py, which builds and trains the model on these files.

Usage:
    python -u load_data.py
    TOTAL_TOKENS=2e7 python -u load_data.py          # quick test (~20M tokens)
    OUTPUT_DIR=/path/to/dir python -u load_data.py
    FORCE=1 python -u load_data.py                   # rebuild even if files exist

Environment variables (all optional):
    OUTPUT_DIR    where the token files go                    (default: data_code)
    TOTAL_TOKENS  training tokens to stream + tokenize        (default: 2.1e9)
    FORCE         set to 1 to rebuild existing token files

Notes:
  * Needs internet access on the first run, and takes roughly an hour or more
    for 2.1B tokens depending on CPU count and connection. Nothing is downloaded
    in full; documents are streamed until each source reaches its token budget.
  * The work is CPU-only. On Colab, run it on a CPU runtime to save compute units.
  * tiktoken downloads its vocabulary on first use. If your machine has no
    internet, run `python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"`
    once somewhere that does (same TIKTOKEN_CACHE_DIR / home dir) beforehand.
"""

import os
import sys
import json
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm.auto import tqdm

# If running in Google Colab, mount Drive (harmless elsewhere). Only matters if
# OUTPUT_DIR points into drive/MyDrive/...
try:
    from google.colab import drive as _colab_drive
    if not os.path.isdir("/content/drive/MyDrive"):
        _colab_drive.mount("/content/drive")
except ImportError:
    pass  # not on Colab (e.g. Slurm)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
DATA_DIR = os.path.abspath(os.environ.get("OUTPUT_DIR", "data_code"))
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "validation.bin")
META_PATH = os.path.join(DATA_DIR, "meta.json")

NUM_PROC = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

TOKENIZER_NAME = "cl100k_base"
TOKEN_DTYPE = np.uint32           # cl100k_base has ~100k tokens, too many for uint16
TOTAL_TOKENS = int(float(os.environ.get("TOTAL_TOKENS", 2.1e9)))  # train tokens across all sources
VAL_EVERY = 100                   # every 100th document goes to validation (~1%)
MAX_CHARS = 200_000               # truncate huge documents before tokenizing
FORCE = os.environ.get("FORCE") == "1"

# Each source is streamed and gets a share of TOTAL_TOKENS proportional to its
# weight. `path` is a Hub dataset id or a local loader name ("json", "parquet",
# "csv", "text"); `data_files` is for local files.
#
# Note: HuggingFaceTB/smollm-corpus "python-edu" is deliberately NOT used. It only
# lists file IDs, and the code itself must be fetched separately from an S3 bucket.
SOURCES = [
    dict(name="python_code", path="codeparrot/codeparrot-clean", config=None,
         data_files=None, text_column="content", weight=0.50),
    dict(name="fineweb_edu", path="HuggingFaceTB/smollm-corpus", config="fineweb-edu-dedup",
         data_files=None, text_column="text", weight=0.35),
    dict(name="cosmopedia_v2", path="HuggingFaceTB/smollm-corpus", config="cosmopedia-v2",
         data_files=None, text_column="text", weight=0.15),
    # Example: your own data (e.g. instruction/tool-call transcripts as JSON lines
    # with a "text" field). Uncomment and give it a weight:
    # dict(name="mine", path="json", config=None,
    #      data_files="data/*.jsonl", text_column="text", weight=0.05),
]

enc = tiktoken.get_encoding(TOKENIZER_NAME)


# ----------------------------------------------------------------------------
# Tokenize + write
# ----------------------------------------------------------------------------
def encode_and_write(texts, doc_start, f_train, f_val):
    """Tokenize a batch of documents and append them to the open train/val files."""
    ids_list = enc.encode_ordinary_batch(texts, num_threads=NUM_PROC)
    train_parts, val_parts = [], []
    for j, ids in enumerate(ids_list):
        ids.append(enc.eot_token)
        arr = np.asarray(ids, dtype=TOKEN_DTYPE)
        (val_parts if (doc_start + j) % VAL_EVERY == 0 else train_parts).append(arr)
    n_train = n_val = 0
    if train_parts:
        chunk = np.concatenate(train_parts)
        f_train.write(chunk.tobytes())
        n_train = len(chunk)
    if val_parts:
        chunk = np.concatenate(val_parts)
        f_val.write(chunk.tobytes())
        n_val = len(chunk)
    return n_train, n_val


def build_bins():
    # Write to temp files first, so a crashed run can't leave a half-written
    # train.bin that the next run (or Mini-Intern.py) mistakes for valid.
    train_tmp, val_tmp = TRAIN_BIN + ".tmp", VAL_BIN + ".tmp"
    total_weight = sum(s["weight"] for s in SOURCES)
    n_train = n_val = 0
    with open(train_tmp, "wb") as f_train, open(val_tmp, "wb") as f_val:
        for src in SOURCES:
            target = int(TOTAL_TOKENS * src["weight"] / total_weight)
            print(f"[{src['name']}] streaming {src['path']} "
                  f"({src['config'] or 'default'}), target {target:,} train tokens", flush=True)
            stream = load_dataset(src["path"], name=src["config"], data_files=src["data_files"],
                                  split="train", streaming=True)
            stream = stream.shuffle(seed=42, buffer_size=10_000)

            col, got, doc_i, texts = src["text_column"], 0, 0, []
            pbar = tqdm(total=target, desc=src["name"], unit="tok", unit_scale=True)
            for ex in stream:
                text = ex.get(col)
                if not isinstance(text, str) or not text.strip():
                    continue
                texts.append(text[:MAX_CHARS])
                if len(texts) == 256:
                    n_tr, n_va = encode_and_write(texts, doc_i, f_train, f_val)
                    doc_i += len(texts)
                    texts = []
                    got += n_tr; n_train += n_tr; n_val += n_va
                    pbar.update(n_tr)
                    if got >= target:
                        break
            if texts and got < target:  # leftover documents at the end of the stream
                n_tr, n_va = encode_and_write(texts, doc_i, f_train, f_val)
                got += n_tr; n_train += n_tr; n_val += n_va
                pbar.update(n_tr)
            pbar.close()
            if got < target:
                print(f"  WARNING: {src['name']} ended early with {got:,} of {target:,} tokens", flush=True)

    if n_val < 10_000:
        raise RuntimeError(f"Only {n_val:,} validation tokens were produced; increase TOTAL_TOKENS.")
    os.replace(train_tmp, TRAIN_BIN)
    os.replace(val_tmp, VAL_BIN)
    with open(META_PATH, "w") as f:
        json.dump({"tokenizer": TOKENIZER_NAME, "dtype": np.dtype(TOKEN_DTYPE).name,
                   "n_vocab": enc.n_vocab, "eot_token": enc.eot_token,
                   "train_tokens": n_train, "val_tokens": n_val}, f, indent=2)
    print(f"\n{TRAIN_BIN}: {n_train:,} tokens\n{VAL_BIN}: {n_val:,} tokens", flush=True)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN) and not FORCE:
        meta = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}
        if meta.get("tokenizer") != TOKENIZER_NAME or meta.get("dtype") != np.dtype(TOKEN_DTYPE).name:
            sys.exit(f"{TRAIN_BIN} exists but was not built with {TOKENIZER_NAME}/"
                     f"{np.dtype(TOKEN_DTYPE).name} (meta: {meta or 'missing'}). "
                     f"Use a fresh OUTPUT_DIR, or rerun with FORCE=1 to overwrite.")
        n = meta.get("train_tokens")
        print(f"Token files already exist in {DATA_DIR} "
              f"({f'{n:,}' if n else 'unknown number of'} train tokens). Nothing to do; "
              f"set FORCE=1 to rebuild.", flush=True)
        return

    print(f"Streaming and tokenizing {TOTAL_TOKENS:,} train tokens from "
          f"{[s['name'] for s in SOURCES]} into {DATA_DIR}", flush=True)
    build_bins()
    print(f"\nDone. To train on another machine, copy the whole folder {DATA_DIR}\n"
          f"(train.bin, validation.bin, meta.json) there, then run: python -u Mini-Intern.py", flush=True)


if __name__ == "__main__":
    main()
