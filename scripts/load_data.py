# -*- coding: utf-8 -*-
"""
Small Language Model from Scratch - adapted for HuggingFaceTB/cosmopedia

Adapted from a TinyStories/Alpaca-based training script.

# Introduction

We build a Small Language Model (SLM) from scratch, keeping the parameter
count at roughly 50-60 million.

We train on Cosmopedia, a large synthetic corpus of textbook-style content,
stories, WikiHow articles, etc. There is no instruction/response structure:
each example's `text` column is plain prose, so we tokenize it directly and
append an end-of-text token after every document so the model learns where
one document ends and the next begins.

Install dependencies once in your shell (not inside this file):
    pip install tiktoken datasets numpy tqdm matplotlib torch

On a cluster node without internet, pre-download the data from a login node
(huggingface-cli download HuggingFaceTB/cosmopedia --repo-type dataset
 --include "data/stories/*" ...) and run with HF_DATASETS_OFFLINE=1.

## Step 1: Import the Dataset
"""
import joblib 
import os
from datasets import load_dataset, DatasetDict, concatenate_datasets

# Cosmopedia subsets (configs) to use. Options: stories, wikihow, openstax,
# khanacademy, stanford, auto_math_text, web_samples_v1, web_samples_v2.
CONFIGS = ["stories", "wikihow"]

# Cap rows per config so tokenization stays manageable. None = use everything.
MAX_ROWS_PER_CONFIG = 500_000

# Everything produced by this script (token files, checkpoint, plot) lives here
# so it can never be confused with files from an earlier Alpaca/TinyStories run.
print ("Creating the dataset...") 

DATA_DIR = "data_cosmopedia"
os.makedirs(DATA_DIR, exist_ok=True)
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "validation.bin")
BEST_MODEL_PATH = os.path.join(DATA_DIR, "best_model_params.pt")
LOSS_PLOT_PATH = os.path.join(DATA_DIR, "loss_curve.png")

# Use the CPUs Slurm gave us, if any
NUM_PROC = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

# Only build the dataset if we still need to tokenize it
if not (os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN)):
    parts = []
    for cfg in CONFIGS:
        d = load_dataset("HuggingFaceTB/cosmopedia", cfg, split="train")
        if MAX_ROWS_PER_CONFIG is not None and len(d) > MAX_ROWS_PER_CONFIG:
            d = d.shuffle(seed=42).select(range(MAX_ROWS_PER_CONFIG))
        d = d.select_columns(["text"])  # keep only text so configs concatenate cleanly
        parts.append(d)

    raw_train = concatenate_datasets(parts).shuffle(seed=42)

    # Cosmopedia only has a "train" split, so carve out a small validation set.
    split = raw_train.train_test_split(test_size=0.01, seed=42)
    ds = DatasetDict({'train': split['train'], 'validation': split['test']})
joblib.dump(ds, "ds.pkl")
print ("Finished creating the datasets") 


