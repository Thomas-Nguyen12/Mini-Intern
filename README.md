# Mini-Intern

A small GPT-style language model, built and pretrained from scratch in PyTorch. The goal is a compact model that can later be fine-tuned to act as a **tool router** for a news project: read an instruction, pick the right tool, and emit a structured call (for example JSON) that ordinary code validates and runs.

> **Status:** the scripts compile, but they have not been run end to end yet. All time, cost and accuracy figures below are rough estimates, not measurements. Do a short test run first (see Quick start).

## What's in this folder

| File | Purpose |
|---|---|
| `load_data.py` | Streams the datasets from the Hugging Face Hub, tokenizes them, and writes `train.bin`, `validation.bin` and `meta.json`. Run once, on any machine (CPU only). |
| `Mini-Intern.py` | Builds the model and trains it on those files. Standalone: needs no internet and no `datasets` library. Resumes automatically after interruptions. |
| `merge_folders.sh` | Unrelated helper: merges several downloaded folders (e.g. a Google Drive download split into parts) into one, skipping duplicate files. |

## Pipeline

```
load_data.py  ->  data_code/{train.bin, validation.bin, meta.json}  ->  Mini-Intern.py  ->  runs/<size>/best_model_params.pt
(stream + tokenize)         (copy to the training PC)                     (train)              (use for fine-tuning)
```

Note that the contents of **data_code/** are too large and will be stored locally. 

### Data

Streamed (never fully downloaded) and tokenized with tiktoken `cl100k_base`, stored as `uint32`:

| Source | Share | Why |
|---|---|---|
| `codeparrot/codeparrot-clean` (Python) | 50% | Code and indentation patterns |
| `HuggingFaceTB/smollm-corpus`, `fineweb-edu-dedup` | 35% | General English and knowledge |
| `HuggingFaceTB/smollm-corpus`, `cosmopedia-v2` | 15% | Synthetic textbooks and how-tos |

The mix is the `SOURCES` list at the top of `load_data.py`; you can add your own local files there. Every 100th document is held out for validation (about 1%). Check each dataset's license before using the model beyond personal experiments.

- `train.bin` / `validation.bin`: one long stream of token IDs, with an end-of-text token after each document.
- `meta.json`: tokenizer name, dtype and token counts, so the trainer reads the files correctly and can detect an incomplete copy.

### Model

Decoder-only transformer (nanoGPT style): learned positional embeddings, pre-norm blocks, tied input/output embeddings, flash attention when available, 1024-token context, 100,352-token padded vocabulary.

| `MODEL_SIZE` | Layers / heads / width | Parameters |
|---|---|---|
| `small` | 8 / 8 / 512 | ~77M |
| `base` (default) | 12 / 12 / 768 | ~163M |

Training uses AdamW, linear warmup then cosine decay, bf16 on supported GPUs, gradient accumulation to a fixed 524,288 tokens per step, and gradient clipping. 4000 steps is about 2.1B tokens.

## Quick start

```bash
pip install numpy tqdm matplotlib torch datasets tiktoken

# 1. Quick pipeline test (a few minutes)
OUTPUT_DIR=test_data TOTAL_TOKENS=2e7 python -u load_data.py
DATA_DIR=test_data RUN_DIR=runs/test MAX_STEPS=20 MODEL_SIZE=small python -u Mini-Intern.py

# 2. Real run
python -u load_data.py                 # builds data_code/ (an hour or more, needs internet)
python -u Mini-Intern.py               # trains; copy data_code/ to the training PC first if different
```

`datasets` is only needed by `load_data.py`; `tiktoken` is only needed for the sample text at the end of training.

### Environment variables

| Variable | Used by | Default | Meaning |
|---|---|---|---|
| `OUTPUT_DIR` | load_data | `data_code` | Where the token files are written |
| `TOTAL_TOKENS` | load_data | `2.1e9` | Training tokens to stream and tokenize |
| `FORCE` | load_data | off | `1` rebuilds existing token files |
| `DATA_DIR` | Mini-Intern | `data_code` | Folder with the token files |
| `RUN_DIR` | Mini-Intern | `runs/<MODEL_SIZE>` | Checkpoints and logs |
| `MODEL_SIZE` | Mini-Intern | `base` | `small` or `base` |
| `MAX_STEPS` | Mini-Intern | `4000` | Total optimizer steps (also sets the LR schedule) |
| `BATCH_SIZE` | Mini-Intern | by GPU memory | Micro-batch size (4 on a 12 GB card); tokens per step stay fixed |
| `RESUME` | Mini-Intern | on | `0` ignores `ckpt.pt` and starts over |
| `COMPILE` | Mini-Intern | off | `1` enables `torch.compile` (CUDA only) |

## Outputs

`runs/<size>/` contains `ckpt.pt` (full checkpoint for resuming), `best_model_params.pt` (lowest validation loss; use this for fine-tuning), `loss_log.csv` and `loss_curve.png`.

**Interruptions:** a checkpoint is saved every 250 steps. Press Ctrl-C once to stop cleanly (the step finishes, a checkpoint is saved); run the same command again to resume. Keep `MAX_STEPS` and `MODEL_SIZE` unchanged when resuming.

## Hardware notes (estimates)

| Run | RTX 5070 (12 GB) | A100 |
|---|---|---|
| `base`, 2.1B tokens, 4000 steps | ~15-30 h | ~6-12 h |
| `small`, 1B tokens, 1900 steps | ~3-6 h | ~2-4 h |
| `small`, 0.5B tokens, 950 steps | ~1.5-3 h | ~1-2 h |

- Rough rule used: training costs about 6 x parameters x tokens FLOPs.
- A CPU-only machine would take weeks. The script warns if it can't see your GPU.
- **RTX 50-series (e.g. 5070):** install a PyTorch build for CUDA 12.8 or newer, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu128`.
- On a free-tier or limited cloud GPU, use `BATCH_SIZE=4` and the `small` model, and keep `RUN_DIR` on persistent storage.

## What to expect from the model

After pretraining it only continues text and code; it does **not** follow instructions until fine-tuned. After fine-tuning on a few thousand varied tool-call examples, my rough guess is:

- **Tool routing, 3-5 tools:** ~85-95% on phrasing similar to the training data, ~55-80% on new phrasing. Larger tool sets and vague requests do worse.
- **Short code:** simple snippets with frequent bugs.
- **Not suitable for:** summarizing or analyzing news articles (it has no news knowledge, a 1024-token context, and invents details), multi-step reasoning, or running arbitrary generated code unsandboxed.

Recommended design: use the model only to turn an instruction into a tool call, validate the output against a schema (retry on failure), keep tools in ordinary code, restrict decoding to valid tool names, and use a stronger model for any summarization.

## Next steps

1. Run the quick test, then the real pretraining run.
2. Write the tool list and generate varied synthetic training examples (20+ phrasings per request, plus "no tool" cases).
3. Write a fine-tuning script that loads `best_model_params.pt` and computes loss only on the response tokens (the model's `forward` already ignores targets set to `-1`). Not written yet.
4. Wrap `generate` with stop sequences and a validator, and plug it into your pipeline.
