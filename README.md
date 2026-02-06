# BabyLM_GPT_2

This repository contains custom data preparation and GPT-style training code for BabyLM-scale experiments, plus a vendored copy of the BabyLM 2025 evaluation pipeline.

## Files In This Repo (Outside `evaluation-pipeline-2025`)

### Tracked files

- `.gitignore`
  - Currently ignores only `hf_token.txt`.
  - Practical implication: most local checkpoints, notebooks, and outputs are not ignored by default.

- `README.md`
  - Project documentation and file map (this file).

- `cleaning_and_tokenization.ipynb`
  - End-to-end preprocessing notebook:
  - Cleans BabyLM train/dev corpora with corpus-specific functions from `mrclean.py`.
  - Trains a Byte-Level BPE tokenizer (including a GPT-2-compatible 50,257 vocab setup).
  - Tokenizes text and builds chunked pickle datasets used by training scripts.
  - Includes experiments for additional datasets/chunking workflows.

- `mrclean.py`
  - Regex-based text cleanup utilities, split by corpus/source.
  - Handles issues like spacing normalization, subtitle artifacts, simple wiki formatting, and corpus-specific heuristics.
  - Used by `cleaning_and_tokenization.ipynb`.

- `training_functions.py`
  - Utility helpers for minibatch construction from tokenized tensors.
  - Supports standard next-token prediction batches and a reverse/previous-token mode.
  - Includes helper to combine chunked tokenized file outputs.

- `ewok_eval.py`
  - Local EWoK evaluation helper.
  - Computes per-token conditional log-likelihood comparisons over EWoK pairs and reports per-domain and average accuracy.
  - Called by training code for periodic intrinsic evaluation.

- `train_gpt.py`
  - Main training script for GPT-style language modeling in this repo.
  - Loads pre-chunked train/validation pickles, builds dataloaders, trains GPT-2-style model, runs periodic validation, and logs/plots losses.
  - Supports either GPT-2 tokenizer or a custom Morfessor-based tokenizer path.
  - Runs EWoK evaluation after each epoch and saves epoch metrics.
  - Saves checkpoints and pushes model/tokenizer to Hugging Face.

### Other top-level local files/folders (currently untracked)

These are present in the working directory but not committed to git right now:

- Experiment/training scripts:
  - `finetune_gpt.py`: fine-tuning workflow starting from an existing model checkpoint.
  - `moonshot.py`: another training/evaluation variant with similar training loop structure.
- Analysis notebooks:
  - `construct-bert.ipynb`, `gutenberg_analysis.ipynb`, `push_to_hub.ipynb`, `testmodelinferenceability.ipynb`, `trial.ipynb`.
- Data/output directories:
  - `data/`, `models/`, `plots/`, `eval_logs/`.
  - Many run/checkpoint folders following patterns like `babygpt-*` and `babyllama-*`.
- Local metric CSVs:
  - `ratio_accuracy.csv`, `uid_accuracy.csv`.

## What The Evaluation Pipeline Does (Broadly)

The `evaluation-pipeline-2025/` folder is the BabyLM 2025 evaluation backend. At a high level it:

- Evaluates models in two modes:
  - `fast` evaluation for quick checkpoint testing.
  - `full` evaluation for final model reporting.
- Supports multiple evaluation families:
  - Sentence-level zero-shot scoring (`evaluation_pipeline/sentence_zero_shot`).
  - Fine-tuning/classification evaluation on GLUE/SuperGLUE-like tasks (`evaluation_pipeline/finetune`).
  - Reading-related evaluations (`evaluation_pipeline/reading`).
  - EWoK data handling/utilities (`evaluation_pipeline/ewok`).
- Supports different LM backends:
  - `causal`, `mlm`, `mntp`, and encoder-decoder variants.
- Writes structured outputs under `results/<model>/<revision>/...` with per-task reports/predictions.

In short, your local scripts (`train_gpt.py`, `ewok_eval.py`, notebooks) are for preparing/training custom models, while `evaluation-pipeline-2025/` is the standardized benchmark harness for comparing those models on BabyLM tasks.
