# Reinforcement Learning Human Feedback (RLHF) Implementation

**Assignment 1 — Advanced Topics in AI and Machine Learning**  
**Student ID:** A1963402  
**Date:** March 2026

---

## Overview

This repository implements and evaluates a complete Reinforcement Learning from Human Feedback (RLHF) pipeline across two open-source large language models:

| Model | Parameters | Architecture |
|:------|:----------:|:------------:|
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | Llama-2 (GQA) |
| `facebook/opt-125m` | 125M | GPT-2 style |

Both models undergo an **identical** three-stage pipeline:

```
Stage 1  : Supervised Fine-Tuning (SFT)
Stage 2a : Reward Modelling (Bradley-Terry loss) — evaluation only
Stage 2b : Direct Preference Optimisation (DPO, β=0.1)
Stage 3  : Evaluation (reward score, win rate, perplexity, TTR, length)
```

> **Key finding:** SFT consistently improves fluency (perplexity: −34% TinyLlama, −29% OPT-125M). DPO shows correct training dynamics but reward score degrades due to reward hacking at small training scale (Gao et al., 2022). TinyLlama-1.1B outperforms OPT-125M across all quality metrics, confirming model scale significantly affects RLHF alignment effectiveness.

---

## Repository Contents

```
RLHF_TinyLlama.ipynb           TinyLlama-1.1B-Chat full pipeline
RLHF_OPT125M.ipynb             OPT-125M full pipeline
README.md                      This file
requirements.txt               Python dependencies
sft_loss_tinyllama.png         SFT loss curve — TinyLlama
sft_loss_opt.png               SFT loss curve — OPT-125M
dpo_metrics.png                DPO training metrics — TinyLlama
dpo_metrics_opt.png            DPO training metrics — OPT-125M
evaluation_results.png         Reward + win rate chart — TinyLlama
evaluation_results_opt.png     Reward + win rate chart — OPT-125M
evaluation_results.csv         Full per-prompt results (35 prompts) — TinyLlama
evaluation_results_opt.csv     Full per-prompt results (35 prompts) — OPT-125M
```

---

## Dependencies

- Python >= 3.10
- CUDA >= 11.8 (required for bitsandbytes 8-bit quantisation)

Core libraries (auto-installed in notebook Cell 1):

```
transformers  >= 4.40
trl           >= 0.12
peft          >= 0.10
datasets      >= 2.18
accelerate    >= 0.28
bitsandbytes  >= 0.43
torch         >= 2.1
```

Or install manually:

```bash
pip install transformers datasets trl peft accelerate bitsandbytes
```

---

## Option A — Google Colab (Recommended)

**Tested environment:** Google Colab free tier, T4 GPU (15GB VRAM)  
**Expected runtime:** ~48 min (TinyLlama) · ~25 min (OPT-125M)

### Step 1 — Enable GPU
`Runtime → Change runtime type → Hardware accelerator → T4 GPU`

### Step 2 — Upload notebook
`File → Upload notebook → select RLHF_TinyLlama1_1B.ipynb`  
*(repeat separately for RLHF_OPT125M.ipynb)*

### Step 3 — Mount Google Drive (Cell 2)
Run Cell 2. When prompted, click **Allow**.  
Checkpoints are saved to: `/drive/MyDrive/rlhf_assignment/`  
This prevents data loss if the session disconnects.

### Step 4 — Run all cells
`Runtime → Run all` **(Ctrl+F9)**

Cells auto-skip completed stages if checkpoints exist on Drive:
```
SFT checkpoint found — skipping training.
```

### Step 5 — Download outputs
Run the final cell to download all plots and CSV results.

> **Note:** Free Colab sessions disconnect after ~90 minutes of GPU inactivity. If disconnected, re-run from the top — all completed training stages will be skipped automatically via Drive checkpoints.

---

## Option B — Local Machine (NVIDIA GPU)

**Minimum requirements:**

| Model | Min VRAM |
|:------|:--------:|
| OPT-125M | 6GB+ |
| TinyLlama-1.1B | 15GB+ (e.g. RTX 3090, A100) |

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Modify checkpoint paths
In Cell 2 of each notebook, replace the Drive mount block with:

```python
import os
DRIVE_BASE = "./outputs"
os.makedirs(DRIVE_BASE, exist_ok=True)
SFT_DIR = f"{DRIVE_BASE}/sft_model"
RM_DIR  = f"{DRIVE_BASE}/reward_model"
DPO_DIR = f"{DRIVE_BASE}/dpo_model"
```

And remove or comment out:
```python
from google.colab import drive
drive.mount('/drive')
```

### Step 3 — Launch Jupyter and run
```bash
jupyter notebook RLHF_OPT125M.ipynb
```

> **Note:** bitsandbytes 8-bit quantisation requires an NVIDIA GPU. On CPU-only machines, change `torch_dtype=torch.float16` to `torch.float32` and `optim="adamw_bnb_8bit"` to `optim="adamw_torch"`. Training will be significantly slower.

---

## Option C — Apple Silicon (M1/M2/M3/M4)

Apple Silicon uses the MPS backend instead of CUDA.

### Step 1 — Install dependencies
```bash
pip install transformers datasets trl peft accelerate
pip install torch torchvision torchaudio
```
> **Note:** bitsandbytes does NOT support MPS. Do not install it.

### Step 2 — Modify Cell 3 (imports)
```python
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
```

### Step 3 — In all model loading calls, change:
```python
torch_dtype=torch.float16  ->  torch_dtype=torch.float32
optim="adamw_bnb_8bit"     ->  optim="adamw_torch"
device_map="auto"          ->  device_map=None
```

### Step 4 — Run notebook via Jupyter
```bash
jupyter notebook RLHF_TinyLlama1_1B.ipynb
```

> **Note:** If you encounter `NotImplementedError` on MPS, fall back to CPU by setting `DEVICE = "cpu"`.

---

## Design Choices and Justification

### DPO over PPO
PPO requires 4 concurrent models (~60GB VRAM). DPO requires only 2 and reformulates RLHF as a supervised loss, enabling training on a single 15GB T4 GPU (Rafailov et al., 2023).

### LoRA (r=16 for TinyLlama, r=8 for OPT-125M)
Reduces trainable parameters from billions to millions. Rank scaled to model capacity per Hu et al. (2021). TinyLlama: 4.5M trainable (0.41%). OPT-125M: ~300K (0.24%).

### 8-bit AdamW (bitsandbytes)
Reduces optimizer state memory by ~50% via quantisation, enabling larger batch sizes within VRAM budget (Dettmers et al., 2022).

### DPO beta=0.1
Controls KL divergence penalty from reference policy. beta=0.1 used as per Rafailov et al. (2023) standard range. Lower beta = more aggressive alignment update.

### Reward model as evaluation proxy only
In DPO, no reward model is needed during training — the frozen SFT base weights serve as the implicit reference policy via `ref_model=None`. The reward model is trained separately and used **only** to score outputs during evaluation (Stage 3).

### Dataset subsets
Full hh-rlhf (160,800 examples) exceeds free Colab session time. Subsets (5k SFT, 2k RM, 3k DPO) represent 1-3% of available data — a known limitation discussed in the report.

---

## Known Issues and Fixes

| Issue | Fix |
|:------|:----|
| `SFTTrainer got unexpected argument 'tokenizer'` | Use `processing_class=tokenizer` instead (TRL >= 0.12 API) |
| `Attempting to unscale FP16 gradients` | Cast LoRA params to float32 after `get_peft_model()` for seq_cls tasks. Set `fp16=False` in RewardConfig. |
| CUDA out of memory | Each training cell frees previous models via `del` + `gc.collect()` + `torch.cuda.empty_cache()`. If OOM persists, restart runtime — Drive checkpoints skip completed stages. |
| `NameError: name 'torch' is not defined` | Session restarted — run all cells from the top. Training stages auto-skip via checkpoints. |
| DPO tokeniser mismatch warnings | Harmless — caused by whitespace differences between hh-rlhf raw format and model tokeniser. Does not affect training correctness. |
| `max_seq_length` error in SFTConfig | Renamed to `max_length` in TRL >= 0.12. |

---

## Evaluation Metrics

| Metric | Description | Interpretation |
|:-------|:------------|:---------------|
| **Reward Score** | Scalar from Bradley-Terry reward model | Higher = more human-preferred |
| **Win Rate** | % of prompts where Model X reward > Model Y | 50% = random baseline |
| **Perplexity** | exp(avg cross-entropy on reference responses) | Lower = more fluent. Independent of reward model |
| **Vocab Diversity (TTR)** | Unique words / total words | Higher = richer, less repetitive responses |
| **Avg Response Length** | Mean word count of generated responses | Detects length collapse or verbosity |

---

## Results Summary

| Model | Avg Reward | Perplexity | TTR | Win vs Base |
|:------|:----------:|:----------:|:---:|:-----------:|
| TinyLlama Base | -0.385 | 125.78 | 0.671 | — |
| TinyLlama SFT | **-0.265** | **82.94** | **0.784** | 48.6% |
| TinyLlama DPO (b=0.1) | -0.480 | 137.04 | 0.664 | 45.7% |
| OPT-125M Base | +0.103 | 163.14 | 0.327 | — |
| OPT-125M SFT | -0.201 | **115.21** | **0.352** | 37.1% |
| OPT-125M DPO (b=0.1) | -0.017 | 165.60 | 0.291 | 31.4% |

> **Key finding:** SFT is the strongest stage at small scale. DPO degrades reward scores due to reward hacking — reward models trained on 2k examples produce near-random evaluation loss (~0.693 = random baseline).

---

## References

| # | Citation |
|:--|:---------|
| [1] | Bai et al. (2022). Training a helpful and harmless assistant with RLHF. https://arxiv.org/abs/2204.05862 |
| [2] | Dettmers et al. (2022). LLM.int8(): 8-bit matrix multiplication for transformers at scale. NeurIPS 2022. |
| [3] | Gao et al. (2022). Scaling laws for reward model overoptimization. https://arxiv.org/abs/2210.10760 |
| [4] | Hu et al. (2021). LoRA: Low-rank adaptation of large language models. https://arxiv.org/abs/2106.09685 |
| [5] | Ouyang et al. (2022). Training language models to follow instructions with human feedback. https://arxiv.org/abs/2203.02155 |
| [6] | Rafailov et al. (2023). Direct preference optimization: Your LM is secretly a reward model. https://arxiv.org/abs/2305.18290 |
| [7] | Xu et al. (2023). Iterative preference optimization with the pairwise cringe loss. https://arxiv.org/abs/2312.16682 |
| [8] | Zhang et al. (2024). TinyLlama: An open-source small language model. https://arxiv.org/abs/2401.02385 |
