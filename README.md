# When Is Unverified Feedback Worth Verifying? — code and data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24487587.svg)](https://doi.org/10.5281/zenodo.24487587)

Code, real-feedback annotations and per-seed results for the paper

> **When Is Unverified Feedback Worth Verifying? A Budget-Aware Study of Learning from Noisy Corrections for Small Tool-Using Agents**
> Phong Quyet Dao, Than Le.

An agent receives corrective feedback for its tool choices, but only a fraction ρ of the corrections can be verified. We measure the value of the unverified remainder by an **exchange rate** ρ_eq/ρ — how many verified samples verified-only training would need to match a learner that also uses the unverified pool — and compare reliability estimators (source prior, Ctx-EB, CA-EB) with crowdsourcing and noisy-label baselines (Dawid–Skene, Raykar, MACE, RGPO-style, small-loss, co-teaching-style) on a synthetic benchmark and on real corrections from seven open LLMs (BFCL and MetaTool).

## Layout

| Path | Content |
|---|---|
| `nrf/data.py` | Synthetic tool-selection benchmark (12 domains, 6 sources, uniform / source / context noise, organisational conventions) |
| `nrf/reliability.py`, `nrf/baselines.py` | All reliability estimators and baselines. Methods only see `observable_view()`; true labels of unverified items are never accessible |
| `tests/test_nrf.py` | Unit tests incl. the label-leakage permutation test for every method |
| `pilot_cpu.py`, `breakeven_cpu.py`, `bfcl_breakeven_cpu.py` | CPU learners (TF-IDF / frozen bge embeddings + logistic regression), exchange-rate sweeps |
| `q1_e4_cpu.py` | Noisy-label baseline benchmark (synthetic and BFCL, 20 seeds) |
| `q1_e3_cpu.py` | Online multi-round loop (4 rounds) |
| `q1_a1_cpu.py` | Verification-selection ablation (random / most suspicious / stratified) |
| `scripts/q1_metatool_cpu.sh`, `scripts/prep_metatool.py` | Second real-feedback task (MetaTool) |
| `annotate_bfcl.py` | Collects corrective feedback from open LLMs (BFCL and MetaTool items) |
| `real_feedback.py`, `kaggle_qlora.py` | Qwen3 QLoRA learners on real / synthetic feedback (GPU) |
| `scripts/q1_stats.py`, `scripts/analyze_breakeven.py` | Paired bootstrap CIs, exact sign-flip tests, Holm correction, exchange-rate inversion |
| `scripts/theory_check.py` | Monte-Carlo check of the exact binary mean-estimation exchange rate |
| `data/` | BFCL items and LLM annotations (`data/bfcl_annot/`); MetaTool items (MetaTool annotations: `results/q1_e1/annotations/`) |
| `results/` | Per-seed raw results and summaries for every experiment in the paper |

## Quick start (CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
OMP_NUM_THREADS=1 pytest -q                                  # incl. leakage tests
python q1_e4_cpu.py --setup synth                             # baseline benchmark, synthetic
python q1_e4_cpu.py --setup bfcl                              # baseline benchmark, real BFCL feedback
python q1_e3_cpu.py                                           # online loop
python q1_a1_cpu.py                                           # verification-selection ablation
python scripts/q1_stats.py                                    # CIs, sign-flip tests, Holm, exchange rates
python scripts/theory_check.py                                # theory Monte-Carlo check
```
Each script prints its options with `--help`. GPU experiments (`annotate_bfcl.py`, `real_feedback.py`, `kaggle_qlora.py`, `scripts/q1_local_gpu*.sh`) need a CUDA GPU (we used NVIDIA T4 and RTX 3050) and download the open models from Hugging Face.

## Results → paper

| Paper item | Result files |
|---|---|
| Context-dependent reliability (BFCL heat map, LR test) | `results/bfcl_feedback_analysis.md` |
| Exchange rate vs ρ | `results/breakeven.md`, `results/cpu_breakeven*/`, `results/q1_e5/` |
| Baseline benchmark (E4) and statistics (E5) | `results/q1_e4/`, `results/q1_e5/` |
| MetaTool replication (E1) | `results/q1_e1/` |
| Online loop (E3) | `results/q1_e3/` |
| Verification-selection ablation (A1) | `results/q1_a1/` |
| QLoRA learners (GPU) | `results/kaggle/`, `results/q1_gpu/`, `results/gpu_*.md` |
| Theory check | `results/theory_t1/` |

## Data licences

- BFCL (Berkeley Function Calling Leaderboard): Apache-2.0, © the Gorilla authors. Files under `data/bfcl/` are redistributed unchanged.
- MetaTool: MIT, © the MetaTool authors.
- LLM annotations in `data/` were generated with open models (Qwen, SmolLM2, Granite, Phi-3.5) under their respective licences.

## Licence and citation

Code: MIT (see `LICENSE`). If you use this repository, please cite the paper (see `CITATION.cff`).
