# SYMBA 2026: Using Next-Gen Transformers to Seed Generative Models for Symbolic Regression
## About Me
I am **Isha Sarvani Telikicherla**, a second-year undergraduate in Data Science and Engineering at MIT Manipal, working at the intersection of graph machine learning, privacy-preserving computation, and scientific AI. I am currently interning at Hanze University of Applied Sciences on cryptographic MPC systems and conducting research on privacy-preserving synthetic graph generation using VGAEs.

Symbolic regression is the problem of discovering closed-form mathematical equations directly from data — not just fitting a curve, but recovering the actual algebraic structure of a physical law. In science, an interpretable equation is a hypothesis about the underlying system, not just a predictor. Task 2.6 asks for a next-generation transformer architecture that seeds a generative search process capable of recovering precise, human-readable formulas from raw numerical observations.

The solution is a *five-stage hybrid pipeline*. A SineKAN encoder maps a 200-point cloud of (input, output) pairs into a compact embedding using sinusoidal basis functions that give it a natural affinity for the periodic functions common in physics. An autoregressive transformer decodes that embedding into candidate equation skeletons in postfix notation, with grammar-constrained beam search guaranteeing structural validity. A bidirectional MLM refiner corrects low-confidence tokens, a two-stage BFGS optimiser fits numerical constants, and finally these candidates seed 40% of a genetic programming population that evolves over 80 generations into precise equations. The pipeline achieves 89.7% mean R² on 15 held-out Feynman equations, solves 100% to R² ≥ 0.5, and perfectly recovers 5 to R² = 1.0.

---

This repository contains my complete implementation for the ML4SCI SYMBA GSoC 2026 evaluation tasks. It directly addresses:

- **Common Task 1.1** — Dataset preprocessing and tokenisation (AI Feynman benchmark)
- **Specific Task 2.6** — Next-gen transformer model seeding a generative technique (PIGP) for symbolic regression, building on ML4SCI GSoC 2025 solutions

---

## Repository Structure

```
SYMBA_Next-GenTransformer_Isha_Sarvani_Telikicherla/
│
├── notebooks/
│   └── Task_1_1_PreProcessing.ipynb      # Executed Task 1.1 notebook
│
├── scripts/
│   ├── train.py                           # Two-phase training
│   └── evaluate.py                        # Five-stage pipeline evaluation
│
├── src/
│   ├── config.py                          # All hyperparameters and paths
│   ├── models/
│   │   ├── sinekan.py                     # Sine-based KAN layer
│   │   ├── encoder.py                     # SineKAN point-cloud encoder
│   │   ├── decoder.py                     # Autoregressive transformer decoder
│   │   └── birefiner.py                   # Bidirectional MLM refiner
│   ├── data/
│   │   ├── dataset.py                     # RealDataset, cloud loader, collate
│   │   ├── synthetic_gen.py               # 10k synthetic equation generator
│   │   └── vocab.py                       # Vocab utilities
│   └── pipeline/
│       ├── beam_search.py                 # Grammar-constrained beam search
│       ├── bfgs.py                        # Two-stage constant optimiser
│       ├── gp_system.py                   # GP core (Samyak Jha, GSoC 2025)
│       ├── gp_wrapper.py                  # PIGP seeding interface
│       └── postfix_eval.py                # Postfix expression evaluator
│
├── data/
│   ├── FeynmanEquations.csv
│   ├── task1_1_tokenized_postfix.json
│   ├── vocab/vocab.json
│   ├── clouds/isha_data_clouds.json
│   └── results/pipeline_results.json
│
└── weights/
    ├── encoder.pth
    └── decoder.pth
```

---

## Quick Links for Evaluators

Here is exactly where the relevant code, data, and outputs for each task are located:

### Task 1.1 — Preprocessing & Tokenisation
| Item | Location |
|------|----------|
| Executed preprocessing notebook | `notebooks/Task_1_1_PreProcessing.ipynb` |
| Postfix token sequences + vocabulary + 67/18/15 split | `data/task1_1_tokenized_postfix.json` |
| 200-point data clouds per equation | `data/clouds/isha_data_clouds.json` |
| Unified 115-token vocabulary | `data/vocab/vocab.json` |

### Task 2.6 — Next-Gen Transformer Pipeline
| Item | Location |
|------|----------|
| SineKAN encoder | `src/models/encoder.py` |
| Autoregressive decoder | `src/models/decoder.py` |
| Grammar-constrained beam search | `src/pipeline/beam_search.py` |
| BiRefiner (bidirectional MLM) | `src/models/birefiner.py` |
| Two-stage BFGS constant fitting | `src/pipeline/bfgs.py` |
| PIGP seeding wrapper | `src/pipeline/gp_wrapper.py` |
| GP core (GSoC 2025 — Samyak Jha) | `src/pipeline/gp_system.py` |
| Full training script | `scripts/train.py` |
| Full evaluation script | `scripts/evaluate.py` |
| Pipeline results | `data/results/pipeline_results.json` |

> **Note:** The GP core (`gp_system.py`) is reused from Samyak Jha's GSoC 2025 implementation. My contribution is the PIGP seeding strategy in `gp_wrapper.py` and the full five-stage pipeline integrating it with the transformer outputs.

---

## How to Run

### Task 1.1 — Preprocessing (Notebook)

An executed version with all cell outputs is already saved at `notebooks/Task_1_1_PreProcessing.ipynb`. To re-run from scratch:

```bash
jupyter nbconvert --to notebook --execute notebooks/Task_1_1_PreProcessing.ipynb \
  --output Task_1_1_PreProcessing.ipynb
```

**Produces:**
- `data/task1_1_tokenized_postfix.json` — postfix sequences, vocabulary, 67/18/15 split
- `data/clouds/isha_data_clouds.json` — 200-point (x, y) clouds per equation

### Task 2.6 — Training (optional — weights included)

Pre-trained weights are in `weights/`. To retrain from scratch:

```bash
python scripts/train.py
```

Runs Phase 1 (60 epochs, 10k synthetic equations) then Phase 2 (120 epochs, 67 Feynman equations). Saves `weights/encoder.pth` and `weights/decoder.pth`.

### Task 2.6 — Evaluation

```bash
python scripts/evaluate.py
```

Runs all five pipeline stages on the 15 held-out test equations. Prints per-equation results and saves to `data/results/pipeline_results.json`.

---

## Key Architectural Upgrades Over GSoC 2025 Baseline

The GSoC 2025 baseline (Krish Malik) used prefix tokenisation, a T-Net encoder, greedy decoding, and teacher-forced evaluation — producing inflated metrics and poor generalisation. Here is what changed:

| Component | GSoC 2025 (Krish) | This Work |
|-----------|-------------------|-----------|
| Tokenisation | Prefix (function-call) strings | **Postfix (RPN)** via recursive tree walker |
| Encoder | T-Net | **SineKAN** — sinusoidal basis for periodic inductive bias |
| Decoding | Greedy, teacher-forced only | **Grammar-constrained beam search** (w=16) |
| Training data | 100 Feynman equations | **10,000 synthetic + 67 Feynman** (two-phase) |
| Refinement | None | **BiRefiner** — bidirectional MLM, 3 iterative passes |
| Constant fitting | Not implemented | **Two-stage BFGS** (Nelder-Mead → L-BFGS-B) |
| Generative seeding | None | **PIGP** — 40% population seeded from beam candidates |
| Evaluation | Teacher-forced accuracy only | **True autoregressive inference + R² on held-out data** |

---

## Results

Evaluated on 15 held-out Feynman equations never seen during any phase of training.

| Metric | Value |
|--------|-------|
| **Mean R²** | **0.897** |
| R² ≥ 0.99 (perfect symbolic recovery) | 5 / 15 (33%) |
| R² ≥ 0.50 (solved) | 15 / 15 (100%) |
| Lowest R² | 0.512 |
| Highest R² | 1.000 |

**PIGP vs. Vanilla GP ablation** (identical GP parameters, same 15 equations):

| | PIGP | Vanilla GP |
|--|------|-----------|
| Mean R² | 0.897 | 0.888 |
| Wins / Losses / Ties | 7 / 3 / 5 | 3 / 7 / 5 |
| Max single-equation gain | +0.082 | +0.031 |

> Results vary slightly across runs due to stochastic GP. Encoder and decoder outputs are deterministic given fixed weights.

---

## Requirements

```bash
pip install torch numpy scipy sympy pandas matplotlib scikit-learn jupyter nbconvert
```

---

## <3