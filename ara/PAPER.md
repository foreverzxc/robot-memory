---
title: "Per-Layer Test-Time Training for Two-Button Password Visuomotor Policies"
authors: [Anonymous]
year: 2026
venue: "Experiment record (unpublished)"
doi: "None"
ara_version: "1.0"
domain: "Robot learning / test-time training"
keywords: [test-time training, fast weights, robot policy, VQ-BeT, visuomotor, curriculum learning, per-layer TTT]
claims_summary:
  - "Per-layer test-time training with within-timestep attention and carried fast weights provides a train/inference-consistent long-context memory mechanism for visuomotor policies."
  - "Per-sample fast-weight gradient updates make batched training numerically equivalent to single-episode inference."
  - "A remaining-password-length curriculum, ordered from short suffix to full sequence, supports stable learning of long-password tasks."
  - "Learnable softplus inner learning rates self-adjust upward during per-layer TTT training."
  - "Deterministic argmax decoding is required for reliable VQ-BeT rollouts."
  - "Full-password cls@224 per-layer TTT reaches 0.8125 held-out success on the 48/16 button task."
abstract: "This artifact records an experiment extending RoboTTT-style test-time training (TTT) to a Patch Policy VQ-BeT model on a two-button password task (64 possible length-6 passwords; 48 train / 16 holdout). TTT modules are placed after every GPT attention block, attention is restricted within each timestep, and fast weights are carried across timesteps with truncated backpropagation. Full-password cls@224 training with a progress head and deterministic argmax eval reaches 13/16 held-out success (0.8125). Optimization notes document embedding caching and batch-size tuning that reduced full-dataset epoch time from ~70 to ~44 minutes."
---

# Per-Layer Test-Time Training for Two-Button Password Visuomotor Policies

## Overview

This is a live experiment record for the `patch_policy_ttt` codebase. It documents the
adaptation of RoboTTT fast-weight memory to a Patch Policy VQ-BeT button-pressing policy,
the training/evaluation protocol, all completed training runs and evaluation rollouts, and
the code/config artifacts used to reproduce them.

## Layer Index

### Cognitive Layer (`/logic`)
| File | Description |
|------|-------------|
| [problem.md](logic/problem.md) | Observations → gaps → key insight |
| [claims.md](logic/claims.md) | 5 falsifiable claims (C01–C05) |
| [concepts.md](logic/concepts.md) | Key technical terms |
| [experiments.md](logic/experiments.md) | 5 declarative experiments (E01–E05) |
| [solution/architecture.md](logic/solution/architecture.md) | Model/training architecture |
| [solution/algorithm.md](logic/solution/algorithm.md) | TTT update equations and protocol |
| [solution/constraints.md](logic/solution/constraints.md) | Boundary conditions and limitations |
| [solution/heuristics.md](logic/solution/heuristics.md) | Practical tricks used in this work |
| [related_work.md](logic/related_work.md) | Typed dependency graph |

### Physical Layer (`/src`)
| File | Description | Claims |
|------|-------------|--------|
| [artifacts.md](src/artifacts.md) | Pointer index to the `patch_policy_ttt` codebase | C01–C05 |
| [configs/training.md](src/configs/training.md) | Training hyperparameters with rationale | C02, C03 |
| [environment.md](src/environment.md) | Software, hardware, seeds | all |

### Data
| File | Description |
|------|-------------|
| [data/dataset.md](data/dataset.md) | Dataset provenance and label protocol |

### Exploration Graph (`/trace`)
| File | Description |
|------|-------------|
| [exploration_tree.yaml](trace/exploration_tree.yaml) | Research DAG with decisions and dead ends |

### Evidence (`/evidence`)
| File | Description |
|------|-------------|
| [README.md](evidence/README.md) | Full evidence index |
| [figures/figure1_training_loss.md](evidence/figures/figure1_training_loss.md) | Training loss curve |
| [tables/table1_training_loss.md](evidence/tables/table1_training_loss.md) | Exact per-epoch loss values |
| [tables/table2_eval_results.md](evidence/tables/table2_eval_results.md) | Simulator success rates |
| [results/](evidence/results/) | Per-run record tables |
| [logs/log_pointers.md](evidence/logs/log_pointers.md) | Direct log pointers |
