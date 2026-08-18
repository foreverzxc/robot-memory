# Codebase Pointer Index

This ARA points at the live codebase under `E:\deepseekharness\patch_policy_ttt`. All paths below were verified to exist.

## TTT core
- **File(s) in repo**: `button_task/ttt_layer.py`
- **Nature**: library module
- **What it does / contains**: per-sample `vmap(grad(...))` fast-weight inner update, TTTLayer gated residual apply, TTTSequence chunked forward with TBPTT, valid masks, damp consolidation, softplus inner lr, cached store-grad transform.
- **How to use / run**: imported by `train_button_ttt.py`, `scripts/eval_button_checkpoint.py`, `scripts/probe_button_features.py`, and pipeline tests.
- **Claims supported**: C01, C02, C04

## GPT integration
- **File(s) in repo**: `vendor/patch_policy/models/vq_behavior_transformer/gpt.py`
- **Nature**: library module
- **What it does / contains**: GPT with block-diagonal `per_timestep_attn`, repeated per-timestep positions, per-layer TTT ModuleList, `ttt_modules()`, `init_ttt_fast_weights()`, `set_ttt_tbptt_step_size()`.
- **How to use / run**: used through `BehaviorTransformer`; no standalone CLI.
- **Claims supported**: C01

## VQ-BeT policy wrapper
- **File(s) in repo**: `vendor/patch_policy/models/vq_behavior_transformer/bet.py`
- **Nature**: library module
- **What it does / contains**: `BehaviorTransformer` forward with `loss_mask`, masked FocalLoss/L1 action losses, per-layer TTT pass-through.
- **How to use / run**: instantiated by `train_button_ttt.py::build_model`.
- **Claims supported**: C01, C03

## Training entry point
- **File(s) in repo**: `train_button_ttt.py`
- **Nature**: script
- **What it does / contains**: CLI, dataset/embedding pipeline, carry-windows chunked training, remaining-length curriculum, TBPTT setup, frozen mode, eval_in_sim with remaining-password updates.
- **How to use / run**: `python train_button_ttt.py --h5 ... --out ...`
- **Claims supported**: C03, C04, C05

## Sequence dataset
- **File(s) in repo**: `button_task/train_dataset.py`, `button_task/button_dataset.py`
- **Nature**: library modules
- **What it does / contains**: shard iteration, sequence windows with valid masks, allowed-episode filtering, HDF5 reading.
- **How to use / run**: imported by the training entry point.
- **Claims supported**: C05

## Label generation
- **File(s) in repo**: `scripts/label_button_demos.py`, `scripts/label_missing_best_effort.py`
- **Nature**: scripts
- **What it does / contains**: replay-based per-frame press-count labeling; strict skip rules; best-effort final-press correction.
- **How to use / run**: `python scripts/label_button_demos.py --h5 ... --out ...`
- **Claims supported**: C05

## Checkpoint evaluation
- **File(s) in repo**: `scripts/eval_button_checkpoint.py`
- **Nature**: script
- **What it does / contains**: model reconstruction (single or per-layer TTT), simulator rollout, remaining-password conditioning.
- **How to use / run**: `python scripts/eval_button_checkpoint.py --ckpt ... --args ... --passwords ...`
- **Claims supported**: C01, C03

## Rollout visualization
- **File(s) in repo**: `scripts/visualize_button_rollouts.py`
- **Nature**: script
- **What it does / contains**: proportional train/holdout rollout sampling, annotated GIF export (default fps=20).
- **How to use / run**: `python scripts/visualize_button_rollouts.py --ckpt ... --args ... --out ...`
- **Claims supported**: C01, C03

## Tests
- **File(s) in repo**: `scripts/test_button_ttt_pipeline.py`, `scripts/test_button_train_pipeline.py`
- **Nature**: test suites
- **What it does / contains**: 15 TTT pipeline checks and 4 baseline pipeline checks covering consistency, per-sample updates, frozen mode, TBPTT, damp, valid masks, per-step attention, per-layer TTT.
- **How to use / run**: `python scripts/test_button_ttt_pipeline.py`
- **Claims supported**: C01, C02
