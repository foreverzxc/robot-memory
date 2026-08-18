# Training Configuration

Source of record: `runs/b4_ttt_perlayer_patch112_ep30/args.json` (same effective model config as all b4 runs).

## Dataset / sequence
- **Value**: `random_pw6_lang_1000/demos.h5`, split 48 train / 16 holdout via `button_task/password_split.json`
- **Rationale**: maximizes train password coverage while keeping held-out passwords for OOD evaluation
- **Search range**: not searched
- **Sensitivity**: high (split is fixed)
- **Source**: args.json

## encoder_mode / image_size
- **Value**: `patch` / `112`
- **Rationale**: 128 patch tokens per timestep give the TTT inner loop a larger K→V mini-batch than 2 CLS tokens
- **Search range**: `cls@224` previously used; not swept here
- **Sensitivity**: high
- **Source**: args.json

## t_window / action_window / carry_windows
- **Value**: 16 / 12 / true
- **Rationale**: chunked carry-windows training with TBPTT boundary detach; 12-step action chunks match the Patch Policy head
- **Search range**: not searched
- **Sensitivity**: medium
- **Source**: args.json

## TTT modules
- **Value**: 8 per-layer TTTSequence modules (one after each GPT block), fast_hidden=256, num_layers=1 per module
- **Rationale**: matches RoboTTT's per-attention-layer placement
- **Search range**: not searched
- **Sensitivity**: high
- **Source**: `train_button_ttt.py::build_model`

## ttt_base_lr
- **Value**: 0.01 (softplus inner lr initialized at 0.01)
- **Rationale**: small initial inner step; can grow/shrink via outer gradients
- **Search range**: 0.1 used in earlier runs, replaced after review
- **Sensitivity**: medium
- **Source**: args.json

## tbptt
- **Value**: 16 (auto-set to t_window for carry mode)
- **Rationale**: detach fast-weight graphs at every chunk boundary
- **Search range**: not searched
- **Sensitivity**: high (required for multi-chunk backward)
- **Source**: `train_button_ttt.py`

## curriculum_epochs
- **Value**: 5 for the initial run; 0 for continuations
- **Rationale**: remaining password length ramps 1→2→4→5→6 over 5 epochs, then full sequences; labels are used ONLY to schedule loss, never to change the password conditioning
- **Search range**: not searched
- **Sensitivity**: medium
- **Source**: `runs/b4_ttt_perlayer_patch112_smoke/args.json`

## Password conditioning (fixed protocol)
- **Value**: full episode password, constant for every window and every eval timestep
- **Rationale**: dynamic remaining-suffix conditioning leaks ground-truth progress and is forbidden; the model must infer progress from observations/history
- **Search range**: n/a
- **Sensitivity**: high
- **Source**: `train_button_ttt.py`, `train_dataset.py`

## Optimization
- **Value**: lr 1e-4, weight_decay 2e-4, AdamW (0.9, 0.999), batch_size 4, bf16 autocast
- **Rationale**: default Patch Policy fine-tuning values; batch 4 for GPU memory with patch@112
- **Search range**: not searched
- **Sensitivity**: medium
- **Source**: args.json

## Evaluation
- **Value**: eval_freq at checkpoint epochs; max_env_steps 600; one rollout per password; 4 train passwords sampled
- **Rationale**: budget-constrained smoke protocol
- **Search range**: not searched
- **Sensitivity**: medium
- **Source**: args.json
