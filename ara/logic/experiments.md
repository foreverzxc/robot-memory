# Experiments

## E01: Train/inference equivalence unit tests
- **Verifies**: C01, C02
- **Evidence**: `evidence/logs/log_pointers.md`; test output captured in session
- **Run**: `scripts/test_button_ttt_pipeline.py`
- **Setup**:
  - Model: small GPT (2 blocks, n_embd 16) with single and per-layer TTTSequence modules
  - Hardware: CUDA GPU (RTX 5060 Ti 16GB)
  - Dataset: synthetic random token streams
  - System: `patch_policy_ttt` with per-step attention and per-layer TTT code paths
- **Procedure**:
  1. Run all pipeline unit tests.
  2. For TTT modules, compare one T-window forward against sequential T=1 forwards with carried fast weights.
  3. For GPT, compare per-step attention T=4 against sequential T=1 calls.
  4. For batched fast weights, compare B=2 against two independent B=1 runs.
  5. Check frozen mode, TBPTT multi-chunk backward, damp effect, valid masks, and update direction.
- **Metrics**: maximum absolute differences in outputs and fast weights; pass/fail of gradient checks.
- **Expected outcome**:
  - All consistency tests pass within numerical tolerance.
  - Frozen mode and multi-chunk TBPTT runs do not raise graph errors.
- **Baselines**: none (internal consistency tests)
- **Dependencies**: none

## E02: Five-epoch curriculum smoke training
- **Verifies**: C03, C04, C05
- **Evidence**: `evidence/results/smoke5.md`, `evidence/tables/table1_training_loss.md`
- **Run**: `runs/b4_ttt_perlayer_patch112_smoke`
- **Setup**:
  - Model: Patch Policy VQ-BeT + 8 per-layer TTT modules (patch@112)
  - Hardware: CUDA GPU (RTX 5060 Ti 16GB)
  - Dataset: random_pw6_lang_1000, 48 train / 16 holdout passwords; 317 labeled episodes
  - System: carry-windows T=16, batch 4, softplus inner lr 0.01, remaining-password conditioning, curriculum cap ramping over 5 epochs
- **Procedure**:
  1. Train from scratch for 5 epochs with remaining-length curriculum.
  2. Fit the VQ codebook after epoch 0.
  3. At epoch 5, evaluate 16 holdout passwords and 4 train passwords in simulation.
  4. Generate rollout GIFs from the best checkpoint.
- **Metrics**: epoch mean training loss; simulator success rate and press count.
- **Expected outcome**:
  - Training loss decreases across epochs.
  - Holdout success improves relative to the pre-TTT baseline.
  - The TTT gate and inner lr increase from their small initialization.
- **Baselines**: earlier `b3_ttt_carry` single-layer TTT run; no-TTT VQ-BeT baselines.
- **Dependencies**: E01

## E03: Continuation to 10 total epochs
- **Verifies**: C03, C04
- **Evidence**: `evidence/results/ep10.md`, `evidence/tables/table2_eval_results.md`
- **Run**: `runs/b4_ttt_perlayer_patch112_ep10`
- **Setup**:
  - Same model and dataset as E02; initialized from `runs/b4_ttt_perlayer_patch112_smoke/snapshot.pt`
  - Hardware: CUDA GPU (RTX 5060 Ti 16GB)
  - System: 5 additional epochs, curriculum disabled (full sequences)
- **Procedure**:
  1. Load the epoch-5 snapshot (all 264 matching keys).
  2. Train 5 epochs with full-sequence task loss.
  3. Evaluate holdout (16) and train-sampled (4) passwords in simulation.
- **Metrics**: epoch mean training loss; simulator success rate and press count.
- **Expected outcome**:
  - Loss continues to decrease.
  - Holdout success increases versus the epoch-5 checkpoint.
- **Baselines**: epoch-5 checkpoint from E02.
- **Dependencies**: E02

## E04: Continuation to 20 total epochs
- **Verifies**: C03, C04
- **Evidence**: `evidence/results/ep20.md`, `evidence/tables/table2_eval_results.md`
- **Run**: `runs/b4_ttt_perlayer_patch112_ep20`
- **Setup**:
  - Initialized from `runs/b4_ttt_perlayer_patch112_ep10/snapshot.pt`
  - Hardware: CUDA GPU (RTX 5060 Ti 16GB)
  - System: 10 additional epochs, full sequences, evaluation at the final epoch
- **Procedure**:
  1. Load the epoch-10 snapshot.
  2. Train 10 epochs.
  3. Evaluate holdout and train-sampled passwords in simulation.
- **Metrics**: epoch mean training loss; simulator success rate and press count.
- **Expected outcome**:
  - Loss keeps decreasing.
  - Holdout success increases versus epoch 10.
- **Baselines**: epoch-10 checkpoint from E03.
- **Dependencies**: E03

## E05: Continuation toward 30 total epochs with convergence gate
- **Verifies**: C03, C04
- **Evidence**: `evidence/results/ep30.md` (in progress), `evidence/tables/table1_training_loss.md`
- **Run**: `runs/b4_ttt_perlayer_patch112_ep30`
- **Setup**:
  - Initialized from `runs/b4_ttt_perlayer_patch112_ep20/snapshot.pt`
  - Hardware: CUDA GPU (RTX 5060 Ti 16GB)
  - System: up to 10 additional epochs, full sequences, stop early if loss plateaus; total budget never exceeds 50 epochs
- **Procedure**:
  1. Load the epoch-20 snapshot.
  2. Train and log per-epoch loss.
  3. Continue until loss plateaus or the epoch budget is reached.
  4. Evaluate holdout and train-sampled passwords and regenerate rollout GIFs at fps=20.
- **Metrics**: epoch mean training loss; simulator success rate; loss change between consecutive epochs.
- **Expected outcome**:
  - Loss change per epoch shrinks toward a plateau.
  - Holdout success remains stable or improves.
- **Baselines**: epoch-20 checkpoint from E04.
- **Dependencies**: E04

## E06: patch@112 vs cls@224 small overfit (full-password protocol)
- **Verifies**: encoder choice for formal runs
- **Evidence**: `evidence/results/patch_vs_cls_small.md`, `runs/b7_patch_overfit_small`, `runs/b6_aux3_overfit_small2`
- **Setup**:
  - Data: `random_pw6_lang_small`, single demo `111222`, max-train-episodes=1
  - Both arms: 60 epochs, lr=1e-4, batch-size=2, TTT online, carry-windows T=16, prog-weight=1.0, seed=42
- **Result**: cls@224 final train_loss 30.59 vs patch@112 39.19; patch did not converge faster; both rollout press=0.
- **Decision**: keep cls@224 as default encoder.

## E07: progress head vs no progress head small comparison
- **Verifies**: progress head effect
- **Evidence**: `evidence/results/prog_vs_noprog_small.md`, `runs/b6_aux3_overfit_small2`, `runs/b7_noprog_main_small`
- **Setup**:
  - Data: `random_pw6_lang_small`, single demo `111222`, cls@224, 60 epochs, lr=1e-4, batch-size=2, TTT online, carry-windows T=16, seed=42
- **Result**: progress head consistently lower loss at matched epochs; both arms still no rollout success.

## E08: multi-password token-swap diagnostic and argmax inference fix
- **Verifies**: password-token conditioning and deterministic inference
- **Evidence**: `evidence/results/token_swap_diag.md`, `runs/b8_token_swap_diag`, `runs/b8_token_swap_diag_cont`
- **Setup**:
  - Data: `random_pw6_lang_100`, passwords `111222`+`222111`, 3 demos
  - Base 40 epochs + continued 30 epochs, cls@224, lr=1e-4, batch-size=2, prog-weight=0.0
- **Fix**: `_sample_action` now uses argmax at eval instead of multinomial.
- **Result**: continued model with argmax presses 3 times for correct `222111` token and 0 for wrong token; token-conditioned behavior observed.

## E09: medium cls@224 full-password training
- **Verifies**: scaling full-password cls training
- **Evidence**: `evidence/results/medium_cls_fullpw.md`, `runs/b9_cls_medium_fullpw_noprog*`, `runs/b9_cls_medium_fullpw_prog*`
- **Setup**:
  - Data: `random_pw6_lang_100`, default 48/16 split (80 train / 20 holdout episodes available), cls@224, lr=1e-4, batch-size=4, TTT online, carry-windows T=16
  - 30 epochs prog-weight=0.0, then 30 epochs prog-weight=1.0 with `runs/medium100_labels.npz`
- **Result**: first holdout success: password `122221` success=1 press=6 steps=344 (reproduced); overall holdout success 1/16; other holdout passwords press 4-5 keys.

## E10: full 48/16 cls@224 training (in progress)
- **Verifies**: formal 48/16 full-password training
- **Evidence**: `runs/b10_full48_cls` (stopped at epoch 4 for optimization), `runs/b10_full48_cls_opt` (in progress)
- **Setup**:
  - Data: `random_pw6_lang_1000` (764 train / 236 holdout episodes), default 48/16 split
  - cls@224, lr=1e-4, TTT online, carry-windows T=16, prog-weight=1.0, labels=`runs/full48_labels.npz`
  - Initial run batch=4, 70 min/epoch; optimized run: embedding disk cache + light shards + batch=8
  - Initial 10 epochs from epoch-4 snapshot, eval-freq=5, then continue to ≤50 epochs based on results
- **Result**: optimized run holdout success 13/16 (0.8125), mean_press 5.5, train sampled 2/2. See `evidence/results/full48_cls.md`.
- **Goal**: formal holdout success rate and final GIF/ARA evidence.
