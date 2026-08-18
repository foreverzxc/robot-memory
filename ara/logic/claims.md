# Claims

## C01: Per-layer test-time training with within-timestep attention provides a train/inference-consistent long-context memory pathway for visuomotor policies
- **Statement**: Restricting attention to within each timestep and inserting independently updated fast-weight modules after every attention block makes a T>1 training window numerically equivalent to sequential T=1 inference while still allowing history to flow across timesteps through the fast weights.
- **Conditions**: Holds for the two-button password task, patch encoder (patch@112), 8-layer GPT policy, batch size 4, sequence window 16, and the code paths tested in `test_button_ttt_pipeline.py`; untested for larger models, different action heads, and longer windows.
- **Sources**: []
- **Status**: supported
- **Falsification criteria**: A GPT+TTT model configured with `per_timestep_attn=True` and per-layer TTT would produce materially different last-timestep logits or fast-weight values when the same inputs are fed as one T>1 window versus sequential T=1 calls with carried fast weights.
- **Proof**: [E01]
- **Evidence basis**: Unit tests `ttt_chunked_equals_stepwise`, `gpt_per_timestep_attn_matches_stepwise`, and `gpt_per_layer_ttt_matches_stepwise` all pass (`scripts/test_button_ttt_pipeline.py`, 15/15 checks).
- **Dependencies**: none
- **Tags**: test-time training, train-inference consistency, fast weights

## C02: Per-sample inner-loop gradients make batched fast-weight training equivalent to single-episode inference
- **Statement**: Computing the fast-weight update with per-sample gradients — one independent K→V MSE gradient per batch element, in the style of `vmap(grad(...), in_dims=(0,0))` — removes cross-batch coupling and preserves the same per-episode update rule at training and inference batch sizes.
- **Conditions**: Holds for batched fast-weight states of shape (B, ...) in `button_task/ttt_layer.py` with valid-mask handling and TBPTT detachment; untested for very large B, distributed training, and alternative inner optimizers.
- **Sources**: []
- **Status**: supported
- **Falsification criteria**: A batched B=2 update would not match two independent B=1 updates on the same per-episode inputs within numerical tolerance.
- **Proof**: [E01]
- **Evidence basis**: Unit test `ttt_per_sample_update_equals_batch1` passes; `ttt_inner_update_descent_direction` confirms the update equals `W - lr * grad MSE`.
- **Dependencies**: C01
- **Tags**: per-sample update, vmap, fast weights

## C03: Ordering supervision from short remaining-password suffixes to full sequences stabilizes learning for long-password visuomotor tasks
- **Statement**: A remaining-length curriculum that first applies task loss only to short suffix contexts and gradually opens longer prefixes enables the recurrent memory mechanism to acquire useful progress tracking before full-sequence training, without revealing progress information to the policy.
- **Conditions**: Holds for the 48/16 train/holdout password split, 317 replay-labeled episodes, carry-windows T=16, and full-password conditioning (labels used only to schedule the curriculum); untested without labels, for other curricula, or for tasks whose progress cannot be measured by a scalar count.
- **Sources**: []
- **Status**: hypothesis
- **Falsification criteria**: Training the same model with curriculum disabled would reach equal or higher held-out success at matched epochs, or the training loss would fail to decrease monotonically across curriculum and full-sequence phases.
- **Proof**: [E02, E03, E04]
- **Evidence basis**: Exploratory b4 runs (which leaked progress via remaining-password conditioning) show monotonic loss decrease and improving holdout success; these are recorded with a leakage caveat. Formal full-password cls@224 runs use `curriculum_epochs=0` (no remaining-length curriculum) and reach holdout success 0.8125, so the curriculum itself remains untested under the no-leakage protocol.
- **Dependencies**: C01, C02
- **Tags**: curriculum learning, long-horizon, memory

## C04: A learnable softplus inner learning rate self-adjusts during per-layer TTT training
- **Statement**: When the inner-loop learning rate is parameterized as softplus and initialized small, outer-task gradients drive it upward during training as the fast-weight contribution becomes useful, rather than it remaining fixed at its initialization or clamp bound.
- **Conditions**: Holds for the per-layer TTT module with base lr 0.01, patch@112 input, and batch 4 on this task; untested for other initialization ranges and longer training beyond 30 total epochs.
- **Sources**: []
- **Status**: supported
- **Falsification criteria**: The logged `ttt/inner_lr` would remain at its initial value or decrease across checkpoints while the TTT gate contribution is increasing.
- **Proof**: [E02, E03, E04]
- **Evidence basis**: Run logs show `ttt/inner_lr` moving from 0.0100 (epoch 1) through 0.0136 (epoch 5), 0.0216 (epoch 11), 0.0260 (epoch 20), and 0.0271 (epoch 24); exact per-epoch values in run logs.
- **Dependencies**: C01
- **Tags**: meta-learning, inner learning rate, softplus

## C06: Deterministic argmax decoding is required for reliable VQ-BeT rollouts
- **Statement**: Sampling VQ centers with `torch.multinomial` at eval time adds rollout noise that prevents trained actions from being reproduced; using argmax at eval (while keeping sampling at training time) makes rollouts reproducible and materially improves action quality.
- **Conditions**: Holds for the VQ-BeT button policy (16 codes x 2 groups, action window 12) in this codebase; untested for other sampling strategies or temperatures.
- **Sources**: []
- **Status**: supported
- **Falsification criteria**: A deterministic eval would still fail to reproduce a known successful rollout (same checkpoint, same seed) when the action path is deterministic.
- **Proof**: [E08]
- **Evidence basis**: Before the fix, small models produced near-zero actions; after the fix, `122221` success (press=6, steps=344) reproduced exactly across training-internal and standalone eval. See `runs/b8_token_swap_diag_cont`, `runs/b9_cls_medium_fullpw_prog_cont2`.

## C07: Full-password cls@224 per-layer TTT solves most held-out passwords on the 48/16 button task
- **Statement**: With full constant password conditioning, cls@224 observations, per-layer TTT with progress-head supervision, and deterministic argmax eval, the trained policy succeeds on 13/16 held-out passwords at epoch 14-equivalent training (4 epochs batch=4 + 10 epochs batch=8 optimized).
- **Conditions**: Holds for `random_pw6_lang_1000`, lr=1e-4, prog-weight=1.0, carry-windows T=16, seed=42, 1-2 rollouts per password; not a multi-seed statistical result.
- **Sources**: []
- **Status**: supported
- **Falsification criteria**: Re-running the same checkpoint evaluation with the same protocol would produce a materially different held-out success rate.
- **Proof**: [E10]
- **Evidence basis**: `runs/b10_full48_cls_opt` holdout 13/16 = 0.8125 (2-repeat eval 26/32), mean_press 5.5; train sampled 2/2 success. Remaining failures: `222112`, `122221`, `212112`.

## C05: Replay-based label filtering with best-effort final-press correction preserves full password coverage for progress-conditioned training
- **Statement**: For replay-drifted demonstrations whose press sequence matches through the penultimate press, accepting replay counts and correcting only the final frame retains per-frame progress labels usable for remaining-password conditioning without excluding whole passwords from the training set.
- **Conditions**: Holds when replay final count differs from recorded final count by exactly one and replay length covers the episode; episodes drifting by more than one press or terminating early are excluded.
- **Sources**: []
- **Status**: supported
- **Falsification criteria**: The merged label set would fail to cover all train passwords, or best-effort episodes would show press-count errors before the final press when manually inspected.
- **Proof**: [E02]
- **Evidence basis**: `runs/full48_labels.npz` covers all 48 train passwords across 317 episodes; `scripts/label_missing_best_effort.py` produced 27 best-effort episodes for the three replay-failed passwords.
- **Dependencies**: C03
- **Tags**: labels, replay, data quality, best-effort
