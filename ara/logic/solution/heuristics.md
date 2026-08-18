# Heuristics

## H01: Restrict attention to within each timestep
- **Rationale**: Prevents training-time cross-timestep attention shortcuts that do not exist at T=1 inference; makes TTT the only cross-time pathway.
- **Sensitivity**: high
- **Bounds**: Must be combined with repeated per-timestep position embeddings for exact T>1 ≡ sequential T=1 equivalence.
- **Code ref**: `vendor/patch_policy/models/vq_behavior_transformer/gpt.py`
- **Source**: experiment code review; `--per-step-attn` default in `train_button_ttt.py`

## H02: Per-layer TTT instead of a single post-ln_f module
- **Rationale**: Increases memory capacity and places associative updates close to each block's representation.
- **Sensitivity**: high
- **Bounds**: One TTTSequence per GPT block; only the last module owns the progress head.
- **Code ref**: `train_button_ttt.py::build_model`
- **Source**: RoboTTT paper §3.1

## H03: Per-sample inner gradients
- **Rationale**: Eliminates 1/B scaling of inner updates from batch-averaged MSE, matching inference at B=1.
- **Sensitivity**: high
- **Bounds**: Valid-masked timesteps carry fast weights unchanged.
- **Code ref**: `button_task/ttt_layer.py::inner_update`
- **Source**: official RoboTTT `robo_ttt.py` `MemoryKeyValueBind`

## H04: Softplus inner learning rate initialized small
- **Rationale**: Allows the outer loop to grow or shrink the inner step size instead of clamping it at initialization.
- **Sensitivity**: medium
- **Bounds**: Initialized at 0.01 for the current runs; larger initialization was not tested here.
- **Code ref**: `button_task/ttt_layer.py::TTTLayer.inner_lr`
- **Source**: experiment code; official RoboTTT `MemoryKeyValueBind.learnable_lr`

## H05: Remaining-length curriculum with context-only long prefixes
- **Rationale**: Early epochs supervise only short suffixes so the memory mechanism first learns to track progress near task completion; long prefixes still update fast weights as context.
- **Sensitivity**: medium
- **Bounds**: Requires per-frame press-count labels; cap ramps 1→6 over `curriculum_epochs`.
- **Code ref**: `train_button_ttt.py`
- **Source**: experiment design decided by researcher

## H06: Best-effort labels for replay-drifted final presses
- **Rationale**: Prevents losing whole passwords when replay misses only the last press; corrects the final frame to the recorded press count.
- **Sensitivity**: medium
- **Bounds**: Only when replaypc == h5pc - 1; larger drift is rejected.
- **Code ref**: `scripts/label_missing_best_effort.py`
- **Source**: experiment design
