# Problem Specification

## Observations

### O1: Single-step baselines cannot solve non-Markov password tasks
- **Statement**: A VQ-BeT button policy trained on single observations (T=1) reads the password tokens but collapses to repeating the first observed key; only password 222222 was solved in early single-password fits.
- **Evidence**: project handoff records (`HANDOFF.md`); baseline runs in `runs/b1_*`.
- **Implication**: The policy needs memory of which buttons have already been pressed.

### O2: A fixed recurrent vector state is one design option, but fast weights offer higher capacity
- **Statement**: RoboTTT-style TTT stores history in the parameters of a small fast model updated by gradient descent at every timestep, rather than in a vector-valued recurrent state.
- **Evidence**: RoboTTT paper (`2607.15275v1.pdf`, §2 Eq. 1–2).
- **Implication**: Adopting this memory mechanism may scale context without growing inference cost.

### O3: A single TTT module after the GPT `ln_f` underperforms
- **Statement**: Early runs with one TTT module after the final layer norm, CLS encoder features (2 tokens/timestep), and clamped learnable inner lr did not solve held-out passwords (0% holdout in early runs).
- **Evidence**: `runs/b3_ttt_carry/log.jsonl` (holdout 0.0 at epoch 39) and code review notes.
- **Implication**: Placement, token count, and inner-lr parameterization materially affect TTT capacity.

### O4: Label generation via environment replay is lossy
- **Statement**: Replaying all training demos with `label_button_demos.py` produced per-frame press-count labels for 290 episodes out of 764 train episodes; 45 of 48 passwords were covered, and the last press often drifts near episode end.
- **Evidence**: background label job output; `runs/full_train_labels.npz`.
- **Implication**: Remaining-password conditioning needs a robust label policy and must handle unlabeled episodes explicitly.

## Gaps

### G1: Cross-time attention breaks train/inference equivalence
- **Statement**: When GPT attention is block-lower-triangular over timesteps, training with T>1 lets later timesteps attend directly to earlier frames, but inference uses T=1 and loses that pathway.
- **Caused by**: O3 and the original GPT attention mask implementation.
- **Existing attempts**: Original `generate_mask_matrix` block-causal mask.
- **Why they fail**: Training-time shortcut unavailable at inference; also violates RoboTTT's "attention within a timestep, TTT across timesteps" design.

### G2: Batched fast-weight updates are not per-episode
- **Statement**: A scalar MSE loss averaged over the batch scales each episode's inner gradient by 1/B, so training update magnitude differs from B=1 inference.
- **Caused by**: Batched `F.mse_loss` default `mean` over all elements.
- **Existing attempts**: Initial `inner_update` implementation.
- **Why they fail**: The learned inner lr is calibrated to a different update scale than inference.

### G3: No short-to-long curriculum
- **Statement**: Without curriculum, all sequence positions receive task loss from the first epoch; long prefixes dominate before the memory mechanism is trained.
- **Caused by**: Full-sequence loss masking in early code.
- **Existing attempts**: Full-sequence `--carry-windows` training.
- **Why they fail**: Early training signals do not focus on learnable short suffixes.

## Key Insight
- **Insight**: Make TTT the *only* cross-time pathway — restrict attention to within each timestep, place a TTT module after every attention block, update each batch element's fast weights independently with per-sample gradients, and order supervision from short remaining-password suffixes to full sequences.
- **Derived from**: O1–O4 and G1–G3.
- **Enables**: Train/inference-equivalent long-context memory and stable curriculum learning.

## Assumptions
- A1: The action distribution is learned with a VQ-BeT head (classification + offset regression), matching the Patch Policy baseline.
- A2: Per-frame press-count labels are needed for remaining-password conditioning and the curriculum.
- A3: Best-effort labels for episodes whose replay misses only the final press are acceptable if the final frame is corrected and the uncertainty is documented.
- A4: Simulator success (6 correct presses, no failure) is the primary evaluation outcome.
