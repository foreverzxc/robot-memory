# Constraints

- **Task domain**: two-button password task, passwords of length 6 over alphabet {1,2}; 64 possible passwords; 48 train / 16 holdout.
- **Data**: `random_pw6_lang_1000` HDF5; only 317 replay-labeled train episodes used for curriculum/remaining-password training; 290 strict labels plus 27 best-effort labels.
- **Best-effort label assumption**: for the three replay-failed passwords, replay counts are trusted through the penultimate press; only the final frame is corrected to the recorded final press count. Episodes whose replay differs by more than one press are excluded.
- **Model**: Patch Policy VQ-BeT; GPT n_layer=8, n_head=8, n_embd=512, gpt_block_size=16; patch@112 with 128 observation tokens per timestep; 8 per-layer TTT modules; fast MLP hidden 256.
- **Training protocol**: carry-windows T=16, batch size 4, bf16 autocast, lr 1e-4, weight decay 2e-4, inner lr softplus initial 0.01, TBPTT detach at chunk boundaries, 5-epoch remaining-length curriculum (labels schedule the loss only) then full sequences. **The policy always receives the full 6-digit password; remaining-password conditioning is not used in formal runs because it leaks ground-truth progress.**
- **Evaluation**: one rollout per password in the simulator, max 600 steps, action chunk averaging over 12 predicted chunks; holdout split is fixed by `password_split.json`.
- **Known limitations**:
  - Only 5–30 total epochs currently recorded; final convergence check still in progress.
  - Best-effort labels for three passwords carry tail-frame uncertainty near the final press.
  - Evaluation uses a single seed/rollout per password; no multiple-trial statistics.
  - No forget gate, Muon update, or RoPE in the fast-weight module (present in the official RoboTTT implementation).
- **Total budget**: training will not exceed 50 total epochs.
