# Result: b6_aux3_overfit_small2 (three auxiliary heads, corrected gradient policy)

- **Trace node**: in progress (3-head probe design)
- **Claims**: C04 (progress head learns); count/next-key heads are observational probes with no TTT gradient

| run_id | epoch | train_loss | classification_loss | offset_loss | prog_mse | aux_count_acc | aux_nextkey_acc |
|--------|-------|------------|---------------------|-------------|----------|---------------|-----------------|
| b6_aux3_overfit_small2 | 59 (final) | 30.585637 | 1.479236 | 2.907586 | 0.030564 | 0.119048 | 0.485119 |

Simulator eval at epoch 59: holdout and train-sampled `111222` both failed with press_count=0 (max 600 steps).

Gradient policy: only the progress head (normalized press count, task-agnostic) receives TTT gradients; count head (7-class press count) and next-key head (2-class next key) are task-related probes computed on `feat.detach()`, contribute no loss, and do not train TTT.

Interpretation: the progress head learns (prog_mse decreases from ~0.077 to ~0.031), while the task-related probes remain near chance (count 0.119 vs 0.143 random; next-key 0.485 vs 0.5 random), as expected because no task-specific supervision is allowed to enter TTT. Rollout still fails before the first press; password-token read diagnostic is the next step.
