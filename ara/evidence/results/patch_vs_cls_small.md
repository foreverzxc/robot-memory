# Result: patch@112 vs cls@224 small overfit (full-password protocol)

## Setup
- Data: `random_pw6_lang_small`, single demo `111222`, max-train-episodes=1
- Both arms: 60 epochs, lr=1e-4, batch-size=2, TTT online, carry-windows T=16, prog-weight=1.0, seed=42
- Arms:
  - cls@224: `runs/b6_aux3_overfit_small2`
  - patch@112: `runs/b7_patch_overfit_small`

## Final epoch (epoch 59)
| metric | cls@224 | patch@112 |
|---|---|---|
| train_loss | 30.586 | 39.189 |
| classification_loss | 1.479 | 1.921 |
| offset_loss | 2.908 | 3.723 |
| prog_mse | 0.0306 | 0.0361 |
| aux_count_acc | 0.119 | 0.146 |
| aux_nextkey_acc | 0.485 | 0.554 |
| rollout success | 0 | 0 |

## Interpretation
- In this small single-demo setting, patch@112 did **not** converge faster or reach lower loss than cls@224.
- Both fail to press in simulation.
- Decision: keep cls@224 as the default formal encoder. patch@224 could be revisited separately if desired.
