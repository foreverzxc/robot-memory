# Result: progress head vs no progress head (small overfit, cls@224)

## Setup
- Data: `random_pw6_lang_small`, single demo `111222`, max-train-episodes=1
- Both arms: cls@224, 60 epochs, lr=1e-4, batch-size=2, TTT online, carry-windows T=16, seed=42
- Arms:
  - with progress head: `runs/b6_aux3_overfit_small2` (prog-weight=1.0)
  - no progress head: `runs/b7_noprog_main_small` (prog-weight=0.0)

## Training loss at sampled epochs
| epoch | prog train_loss | noprog train_loss |
|---|---|---|
| 9 | 73.24 | 74.57 |
| 19 | 60.92 | 62.63 |
| 29 | 49.51 | 51.95 |
| 39 | 41.98 | 44.84 |
| 49 | 33.85 | 41.82 |
| 59 | 30.59 | 32.72 |

## Final eval (`111222`)
- with progress head: success 0, press 0, steps 600
- no progress head: success 0, press 0, steps 89 (failed)

## Interpretation
- Progress head consistently gives lower training loss at matched epochs in this small run, suggesting it helps convergence.
- Both arms still fail to press, so final success is not yet informative.
