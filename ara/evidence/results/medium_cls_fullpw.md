# Result: medium cls@224 full-password training (random_pw6_lang_100)

## Setup
- Data: `random_pw6_lang_100`, default 48/16 password split; actual 80 train / 20 holdout episodes
- cls@224, TTT online, carry-windows T=16, lr=1e-4, batch-size=4, seed=42, argmax eval fix applied
- Runs:
  - `b9_cls_medium_fullpw_noprog`: 10 epochs, prog-weight=0.0
  - `b9_cls_medium_fullpw_noprog_cont`: +10 epochs (20 total), prog-weight=0.0
  - `b9_cls_medium_fullpw_noprog_cont2`: +10 epochs (30 total), prog-weight=0.0
  - `b9_cls_medium_fullpw_prog`: +10 epochs (30 total + new heads), prog-weight=1.0, labels=`runs/medium100_labels.npz` (36 episodes)
  - `b9_cls_medium_fullpw_prog_cont`: +10 epochs (40 total + 20 with heads), prog-weight=1.0
  - `b9_cls_medium_fullpw_prog_cont2`: +10 epochs (50 total + 30 with heads), prog-weight=1.0

## Results
| run (final) | train_loss | holdout success | holdout mean_press | train sampled press |
|---|---|---|---|---|
| noprog 10ep | 50.09 | 0.0 | 0.81 | 0 |
| noprog 20ep | 32.85 | 0.0 | 0.75 | 2 |
| noprog 30ep | 25.95 | 0.0 | 0.69 | 0 |
| prog 10ep on top | 20.63 | 0.0 | 1.19 (epoch4 eval: 2.5) | 4 |
| prog cont 10ep | 16.66 | 0.0 | 2.19 | 1 |
| prog cont2 10ep | 14.61 | 0.0625 (1/16) | epoch8 eval: `122221` success press=6 | 1 |

## First holdout success
- Password `122221` (holdout): success=1, failed=0, press_count=6, steps=344.
- Reproduced identically with `scripts/eval_button_checkpoint.py` (argmax eval): success 1/1, press 6, steps 344.
- Same eval shows other holdout passwords pressing 4-5 keys (e.g. 112212: first four keys left-left-right-right correct).

## Interpretation
- Full-password cls@224 + progress head + argmax eval is **validated**: the first holdout password can be solved.
- Success rate is still low (1/16), so further training or scaling to the full 48/16 dataset is the next step.
