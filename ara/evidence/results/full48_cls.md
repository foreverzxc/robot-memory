# Result: full 48/16 cls@224 full-password training

## Setup
- Data: `random_pw6_lang_1000`, default 48/16 split, 764 train / 236 holdout episodes
- cls@224, TTT online, carry-windows T=16, lr=1e-4, prog-weight=1.0, labels=`runs/full48_labels.npz`, argmax eval fix applied
- Runs:
  - `b10_full48_cls`: batch=4, stopped after epoch 4 for optimization (70 min/epoch)
  - `b10_full48_cls_opt`: resumed from epoch-4 snapshot, batch=8, embedding cache + light shards, 10 epochs (total epoch 14)

## Training result (optimized run)
- Epoch 0 (cache build): 3078s
- Epochs 1-9: ~2624-2696s each
- Timing breakdown (final epoch): embed=0.9s, train=2615.3s, context=0.0s, other=8.3s
- Final train_loss=9.47

## Holdout eval (snapshot epoch 9)
- Training-run eval (1 rollout per password): **success rate 13/16 = 0.8125**, mean_press=5.5
- Formal eval script with 2 repeats per password: **26/32 = 0.8125**
  - 13 passwords 2/2 success; 3 passwords 0/2: `222112` (press 2), `122221` (press 4), `212112` (press 4)
- Train sampled: `121121` and `112111` both success press=6

## Optimization result
- Old run: 4200s/epoch, 4155 steps/epoch, batch=4
- Optimized: ~2630s/epoch after cache, 2132 steps/epoch, batch=8, DINO embedding cached on disk
