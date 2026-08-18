# Table 2: Simulator evaluation results at checkpoints

**Source**: `runs/b4_ttt_perlayer_patch112_smoke/log.jsonl`, `runs/b4_ttt_perlayer_patch112_ep10/log.jsonl`, `runs/b4_ttt_perlayer_patch112_ep20/log.jsonl`
**Caption**: Holdout (16 passwords) and train-sampled (4 passwords) simulator success rates at the evaluation checkpoints. One rollout per password, max 600 steps. NOTE: these b4 runs used remaining-password conditioning at eval (ground-truth press-count leakage) and are exploratory; formal full-password runs are recorded separately.
**Screenshot**: table2_eval_results.png
**Extraction type**: raw_table

| Checkpoint | Holdout success rate | Train-sampled success rate |
|---|---|---|
| epoch 5 | 0.125 (2/16) | 0.0 (0/4) |
| epoch 10 | 0.625 (10/16) | 0.75 (3/4) |
| epoch 20 | 0.9375 (15/16) | 1.0 (4/4) |
| epoch 30 | pending | pending |
