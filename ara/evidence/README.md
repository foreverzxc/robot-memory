# Evidence Index

## Tables
| File | Source | Claims | Description |
|------|--------|--------|-------------|
| [tables/table1_training_loss.md](tables/table1_training_loss.md) | run logs `b4_ttt_perlayer_patch112_{smoke,ep10,ep20,ep30}/log.jsonl` | C03, C04 | Exact per-total-epoch training loss values |
| [tables/table2_eval_results.md](tables/table2_eval_results.md) | run logs (eval_holdout/eval_train fields) | C03 | Simulator success rates at checkpoints |

## Figures
| File | Source | Claims | Description |
|------|--------|--------|-------------|
| [figures/figure1_training_loss.md](figures/figure1_training_loss.md) | derived from table1_training_loss | C03, C04 | Training loss curve with holdout-eval checkpoints marked |

## Results
| File | Run | Claims |
|------|-----|--------|
| [results/smoke5.md](results/smoke5.md) | `runs/b4_ttt_perlayer_patch112_smoke` | C03, C04 |
| [results/ep10.md](results/ep10.md) | `runs/b4_ttt_perlayer_patch112_ep10` | C03, C04 |
| [results/ep20.md](results/ep20.md) | `runs/b4_ttt_perlayer_patch112_ep20` | C03, C04 |
| [results/ep30.md](results/ep30.md) | `runs/b4_ttt_perlayer_patch112_ep30` (in progress) | C03, C04 |
| [results/b6_aux3_overfit_small2.md](results/b6_aux3_overfit_small2.md) | `runs/b6_aux3_overfit_small2` (3-head probe design) | C04 |

## Logs
| File | Description |
|------|-------------|
| [logs/log_pointers.md](logs/log_pointers.md) | Direct pointers to each run's `log.jsonl` and args |
