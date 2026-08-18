# Figure 1: Training loss curve
- **Source**: derived from `evidence/tables/table1_training_loss.md`
- **Caption**: Mean training loss versus total epoch; red points mark epochs where a holdout simulator evaluation was performed (annotated with holdout success rate).
- **Screenshot**: figure1_training_loss.png
- **Figure type**: quantitative_plot
- **Extraction method**: exact_from_labels
- **Reading confidence**: high

- **Plot kind**: line
- **Axes**: X = total epoch (linear), Y = train_loss (linear)

| Total epoch | train_loss |
|---|---|
| 1 | 0.000000 |
| 2 | 77.027559 |
| 3 | 61.314461 |
| 4 | 35.620969 |
| 5 | 33.872812 |
| 6 | 25.700459 |
| 7 | 23.323382 |
| 8 | 20.392707 |
| 9 | 18.730401 |
| 10 | 17.162408 |
| 11 | 15.949176 |
| 12 | 14.715583 |
| 13 | 13.069460 |
| 14 | 12.551645 |
| 15 | 11.254215 |
| 16 | 10.766561 |
| 17 | 10.130674 |
| 18 | 10.092176 |
| 19 | 9.492808 |
| 20 | 9.061348 |
| 21 | 8.927923 |
| 22 | 8.888893 |
| 23 | 8.295226 |
| 24 | 8.140962 |
| 25 | 8.103082 |

## Trend summary
Training loss decreases monotonically after the VQ-fit epoch, with large drops during the curriculum phase (epochs 1–5) and smaller, continuing decreases through epoch 25. Evaluation checkpoints (epochs 5, 10, 20) show increasing holdout success.
