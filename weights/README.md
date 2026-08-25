# Local weights

Keep large model checkpoints out of Git. The rollout script expects the
following files locally:

```text
weights/decoder_ttt_pickxtimes_split80.pth
weights/decoder_ttt_swingxtimes_split80.pth
weights/pickxtimes_stats.json
weights/swingxtimes_stats.json
```

The two `*_stats.json` files are small normalization metadata generated from
the training splits and are tracked in this repository. Base TurboVLA,
DINOv3, and BERT weights are supplied through `--turbo-root`.
