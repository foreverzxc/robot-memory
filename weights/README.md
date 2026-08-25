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
the training splits and are tracked in this repository. On the first rollout,
the base checkpoint, DINOv3, and BERT are downloaded automatically from
Hugging Face into this directory. DINOv3 may require `hf auth login`.

The two decoder-TTT checkpoints are private experiment outputs and are not
uploaded to GitHub. Copy them here from the training machine:

```text
weights/decoder_ttt_pickxtimes_split80.pth
weights/decoder_ttt_swingxtimes_split80.pth
```
