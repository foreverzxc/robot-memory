# Base Policy Runtime

This is the small inference-only subset of the TurboVLA model implementation
needed to construct the frozen RoboMME base policy. It is vendored here so
rollout does not depend on a separate source checkout. Large checkpoints and
backbones are downloaded into `weights/` by `downloads.py`.
