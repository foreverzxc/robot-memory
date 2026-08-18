# Related Work

## RW01: RoboTTT: Context Scaling for Robot Policies
- **DOI**: arXiv:2607.15275
- **Type**: extends
- **Delta**:
  - What changed: This experiment applies RoboTTT's fast-weight inner-loop to a Patch Policy VQ-BeT button task; adds per-step attention masking, per-layer TTT placement, and remaining-length curriculum; the paper targets GR00T N1.7/DiT action heads and 8K-timestep contexts.
  - Why: Reuse TTT's train/inference-consistent memory while adapting to the available small robot task.
- **Claims affected**: C01, C02, C03, C04
- **Adopted elements**: K→V MSE fast-weight update, gated residual apply, TBPTT carry/detach, per-sample inner gradients.

## RW02: Test-Time Training
- **DOI**: see RoboTTT §2 references (64; 78)
- **Type**: imports
- **Delta**:
  - What changed: Provides the base TTT mechanism (fast weights updated during both training and inference).
  - Why: Foundational mechanism reused here.
- **Claims affected**: C01
- **Adopted elements**: inner-loop update-then-apply.

## RW03: Patch Policy (vendor codebase)
- **DOI**: local repo `E:\deepseekharness\patch_policy_ttt\vendor\patch_policy`
- **Type**: baseline
- **Delta**:
  - What changed: Provides the VQ-BeT policy, button env wrapper, and DINOv2 encoder; the experiment inserts per-layer TTT into its GPT.
  - Why: Keep the action head and data pipeline fixed while changing memory.
- **Claims affected**: C03
- **Adopted elements**: VQ-BeT head, Patch Policy slicing/embedding, ButtonPatchWrapper.

## RW04: Official RoboTTT implementation
- **DOI**: local repo `E:\deepseekharness\robo_ttt`
- **Type**: extends
- **Delta**:
  - What changed: `MemoryKeyValueBind`'s `vmap(grad(_store), in_dims=(0,0))` and `TTTWrapper` are reimplemented in pure torch for the button task; Muon, forget gates, and RoPE are not ported.
  - Why: Avoid extra dependencies while preserving the per-sample update semantics.
- **Claims affected**: C02, C04
- **Adopted elements**: per-sample store gradients, per-layer wrapper loop, layerscale gate.
