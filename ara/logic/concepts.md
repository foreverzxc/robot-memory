# Concepts

## Test-Time Training (TTT)
- **Notation**: \(W_t \leftarrow W_{t-1} - \eta \nabla_W \mathcal{L}_{FW}(f_{W_{t-1}}(K_t), V_t)\)
- **Definition**: A recurrent memory where the hidden state is a set of fast weights of a small model, updated by gradient descent on a self-supervised loss at each timestep during both training and inference.
- **Boundary conditions**: Applies to sequential inputs; requires differentiability of the fast model and an inner loss.
- **Related concepts**: fast weights, inner loop, meta-learning

## Fast weights
- **Notation**: \(W\)
- **Definition**: Parameters updated per timestep (or per episode) during the model's forward pass, in contrast to slow weights updated only by the outer optimizer.
- **Boundary conditions**: Must be carried explicitly across sequence chunks; TBPTT detaches their graph at chunk boundaries.
- **Related concepts**: slow weights, Test-Time Training, TBPTT

## Per-layer TTT
- **Notation**: \(\mathrm{TTT}_l\) for GPT block \(l\)
- **Definition**: Placing one independent TTT module after each attention block of a transformer; each module owns its own fast weights, projections, gate, and inner learning rate.
- **Boundary conditions**: Requires per-timestep token streams at each block; the final module owns the auxiliary progress head.
- **Related concepts**: TTT, transformer block, per-step attention

## Per-sample inner update
- **Notation**: \(\nabla_{W_b} \mathrm{MSE}(f_{W_b}(k_b), v_b)\)
- **Definition**: Computing one independent inner gradient per batch element (via `vmap(grad(...), in_dims=(0,0))`) so each episode's fast-weight state is updated by its own tokens only.
- **Boundary conditions**: Valid-masked timesteps carry their fast weights unchanged.
- **Related concepts**: fast weights, vmap, train-inference consistency

## Per-step attention
- **Notation**: block-diagonal attention mask \(M\)
- **Definition**: Restricting transformer attention so tokens at timestep t attend only to tokens of timestep t (plus shared condition tokens), making TTT the only cross-time pathway.
- **Boundary conditions**: Requires per-timestep position embeddings that repeat across the window to make T>1 windows equal to sequential T=1 calls.
- **Related concepts**: attention mask, TTT, train-inference consistency

## Remaining-password conditioning
- **Notation**: \(\mathrm{suffix}(c) = p[c:]\)
- **Definition**: Feeding the policy the remaining suffix of the password after `c` successful presses, PAD-filled to the fixed token length, so the task token always reflects what is left to do.
- **Boundary conditions**: Requires per-frame press-count labels; episodes without labels are filtered or use full-password fallback depending on protocol.
- **Related concepts**: curriculum, task conditioning, password task

## Carry-windows training with TBPTT
- **Notation**: chunk size \(T\), detach every \(T\) timesteps
- **Definition**: Training over non-overlapping chunks of an episode while carrying fast-weight *values* across chunk boundaries and detaching their *graphs* at each boundary.
- **Boundary conditions**: The TBPTT step size must divide the chunk size so returned fast weights are graph-free.
- **Related concepts**: TBPTT, fast weights, BPTT

## Remaining-length curriculum
- **Notation**: \(\mathrm{cap}(\mathrm{epoch}) = \max(1, \mathrm{round}((\mathrm{epoch}+1)/N \cdot 6))\)
- **Definition**: During the first N epochs, only chunks whose remaining password length is at or below the cap produce task loss; longer prefixes are context-only (fast weights still update).
- **Boundary conditions**: Requires per-frame labels; after N epochs the cap is 6 (full sequences).
- **Related concepts**: curriculum learning, remaining-password conditioning
