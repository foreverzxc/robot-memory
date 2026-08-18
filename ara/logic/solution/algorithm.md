# Algorithm

## Fast-weight update (inner loop)

For layer \(l\), timestep \(t\), and batch element \(b\):

\[
\mathcal{L}_{FW}(W_b; K_{t,b}, V_{t,b}) = \mathrm{MSE}(f_{W_b}(K_{t,b}), V_{t,b})
\]

\[
W_b \leftarrow W_b - \eta\, \nabla_{W_b} \mathcal{L}_{FW}
\]

where \(K_{t,b}, V_{t,b}\) are per-layer Q/K/V projections of the current timestep tokens, \(f_W\) is a two-layer GELU MLP, and

\[
\eta = \mathrm{softplus}(\theta_\eta)
\]

is a learnable scalar initialized at the configured base learning rate.

The per-sample gradient is computed with `vmap(grad(_store), in_dims=(0,0))`, so batch elements never share gradient terms.

## Apply step

\[
O_t = X_t + \tanh(\alpha) \odot \mathrm{LayerNorm}(f_{W_t}(Q_t))
\]

with \(\alpha\) initialized at 0.001 per dimension.

## Outer loop

For a window of length \(T\) and per-timestep VQ-BeT loss \(\ell_t\):

\[
\mathcal{L} = \frac{1}{\sum_t m_t} \sum_t m_t \ell_t
\]

where \(m_t \in \{0,1\}\) is the valid-timestep mask (padding excluded). Gradients flow through every inner update into the GPT backbone, TTT projections, gate, fast-weight initialization, and inner learning rate.

## TBPTT carry

- Split episodes into non-overlapping chunks of \(T\).
- Carry fast-weight values across chunks.
- Detach fast-weight graphs every \(T\) timesteps (chunk boundary).
- The episode-initial fast-weight snapshot used by damp consolidation is carried with the state.

## Curriculum

For the first \(N\) epochs, a chunk produces task loss only when its remaining password length is at most

\[
\mathrm{cap}(e) = \max\left(1, \mathrm{round}\left(\frac{e+1}{N}\cdot 6\right)\right).
\]

Longer prefixes still update fast weights in a context-only forward pass.

## Complexity

- Per timestep and layer: one inner gradient step over an MLP with hidden size 256; memory scales with the number of TTT layers and batch size, not with total episode length.
