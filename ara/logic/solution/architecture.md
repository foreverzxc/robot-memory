# Architecture

## Component graph

| Component | Purpose | Inputs | Outputs | Key design choices |
|---|---|---|---|---|
| DINOv2 encoder | Visual features | 2 camera images at 112×112 | per-timestep patch tokens (2 × 64 × 384) | patch mode; encoder frozen |
| GPT backbone | Per-timestep observation encoding | patch tokens + 6 password condition tokens | per-block per-timestep token streams | 8 layers, 8 heads, n_embd 512; per-step attention mask |
| TTT modules | Cross-timestep memory | each block's observation token stream (B,T,N,512) | gated residual stream + next fast weights | one TTTSequence per GPT block; fast MLP 256 hidden; softplus inner lr |
| Fast model | Associative memory | Q/K/V projections of the stream | K→V reconstructed output | two-layer GELU MLP |
| VQ-BeT head | Action prediction | final per-timestep observation feature | action chunks (T×12×7) | VQ codebook 16 × 2 groups + offset regression |
| Aux progress head | Progress supervision | last TTT module's output stream | press-count progress in [0,1] | Sigmoid head; gradient only into TTT |

## Data flow

1. For each timestep t in a window: DINOv2 embeds the current two camera views into patch tokens.
2. GPT processes the window with block-diagonal attention: timestep t attends only to its own patches plus the shared password condition tokens.
3. After each GPT block l, `TTTSequence_l` updates its fast weights with one gradient step on `MSE(f_W(K_t), V_t)` and outputs `x + tanh(alpha) * f_W(Q_t)`.
4. The final layer norm output feeds the VQ-BeT head, which predicts an action chunk for every timestep.
5. The outer loss is computed for every timestep and averaged over valid timesteps.

## Cross-time pathway

- Attention across timesteps is disabled (`per_timestep_attn=True`).
- History flows only through the per-layer fast weights, carried across timesteps and chunks.
- TBPTT detaches fast-weight graphs at chunk boundaries while preserving their values.
