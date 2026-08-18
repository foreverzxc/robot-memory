"""Pipeline checks for Stage B1 (VQ-BeT + password tokens), no simulator.

Run with the button venv:

    E:\\WM\\turbovla\\.venv\\Scripts\\python.exe scripts\\test_button_train_pipeline.py

Checks:
1. vendor model imports work without accelerate (shim path)
2. GPT condition-token mask structure and forward shape
3. ConditionTokenEncoder seq/sum/lookup embeddings
4. BehaviorTransformer end-to-end forward/backward with password conditioning
5. ButtonSliceDataset slice/chunk correctness against the raw HDF5
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "patch_policy"))

import torch
import einops

PASS = []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"[PASS] {name}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        raise


def test_gpt_cond_mask():
    from models.vq_behavior_transformer.gpt import GPT, GPTConfig

    cfg = GPTConfig(
        block_size=2, input_dim=4, output_dim=8, n_patches=4,
        n_layer=2, n_head=2, n_embd=16, dropout=0.0, cond_len=2,
    )
    gpt = GPT(cfg)
    t = 2  # two timesteps of 4 patches each
    p = 4
    x = torch.randn(2, t, p, 4)
    cond = torch.randn(2, 2, 4)
    out, fw, stats = gpt(x, cond=cond)
    assert out.shape == (2, t, 8), out.shape
    assert fw is None and stats is None

    # mask structure: [cond(2); obs(8)] x same
    mask = gpt._attention_mask(t, 2, torch.device("cpu"))[0, 0]
    assert mask.shape == (10, 10)
    assert mask[:2, :2].all(), "cond tokens must see all cond tokens"
    assert mask[2:, :2].all(), "obs tokens must see all cond tokens"
    # obs rows are causal over time blocks (width 4): block i sees blocks <= i
    assert mask[2:, 2:].shape == (8, 8)
    assert not mask[2, 7], "causal: row block 0 must not see col block 1"
    assert mask[7, 5], "causal: row block 1 may see col block 0"
    assert mask[4, 5] and mask[7, 7], "within-block attention must be allowed"

    # no-cond path stays compatible
    gpt0 = GPT(GPTConfig(block_size=2, input_dim=4, output_dim=8, n_patches=4,
                         n_layer=1, n_head=2, n_embd=16, dropout=0.0, cond_len=0))
    out0, _, _ = gpt0(x)
    assert out0.shape == (2, t, 8)

    # gradient flows to cond through the GPT
    out.sum().backward()
    assert gpt.transformer.wte.weight.grad is not None


def test_condition_tokens():
    from models.vq_behavior_transformer.condition_tokens import ConditionTokenEncoder, PAD_IDX

    idx = torch.tensor([[0, 1, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX],
                        [1, 0, 1, 2, 2, 2]])  # pw "12" and "121"
    enc = ConditionTokenEncoder(dim=32, max_len=6, num_symbols=2, mode="seq")
    tok = enc.forward_idx(idx)
    assert tok.shape == (2, 6, 32)
    # padded positions share the same pad token
    assert torch.equal(tok[0, 2], tok[0, 3])

    enc_sum = ConditionTokenEncoder(dim=32, max_len=6, num_symbols=2, mode="sum")
    assert enc_sum.forward_idx(idx).shape == (2, 1, 32)

    enc_lookup = ConditionTokenEncoder(dim=32, max_len=6, num_symbols=2, mode="lookup")
    assert enc_lookup.forward_idx(idx).shape == (2, 1, 32)
    # different passwords -> different lookup tokens
    a = enc_lookup.forward_idx(idx[:1])
    b = enc_lookup.forward_idx(idx[1:])
    assert not torch.allclose(a, b)


def test_behavior_transformer_fwd_bwd():
    from models.vq_behavior_transformer.bet import BehaviorTransformer

    B, T, P, E = 4, 1, 6, 8
    W, A = 4, 3
    model = BehaviorTransformer(
        obs_dim=8, act_dim=3, goal_dim=0, views=1,
        vqvae_latent_dim=32, vqvae_n_embed=8, vqvae_groups=2,
        vqvae_fit_steps=None, vqvae_iters=5,
        n_patches=P, n_layer=2, n_head=2, n_embd=16,
        dropout=0.0, vqvae_encoder_loss_multiplier=1.0, vqvae_batch_size=16,
        act_scale=1.0, offset_loss_multiplier=1.0,
        obs_window_size=1, act_window_size=W,
        cond_len=3, cond_mode="seq", cond_num_symbols=2,
        gpt_block_size=1, vqvae_max_samples=64,
    )
    obs = torch.randn(B, T, P, E)
    act = torch.randn(B, W, A) * 0.2
    pw = torch.randint(0, 3, (B, 3))

    # collect a few batches, then force-fit the VQ (as train_button.py does at
    # the end of epoch 1). Before the VQ fit the loss is zeroed by design.
    for _ in range(3):
        pred, loss, _, _ = model(obs, None, act, password_idx=pw)
    model.finish_epoch()
    assert model.vqvae_is_fit

    pred, loss, loss_dict, _ = model(obs, None, act, password_idx=pw)
    assert pred.shape == (B, T, W, A), pred.shape
    assert loss is not None and loss.item() > 0

    opt = model.configure_optimizers(weight_decay=0.0, learning_rate=1e-3, betas=(0.9, 0.999))
    opt.zero_grad()
    loss.backward()
    # cond encoder must receive gradient
    cond_grad = model.cond_encoder.table.grad
    assert cond_grad is not None and cond_grad.abs().sum() > 0
    opt.step()

    # inference path without actions
    model.eval()
    pred3, loss3, _, _ = model(obs, None, None, password_idx=pw)
    assert loss3 is None and pred3.shape == (B, T, W, A)

    # password_idx must be provided when cond_len > 0
    try:
        model(obs, None, None)
        raise AssertionError("expected ValueError for missing password_idx")
    except ValueError:
        pass


def test_button_slice_dataset():
    from button_task.train_dataset import ButtonSliceDataset

    h5 = "E:/WM/turbovla/data/button_demos/random_pw6_lang_small/demos.h5"
    ds = ButtonSliceDataset(
        h5, action_window=12, shard_size=4, seed=0, max_episodes=4,
    )
    assert ds.num_episodes == 4
    total_slices = sum(ds.base.get_seq_length(i) for i in range(4))
    assert ds.num_slices == total_slices

    for shard in ds.iter_shards(shuffle=False):
        chunks = ds.build_chunks(shard)
        assert chunks.shape == (shard.starts.shape[0], 12, 7)
        # boundary safety: check a slice near an episode end doesn't borrow
        # the next episode's first action
        pos = 0
        for T in shard.episode_lengths:
            # slice at last frame of this episode: chunk repeats last action
            last_chunk = chunks[pos + T - 1]
            expected = shard.action[pos + T - 1]
            assert torch.allclose(last_chunk, expected.expand(12, -1)), (
                "chunk at episode end must repeat the last action"
            )
            # slice one before boundary: chunk = [a_{T-2}, a_{T-1}, a_{T-1}...]
            if T >= 2:
                c = chunks[pos + T - 2]
                assert torch.allclose(c[0], shard.action[pos + T - 2])
                assert torch.allclose(c[1], shard.action[pos + T - 1])
                assert torch.allclose(c[2:], shard.action[pos + T - 1].expand(10, -1))
            pos += T
        # slice_obs shapes
        idx = torch.arange(min(8, shard.starts.shape[0]))
        obs = ds.slice_obs(shard, idx, torch.device("cpu"))
        assert obs.shape == (len(idx), 1, 2, 3, 224, 224)
        assert obs.min() >= 0 and obs.max() <= 1
        break


def main():
    check("gpt_cond_mask", test_gpt_cond_mask)
    check("condition_tokens", test_condition_tokens)
    check("behavior_transformer_fwd_bwd", test_behavior_transformer_fwd_bwd)
    check("button_slice_dataset", test_button_slice_dataset)
    print(f"\nall {len(PASS)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
