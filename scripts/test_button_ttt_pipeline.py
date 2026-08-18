"""B2/B3 pipeline checks: TTT sequence layer, GPT integration, sequence windows.

Run with the button venv:
    E:\\WM\\turbovla\\.venv\\Scripts\\python.exe scripts\\test_button_ttt_pipeline.py
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


def test_ttt_chunked_equals_stepwise():
    """The hard train/inference consistency requirement."""
    from button_task.ttt_layer import TTTSequence

    torch.manual_seed(0)
    ttt = TTTSequence(dim=16, fast_hidden=32, base_lr=0.1, num_layers=1)
    streams = torch.randn(2, 5, 3, 16)

    # whole sequence at once (training-style)
    out_full, fw_full, _ = ttt(streams)

    # one timestep at a time carrying fast weights (inference-style)
    outs = []
    fw = None
    for t in range(5):
        out_t, fw, _ = ttt(streams[:, t : t + 1], prev_fast_weights=fw)
        outs.append(out_t)
    out_step = torch.cat(outs, dim=1)

    assert torch.allclose(out_full, out_step, atol=1e-5), "chunked != stepwise outputs"
    assert fw_full["step"] == fw["step"] == 5
    for li in range(1):
        for k in fw_full["layers"][li]:
            assert torch.allclose(
                fw_full["layers"][li][k], fw["layers"][li][k], atol=1e-5
            ), f"fast weight {k} diverged between chunked and stepwise"

    # same equivalence under torch.no_grad() (inference wrapper)
    ttt2 = TTTSequence(dim=16, fast_hidden=32, base_lr=0.1, num_layers=1)
    ttt2.load_state_dict(ttt.state_dict())
    with torch.no_grad():
        out_ng, fw_ng, _ = ttt2(streams)
    assert torch.allclose(out_full, out_ng, atol=1e-5), "no_grad path diverged"
    for k in fw_ng["layers"][0]:
        assert torch.allclose(fw_full["layers"][0][k], fw_ng["layers"][0][k], atol=1e-5)


def test_ttt_tbptt_same_values():
    from button_task.ttt_layer import TTTSequence

    torch.manual_seed(1)
    ttt = TTTSequence(dim=16, fast_hidden=32, base_lr=0.1, num_layers=1, tbptt_step_size=2)
    streams = torch.randn(2, 5, 3, 16)
    out, fw, _ = ttt(streams)
    # tbptt only detaches the graph; values must match a no-truncation run
    torch.manual_seed(1)
    ttt2 = TTTSequence(dim=16, fast_hidden=32, base_lr=0.1, num_layers=1)
    out2, fw2, _ = ttt2(streams)
    assert torch.allclose(out, out2, atol=1e-5)
    for k in fw["layers"][0]:
        assert torch.allclose(fw["layers"][0][k], fw2["layers"][0][k], atol=1e-5)


def test_gpt_with_ttt_and_progress():
    from button_task.ttt_layer import TTTSequence
    from models.vq_behavior_transformer.gpt import GPT, GPTConfig

    torch.manual_seed(0)
    B, T, P, D = 3, 4, 2, 16
    ttt = TTTSequence(dim=16, fast_hidden=32, base_lr=0.1, num_layers=1, progress_head=True)
    gpt = GPT(
        GPTConfig(block_size=8, input_dim=16, output_dim=8, n_patches=P,
                  n_layer=2, n_head=2, n_embd=16, dropout=0.0, cond_len=2),
        ttt_module=ttt,
    )
    x = torch.randn(B, T, P, D)
    cond = torch.randn(B, 2, D)

    logits, fw, stats = gpt(x, cond=cond, return_progress=True, debug=True)
    assert logits.shape == (B, T, 8)
    assert stats["progress"].shape == (B, T)
    assert "ttt/gate_mean" in stats

    # progress loss must NOT touch the GPT backbone
    gpt.zero_grad()
    prog_loss = (stats["progress"] - torch.zeros(B, T)).pow(2).mean()
    prog_loss.backward()
    assert gpt.transformer.wte.weight.grad is None, "progress loss leaked into GPT"
    assert gpt.transformer.h[0].attn.c_attn.weight.grad is None
    ttt_grads = [p.grad for p in gpt.ttt.parameters() if p.grad is not None]
    assert len(ttt_grads) > 0, "progress loss must train the TTT module"

    # policy loss flows through both
    gpt.zero_grad()
    logits.sum().backward()
    assert gpt.transformer.wte.weight.grad is not None

    # fast weights carry across calls
    logits2, fw2, _ = gpt(x, cond=cond, prev_fast_weights=fw)
    assert fw2["step"] == fw["step"] + T


def test_gpt_no_ttt_still_works():
    from models.vq_behavior_transformer.gpt import GPT, GPTConfig

    gpt = GPT(GPTConfig(block_size=4, input_dim=16, output_dim=8, n_patches=2,
                        n_layer=1, n_head=2, n_embd=16, dropout=0.0, cond_len=2))
    logits, fw, stats = gpt(torch.randn(2, 2, 2, 16), cond=torch.randn(2, 2, 16))
    assert logits.shape == (2, 2, 8) and fw is None and stats is None


def test_sequence_windows():
    from button_task.train_dataset import ButtonSliceDataset

    h5 = "E:/WM/turbovla/data/button_demos/random_pw6_lang_small/demos.h5"
    ds = ButtonSliceDataset(h5, action_window=12, shard_size=4, seed=0, max_episodes=2)
    shard = next(ds.iter_shards(shuffle=False))
    T, W = 4, 12
    obs_idx, act, pw, counts, valid = ds.build_sequence_windows(shard, T, W, ep_counts=None)
    assert obs_idx.shape[1] == T
    assert act.shape == (obs_idx.shape[0], T + W - 1, 7)
    assert pw.shape == (obs_idx.shape[0], 6)
    assert valid.shape == obs_idx.shape

    # boundary safety: the last window of each episode repeats the final
    # frame/action; windows never cross episode boundaries
    pos = 0
    for L in shard.episode_lengths:
        # window starting at the last frame of this episode
        w_last = obs_idx[pos + L - 1]
        assert (w_last == pos + L - 1).all()
        a_last = act[pos + L - 1]
        assert torch.allclose(a_last, shard.action[pos + L - 1].expand(T + W - 1, -1))
        pos += L

    # labels are used only for the curriculum; the password conditioning is
    # ALWAYS the full episode password, constant across every window
    ep_counts = []
    pos = 0
    for L in shard.episode_lengths:
        c = torch.zeros(L, dtype=torch.long)
        c[L // 2 :] = 2  # mid-episode: 2 presses done
        ep_counts.append(c)
        pos += L
    obs_idx2, act2, pw2, counts2, valid2 = ds.build_sequence_windows(
        shard, T, W, ep_counts=ep_counts
    )
    assert counts2.shape == (obs_idx2.shape[0], T)
    assert valid2.shape == obs_idx2.shape
    pos = 0
    for L in shard.episode_lengths:
        full = shard.pw_idx[pos]
        # every window of this episode receives the same full password
        for s in range(L):
            assert torch.equal(pw2[pos + s], full)
        pos += L


def test_bet_with_ttt_fwd_bwd():
    from button_task.ttt_layer import TTTSequence
    from models.vq_behavior_transformer.bet import BehaviorTransformer

    B, T, P, E = 4, 4, 2, 8
    W, A = 4, 3
    ttt = TTTSequence(dim=16, fast_hidden=32, base_lr=0.1, num_layers=1, progress_head=True)
    model = BehaviorTransformer(
        obs_dim=8, act_dim=3, goal_dim=0, views=1,
        vqvae_latent_dim=32, vqvae_n_embed=8, vqvae_groups=2,
        vqvae_fit_steps=None, vqvae_iters=5,
        n_patches=P, n_layer=2, n_head=2, n_embd=16,
        dropout=0.0, vqvae_encoder_loss_multiplier=1.0, vqvae_batch_size=16,
        act_scale=1.0, offset_loss_multiplier=1.0,
        obs_window_size=1, act_window_size=W,
        cond_len=3, cond_mode="seq", cond_num_symbols=2,
        gpt_block_size=8, vqvae_max_samples=64,
        ttt_module=ttt,
    )
    obs = torch.randn(B, T, P, E)
    act = torch.randn(B, T + W - 1, A) * 0.2
    pw = torch.randint(0, 3, (B, 3))
    for _ in range(3):
        model(obs, None, act, password_idx=pw)
    model.finish_epoch()
    pred, loss, loss_dict, ttt_out = model(obs, None, act, password_idx=pw)
    assert pred.shape == (B, T, W, A)
    assert loss is not None and loss.item() > 0
    assert ttt_out["next_fast_weights"]["step"] == T
    opt = model.configure_optimizers(weight_decay=0.0, learning_rate=1e-3, betas=(0.9, 0.999))
    opt.zero_grad()
    loss.backward()
    # TTT params must receive gradient from the policy loss
    ttt_grads = [p.grad for p in model._gpt_model.ttt.parameters() if p.grad is not None]
    assert len(ttt_grads) > 0
    opt.step()


def test_ttt_per_sample_update_equals_batch1():
    """Batched (B=2) inner updates must equal two independent B=1 updates."""
    from button_task.ttt_layer import TTTSequence

    torch.manual_seed(3)
    dim = 16
    x = torch.randn(2, 4, 3, dim)

    m_batch = TTTSequence(dim=dim, fast_hidden=32, base_lr=0.1, num_layers=1)
    m_a = TTTSequence(dim=dim, fast_hidden=32, base_lr=0.1, num_layers=1)
    m_b = TTTSequence(dim=dim, fast_hidden=32, base_lr=0.1, num_layers=1)
    m_a.load_state_dict(m_batch.state_dict())
    m_b.load_state_dict(m_batch.state_dict())

    with torch.no_grad():
        out_batch, fw_batch, _ = m_batch(x)
        out_a, fw_a, _ = m_a(x[0:1])
        out_b, fw_b, _ = m_b(x[1:2])

    assert torch.allclose(out_batch[0:1], out_a, atol=1e-5)
    assert torch.allclose(out_batch[1:2], out_b, atol=1e-5)
    for k in fw_a["layers"][0]:
        assert torch.allclose(fw_batch["layers"][0][k][0:1], fw_a["layers"][0][k], atol=1e-5)
        assert torch.allclose(fw_batch["layers"][0][k][1:2], fw_b["layers"][0][k], atol=1e-5)


def test_ttt_frozen_trains_forward_only():
    """--ttt frozen must still run fast-weight inner updates without error."""
    from button_task.ttt_layer import TTTSequence

    torch.manual_seed(4)
    m = TTTSequence(dim=8, fast_hidden=16, base_lr=0.1, num_layers=1)
    for p in m.parameters():
        p.requires_grad = False
    m.train()
    x = torch.randn(2, 3, 4, 8)
    out, fw, _ = m(x)
    assert out.shape == x.shape
    assert fw["step"] == 3
    # outer policy gradient must not reach frozen TTT params
    out.pow(2).mean().backward()
    assert all(p.grad is None for p in m.parameters())


def test_ttt_two_carry_chunks_detach_boundary():
    """Two consecutive carry chunks must both backward without graph errors."""
    from button_task.ttt_layer import TTTSequence

    torch.manual_seed(5)
    m = TTTSequence(dim=8, fast_hidden=16, base_lr=0.1, num_layers=1,
                    tbptt_step_size=4)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x1 = torch.randn(1, 4, 3, 8)
    x2 = torch.randn(1, 4, 3, 8)
    _, fw1, _ = m(x1)
    opt.zero_grad()
    m(x1, prev_fast_weights=None)[0].pow(2).mean().backward()
    opt.step()
    out2, fw2, _ = m(x2, prev_fast_weights=fw1)
    opt.zero_grad()
    out2.pow(2).mean().backward()
    assert fw2["step"] == 8


def test_ttt_damp_has_effect():
    """damp > 0 must pull fast weights back toward episode-initial weights."""
    from button_task.ttt_layer import TTTSequence

    torch.manual_seed(6)
    x = torch.randn(2, 5, 3, 8)
    m0 = TTTSequence(dim=8, fast_hidden=16, base_lr=0.1, num_layers=1, damp=0.0)
    m1 = TTTSequence(dim=8, fast_hidden=16, base_lr=0.1, num_layers=1, damp=0.5)
    m1.load_state_dict(m0.state_dict())
    with torch.no_grad():
        out0, fw0, _ = m0(x)
        out1, fw1, _ = m1(x)
    k = "W1"
    base = m0.layers[0].fast.W1.detach().unsqueeze(0)
    d0 = (fw0["layers"][0][k] - base).abs().mean()
    d1 = (fw1["layers"][0][k] - base).abs().mean()
    assert not torch.allclose(out0, out1)
    assert d1 < d0


def test_ttt_valid_mask_skips_update():
    """Invalid timesteps must leave their fast weights unchanged."""
    from button_task.ttt_layer import TTTSequence

    torch.manual_seed(7)
    m = TTTSequence(dim=8, fast_hidden=16, base_lr=0.1, num_layers=1)
    x = torch.randn(2, 3, 4, 8)
    valid = torch.tensor([[True, True, True], [True, False, False]])
    with torch.no_grad():
        out, fw, _ = m(x, valid=valid)
    # row 1 saw only the first update, not updates at t=1/2
    m_ref = TTTSequence(dim=8, fast_hidden=16, base_lr=0.1, num_layers=1)
    m_ref.load_state_dict(m.state_dict())
    with torch.no_grad():
        _, fw_ref, _ = m_ref(x[1:2, 0:1])
    k = "W1"
    assert torch.allclose(
        fw["layers"][0][k][1:2], fw_ref["layers"][0][k], atol=1e-5
    )


def test_gpt_per_timestep_attn_matches_stepwise():
    """per_timestep_attn=True must make a T=4 window equal four T=1 calls."""
    from models.vq_behavior_transformer.gpt import GPT, GPTConfig

    torch.manual_seed(8)
    cfg = GPTConfig(block_size=4, input_dim=8, output_dim=8, n_patches=2,
                    n_layer=2, n_head=2, n_embd=16, dropout=0.0, cond_len=2,
                    per_timestep_attn=True)
    gpt = GPT(cfg)
    x = torch.randn(1, 4, 2, 8)
    cond = torch.randn(1, 2, 8)
    with torch.no_grad():
        logits_full, _, _ = gpt(x, cond=cond)
        logits_step = []
        for t in range(4):
            lt, _, _ = gpt(x[:, t : t + 1], cond=cond)
            logits_step.append(lt)
        logits_step = torch.cat(logits_step, dim=1)
    assert torch.allclose(logits_full, logits_step, atol=1e-5)


def test_ttt_inner_update_descent_direction():
    """The fast-weight step must be W <- W - lr * grad MSE(K->V)."""
    from torch.func import functional_call
    from button_task.ttt_layer import FastMLP, inner_update

    torch.manual_seed(21)
    fast = FastMLP(4, 8, 4)
    params = {
        k: v.detach().clone().unsqueeze(0).requires_grad_(True)
        for k, v in fast.named_parameters()
    }
    k = torch.randn(1, 3, 4)
    v = torch.randn(1, 3, 4)
    lr = torch.tensor(0.1)
    new = inner_update(fast, params, k, v, lr, create_graph=True, grad_clip=0.0)

    pred = functional_call(fast, params, (k,))
    loss = torch.nn.functional.mse_loss(pred, v)
    grads = torch.autograd.grad(loss, list(params.values()))
    for name, g in zip(params.keys(), grads):
        assert torch.allclose(new[name], params[name] - lr * g, atol=1e-5)


def test_ttt_damp_chunked_equals_stepwise():
    """damp must consolidate toward episode-initial W0, so chunked T=4 and
    four T=1 carried steps must stay numerically identical."""
    from button_task.ttt_layer import TTTSequence

    torch.manual_seed(22)
    x = torch.randn(2, 4, 3, 8)
    m_full = TTTSequence(dim=8, fast_hidden=16, base_lr=0.1, num_layers=1, damp=0.5)
    m_step = TTTSequence(dim=8, fast_hidden=16, base_lr=0.1, num_layers=1, damp=0.5)
    m_step.load_state_dict(m_full.state_dict())

    with torch.no_grad():
        out_full, fw_full, _ = m_full(x)
        fw = None
        outs = []
        for t in range(4):
            out_t, fw, _ = m_step(x[:, t : t + 1], prev_fast_weights=fw)
            outs.append(out_t)
        out_step = torch.cat(outs, dim=1)

    assert torch.allclose(out_full, out_step, atol=1e-5)
    for k in fw_full["layers"][0]:
        assert torch.allclose(
            fw_full["layers"][0][k], fw["layers"][0][k], atol=1e-5
        )


def test_gpt_per_layer_ttt_matches_stepwise():
    """Per-layer TTT mode: one TTT per block, and T=4 must equal four T=1
    calls with the per-layer fast-weight list carried."""
    from button_task.ttt_layer import TTTSequence
    from models.vq_behavior_transformer.gpt import GPT, GPTConfig

    torch.manual_seed(9)
    n_layer = 2
    ttts = [
        TTTSequence(dim=16, fast_hidden=32, base_lr=0.1, num_layers=1)
        for _ in range(n_layer)
    ]
    cfg = GPTConfig(block_size=4, input_dim=8, output_dim=8, n_patches=2,
                    n_layer=n_layer, n_head=2, n_embd=16, dropout=0.0,
                    cond_len=2, per_timestep_attn=True)
    gpt = GPT(cfg, ttt_module=ttts)
    x = torch.randn(1, 4, 2, 8)
    cond = torch.randn(1, 2, 8)

    with torch.no_grad():
        logits_full, fw_full, _ = gpt(x, cond=cond)
        logits_step = []
        fw = None
        for t in range(4):
            lt, fw, _ = gpt(x[:, t : t + 1], cond=cond, prev_fast_weights=fw)
            logits_step.append(lt)
        logits_step = torch.cat(logits_step, dim=1)

    assert isinstance(fw_full, list) and len(fw_full) == n_layer
    assert torch.allclose(logits_full, logits_step, atol=1e-5)
    for li in range(n_layer):
        for k in fw_full[li]["layers"][0]:
            assert torch.allclose(
                fw_full[li]["layers"][0][k], fw[li]["layers"][0][k], atol=1e-5
            )


def main():
    check("ttt_chunked_equals_stepwise", test_ttt_chunked_equals_stepwise)
    check("ttt_tbptt_same_values", test_ttt_tbptt_same_values)
    check("gpt_with_ttt_and_progress", test_gpt_with_ttt_and_progress)
    check("gpt_no_ttt_still_works", test_gpt_no_ttt_still_works)
    check("sequence_windows", test_sequence_windows)
    check("bet_with_ttt_fwd_bwd", test_bet_with_ttt_fwd_bwd)
    check("ttt_per_sample_update_equals_batch1", test_ttt_per_sample_update_equals_batch1)
    check("ttt_inner_update_descent_direction", test_ttt_inner_update_descent_direction)
    check("ttt_frozen_trains_forward_only", test_ttt_frozen_trains_forward_only)
    check("ttt_two_carry_chunks_detach_boundary", test_ttt_two_carry_chunks_detach_boundary)
    check("ttt_damp_has_effect", test_ttt_damp_has_effect)
    check("ttt_damp_chunked_equals_stepwise", test_ttt_damp_chunked_equals_stepwise)
    check("ttt_valid_mask_skips_update", test_ttt_valid_mask_skips_update)
    check("gpt_per_timestep_attn_matches_stepwise", test_gpt_per_timestep_attn_matches_stepwise)
    check("gpt_per_layer_ttt_matches_stepwise", test_gpt_per_layer_ttt_matches_stepwise)
    print(f"\nall {len(PASS)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
