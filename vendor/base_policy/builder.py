"""Build the frozen vision-language base policy used by the TTT decoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .models.turbovla import build_turbovla


def build_base_model(
    base_checkpoint: Path,
    dinov3_path: Path,
    bert_path: Path,
    device: torch.device | None = None,
):
    """Construct the base policy and load its released checkpoint.

    The TTT checkpoints contain the action decoder and TTT layers only. This
    frozen base supplies the DINOv3, BERT, fusion, and state projection parts.
    """
    args = argparse.Namespace(
        dinov3_path=str(dinov3_path),
        bert_path=str(bert_path),
        hidden_dim=256,
        nheads=8,
        dim_feedforward=2048,
        max_text_len=256,
        text_padding_length=21,
        text_padding_length_by_instruction={},
        vla_feature_enhancer_layers=6,
        enhancer_inner_dim=1024,
        text_dropout=0.0,
        fusion_dropout=0.0,
        fusion_droppath=0.1,
        action_dim=7,
        chunk_size=12,
        state_dim=8,
        num_state_tokens=2,
        local_files_only=True,
        freeze_vision_encoder=False,
        freeze_text_encoder=True,
        dinov3_precision="bf16_autocast",
        expected_image_size=256,
        num_views=2,
        position_embedding="view",
        encode_views_separately=True,
        padding_strategy="key_padding_mask",
    )
    model = build_turbovla(args)
    checkpoint = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"base checkpoint mismatch: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
    model.to(device or torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model
