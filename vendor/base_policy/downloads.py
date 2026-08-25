"""Download non-experiment model assets required for simulator rollout."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


BASE_REPO = "H-EmbodVis/TurboVLA"
DINO_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"
BERT_REPO = "google-bert/bert-base-uncased"


def _snapshot(repo_id: str, target: Path, patterns: list[str]) -> None:
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(target),
        allow_patterns=patterns,
        token=os.environ.get("HF_TOKEN"),
    )


def _file(repo_id: str, filename: str, target: Path) -> None:
    from huggingface_hub import hf_hub_download

    if target.exists():
        return
    cached = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=filename,
            token=os.environ.get("HF_TOKEN"),
        )
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(cached, target)
    except OSError:
        shutil.copy2(cached, target)


def ensure_base_policy_weights(
    base_checkpoint: Path,
    dinov3_path: Path,
    bert_path: Path,
) -> None:
    """Download missing base assets; trained decoder weights stay user-local."""
    _file(BASE_REPO, "checkpoints/libero/spatial.pth", base_checkpoint)
    dino_files = ("config.json", "preprocessor_config.json", "model.safetensors")
    if not all((dinov3_path / name).exists() for name in dino_files):
        _snapshot(
            DINO_REPO,
            dinov3_path,
            ["config.json", "preprocessor_config.json", "model.safetensors"],
        )
    bert_files = ("config.json", "model.safetensors", "vocab.txt")
    if not all((bert_path / name).exists() for name in bert_files):
        _snapshot(
            BERT_REPO,
            bert_path,
            [
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.txt",
            ],
        )
