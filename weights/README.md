# Weights

Patch Policy 默认 encoder `dino_patch` 使用 DINOv2 ViT-S。

## 当前状态

权重已缓存在本机 torch hub 目录：

```
C:\Users\Administrator\.cache\torch\hub\checkpoints\dinov2_vits14_pretrain.pth
```

`models/encoder/dino.py` 调用 `torch.hub.load("facebookresearch/dinov2:b48308a", "dinov2_vits14")`，
会直接命中该缓存，**不需要把 .pth 复制到本目录**。

## 校验

```powershell
python scripts\check_env.py
```

检查 `5. DINOv2 ViT-S weight` 是否为 `ok`。

## 离线注意事项

如果重装系统或清理缓存导致权重丢失，可重新下载：

```powershell
python -c "import torch; torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth', model_dir='C:/Users/Administrator/.cache/torch/hub/checkpoints')"
```

或使用 Patch Policy 提供的其他 encoder 配置（需要对应权重）。
