import os

import torch
import einops
import torch.nn as nn
from torchvision import transforms

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

_HUB_REPO = "facebookresearch/dinov2:b48308a"
_LOCAL_HUB_DIR = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")


def _load_base(name):
    # Prefer the already-cached checkout so no network write is needed
    # (the weights README documents this machine's cache location).
    if os.path.isdir(_LOCAL_HUB_DIR):
        return torch.hub.load(_LOCAL_HUB_DIR, name, source="local")
    return torch.hub.load(_HUB_REPO, name)


class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key, output_dim=None, postprocess=None, n_patches=256):
        super().__init__()
        print("Encoder feature_key:", feature_key)
        self.name = name
        self.base_model = _load_base(name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        self.output_dim = self.emb_dim # for compatibility
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

        self.postprocess = postprocess
        if postprocess is not None:
            if postprocess == 'avg_pool':
                self.latent_ndim = 1

        self.normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.normalization = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def forward(self, x):
        # Accept arbitrary number of leading dimensions before (C, H, W)
        # and preserve them on return.
        # Example: input shape (...prefix, C, H, W)
        assert x.max() <= 1.0 and x.min() >= 0, "expect 0..1 range"
        x = self.normalization(x)

        prefix_shape = x.shape[:-3]
        c, h, w = x.shape[-3:]

        # Collapse all leading dims into a single batch dimension for the base model
        prod_prefix = 1
        for d in prefix_shape:
            prod_prefix *= d
        x = x.reshape(prod_prefix, c, h, w)

        emb = self.base_model.forward_features(x)[self.feature_key]
        emb = emb.reshape(*prefix_shape, *emb.shape[1:])

        if self.postprocess == 'avg_pool':
            emb = torch.mean(emb, dim=-2)  # (...prefix, E)

        if self.latent_ndim == 1:
            emb = emb.unsqueeze(len(prefix_shape))

        return emb
