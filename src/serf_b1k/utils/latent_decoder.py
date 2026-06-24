from __future__ import annotations

import logging

import numpy as np
import torch

from serf_b1k.mapping.models.mlp import MLP


logger = logging.getLogger("serf_b1k.latent_decoder")


def load_mlp_checkpoint(path: str, device: str, label: str):
    logger.info("[LOAD] Loading %s from %s", label, path)
    state_dict = torch.load(path, map_location="cpu")

    input_dim = state_dict["input_proj.0.weight"].shape[1]
    hidden_dim = state_dict["input_proj.0.weight"].shape[0]
    output_dim = state_dict["output_proj.weight"].shape[0]
    num_res_blocks = sum(
        1
        for key in state_dict
        if key.startswith("res_blocks.") and key.endswith(".block.1.weight")
    )

    mlp = MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_res_blocks=num_res_blocks,
    )
    mlp.load_state_dict(state_dict)
    mlp = mlp.to(device)
    mlp.eval()
    logger.info("[LOAD] %s: %s -> %s -> %s", label, input_dim, hidden_dim, output_dim)
    return mlp


def decode_latent_features(
    decoder,
    feat: np.ndarray,
    device: str,
    *,
    batch_size: int = 50_000,
) -> np.ndarray:
    import torch.nn.functional as F

    feat_tensor = torch.from_numpy(feat).to(device)
    all_decoded = []

    with torch.no_grad():
        for i in range(0, feat_tensor.shape[0], batch_size):
            batch = feat_tensor[i : i + batch_size]
            decoded = decoder(batch)
            decoded = F.normalize(decoded, p=2, dim=-1)
            all_decoded.append(decoded.cpu())

    return torch.cat(all_decoded, dim=0).numpy().astype(np.float32)
