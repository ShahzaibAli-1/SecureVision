"""
SecureVision – Simulated Encoder
==================================
Since the real CLIP / EVA weights are multi-GB downloads we use a lightweight
*simulated* encoder that reproduces all the mathematical properties needed for
the three-stage pipeline: it maps (B, 3, H, W) image tensors → (B, D) feature
vectors with the same algebra as the real ViT encoders.

For the real system replace `SimulatedEncoder` with a CLIP / EVA wrapper.
"""

import torch
import torch.nn as nn
import numpy as np
from securevision.config import IMAGE_SIZE, ENCODER_DIM, NUM_PATCHES, SEED


class SimulatedEncoder(nn.Module):
    """
    Lightweight MLP encoder that maps flattened image patches to a feature
    vector.  Weights are fixed (frozen) unless explicitly unfrozen for
    backdoor fine-tuning (Stage 3).
    """

    def __init__(self, dim: int = ENCODER_DIM, seed: int = SEED):
        super().__init__()
        torch.manual_seed(seed)
        in_dim = 3 * IMAGE_SIZE * IMAGE_SIZE

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, dim),
        )
        # Initialise weights deterministically
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised feature vectors."""
        feat = self.proj(x)
        return nn.functional.normalize(feat, dim=-1)


def build_encoder(frozen: bool = True) -> SimulatedEncoder:
    """Return a SimulatedEncoder; freeze weights by default (Stage 1 & 2)."""
    enc = SimulatedEncoder()
    for p in enc.parameters():
        p.requires_grad_(not frozen)
    return enc


def generate_images(n: int, seed: int = SEED) -> torch.Tensor:
    """Generate *n* random images in [0, 1] — stand-in for a shadow dataset."""
    rng = np.random.default_rng(seed)
    imgs = rng.random((n, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    return torch.from_numpy(imgs)


def generate_target_image(seed: int = 7) -> torch.Tensor:
    """Generate a fixed target image x_tar (single image, shape [1,3,H,W])."""
    rng = np.random.default_rng(seed)
    img = rng.random((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    return torch.from_numpy(img)
