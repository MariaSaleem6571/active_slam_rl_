"""
State encoder E_phi (thesis section 5.5, feedback loops A/E/F/G).

    s_t = E_phi( local_map_patch, uncertainty_patch, change_mask_patch,
                 q_t, ell_t, tr(Sigma_t), mission_constraints )

Architecture:
  * A small 2D CNN ingests a stack of 3 co-registered local patches
    (occupancy belief, voxel uncertainty, change mask) cropped around the
    current pose -- this is the "local volumetric patches" + "uncertainty
    fields" + change-detection input from the thesis (loops B, E).
  * A small MLP ingests the scalar signals: registration quality q_t (loop
    A), loop-closure saliency ell_t (loop F), trace of pose covariance, and
    mission constraints (battery/time remaining).
  * The two branches are concatenated and projected to a d=128 latent s_t.

This module is trained jointly with the PPO policy (loop G: policy
gradients flow back through E_phi), which is why it is implemented as a
`torch.nn.Module` usable as a Stable-Baselines3 custom features extractor
(see rl/policy.py) rather than a frozen, hand-designed feature vector.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PatchCNN(nn.Module):
    """Small conv stack over the (belief, uncertainty, change-mask) patch
    stack. Kept intentionally small: patches are small local crops, not
    full images, and training runs on CPU by default."""

    def __init__(self, in_channels: int = 3, patch_size: int = 32, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> patch_size/2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> patch_size/4
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.proj = nn.Linear(32 * 4 * 4, out_dim)

    def forward(self, x):
        h = self.net(x)
        h = h.flatten(start_dim=1)
        return torch.relu(self.proj(h))


class ScalarMLP(nn.Module):
    """Fuses registration quality q_t, loop-closure saliency ell_t, pose
    covariance trace, and mission constraints (battery/time remaining)."""

    def __init__(self, in_dim: int, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class StateEncoder(nn.Module):
    """E_phi: full fusion encoder producing s_t in R^d."""

    def __init__(self, patch_channels: int = 3, patch_size: int = 32,
                 n_scalars: int = 5, latent_dim: int = 128):
        super().__init__()
        self.patch_cnn = PatchCNN(in_channels=patch_channels, patch_size=patch_size, out_dim=64)
        self.scalar_mlp = ScalarMLP(in_dim=n_scalars, out_dim=32)
        self.fuse = nn.Sequential(
            nn.Linear(64 + 32, 96),
            nn.ReLU(),
            nn.Linear(96, latent_dim),
            nn.ReLU(),
        )
        self.latent_dim = latent_dim

    def forward(self, patch: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        patch_feat = self.patch_cnn(patch)
        scalar_feat = self.scalar_mlp(scalars)
        fused = torch.cat([patch_feat, scalar_feat], dim=-1)
        return self.fuse(fused)


class SB3StateEncoderExtractor(BaseFeaturesExtractor):
    """Adapter so `StateEncoder` can be dropped straight into Stable-
    Baselines3's PPO as `policy_kwargs={"features_extractor_class": ...}`.
    The Gym observation space is a Dict with "patch" (C,H,W) and "scalars"
    (n,) keys -- see env/sim_env.py's observation_space definition, which
    is built to match this exactly.
    """

    def __init__(self, observation_space: gym.spaces.Dict, latent_dim: int = 128):
        super().__init__(observation_space, features_dim=latent_dim)
        patch_shape = observation_space["patch"].shape       # (C, H, W)
        scalar_shape = observation_space["scalars"].shape     # (n,)
        self.encoder = StateEncoder(
            patch_channels=patch_shape[0],
            patch_size=patch_shape[1],
            n_scalars=scalar_shape[0],
            latent_dim=latent_dim,
        )

    def forward(self, observations) -> torch.Tensor:
        return self.encoder(observations["patch"], observations["scalars"])
