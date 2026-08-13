"""
Behavioral cloning (BC) pretraining -- thesis section 5.9, Stage 1:
"Bootstrap encoder E_phi and policy pi_theta using expert demonstrations
(heuristic frontier-based exploration)... Behavioral cloning to initialize
policy, followed by fine-tuning with PPO."

Why this matters concretely: PPO learning active-SLAM behavior completely
from scratch has to discover "explore new corridor" via random exploration
under a reward that also punishes collisions -- with a strict safety
penalty, a barely-trained policy finds it easier to just not move than to
stumble into the reward it would get from successful exploration. BC
sidesteps this cold-start problem by directly supervising the policy
toward "what would the frontier-based heuristic do here", using the exact
same joint CNN+MLP encoder (E_phi) the RL policy will keep fine-tuning
afterward -- so the encoder is already primed to extract useful features
before a single PPO gradient step happens.

This is a small, from-scratch BC loop (collect (obs, expert_action) pairs,
maximize the policy's log-probability of the expert's action) rather than
a pretrained SB3 module, because SB3 doesn't ship BC out of the box; it
operates directly on a `stable_baselines3.PPO` model's
`policy.evaluate_actions`, so the resulting weights load straight into
`model.learn()` afterward with zero glue code.
"""

from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import PPO


def collect_demonstrations(env, expert_policy, n_episodes: int = 20, seed_start: int = 5000):
    """Rolls out `expert_policy` (e.g. FrontierBasedPolicy) and records
    every (obs, action) pair encountered -- the expert-demonstration
    dataset behavioral cloning trains against."""
    from dataclasses import replace

    patches, scalars, actions = [], [], []
    base_cfg = env.cfg
    for ep in range(n_episodes):
        env.cfg = replace(base_cfg, seed=seed_start + ep)
        obs, info = env.reset()
        if hasattr(expert_policy, "reset"):
            expert_policy.reset()
        done = False
        while not done:
            action, _ = expert_policy.predict(obs, deterministic=True)
            action = int(action)
            patches.append(obs["patch"])
            scalars.append(obs["scalars"])
            actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    env.cfg = base_cfg
    return {
        "patch": np.stack(patches).astype(np.float32),
        "scalars": np.stack(scalars).astype(np.float32),
        "action": np.array(actions, dtype=np.int64),
    }


def behavioral_clone(model: PPO, dataset: dict, n_epochs: int = 8, batch_size: int = 128,
                      lr: float = 1e-3, verbose: bool = True) -> list:
    """Supervises `model.policy` toward the expert actions in `dataset` by
    maximizing the policy distribution's log-probability of those actions
    (standard BC-as-classification). Trains in place; returns the per-epoch
    mean loss for logging/plotting.
    """
    device = model.policy.device
    n = dataset["action"].shape[0]
    patch_t = torch.as_tensor(dataset["patch"], device=device)
    scalars_t = torch.as_tensor(dataset["scalars"], device=device)
    actions_t = torch.as_tensor(dataset["action"], device=device)

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)
    losses = []

    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)
        epoch_losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            obs_batch = {"patch": patch_t[idx], "scalars": scalars_t[idx]}
            act_batch = actions_t[idx]

            values, log_prob, entropy = model.policy.evaluate_actions(obs_batch, act_batch)
            loss = -log_prob.mean() - 0.01 * entropy.mean()  # small entropy bonus keeps the policy from collapsing too hard onto the expert

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        if verbose:
            print(f"[BC] epoch {epoch + 1}/{n_epochs}  loss={mean_loss:.4f}")

    return losses


def bc_action_accuracy(model: PPO, dataset: dict) -> float:
    """Fraction of expert actions the (now-cloned) policy would pick
    greedily -- a quick sanity metric that BC actually did something,
    reported before handing off to PPO fine-tuning."""
    device = model.policy.device
    patch_t = torch.as_tensor(dataset["patch"], device=device)
    scalars_t = torch.as_tensor(dataset["scalars"], device=device)
    with torch.no_grad():
        obs_batch = {"patch": patch_t, "scalars": scalars_t}
        actions, values, log_prob = model.policy(obs_batch, deterministic=True)
    predicted = actions.cpu().numpy().flatten()
    return float(np.mean(predicted == dataset["action"]))
