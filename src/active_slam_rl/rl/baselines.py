"""
Baseline policies (thesis section 8.5) used purely for comparison against
the learned PPO policy -- none of these are trained, they're heuristics
that consume the same observation the RL policy would (patch + scalars).

  * RandomWalkPolicy      -- uniform random action every step.
  * FrontierBasedPolicy   -- greedily heads toward the nearest boundary
                              between "known free" and "unknown" cells in
                              its local patch (mimics a ROS-navigation-
                              stack-style frontier explorer).
  * NextBestViewPolicy    -- picks the action whose resulting local patch
                              (approximated one step ahead using current
                              heading) has the highest map uncertainty --
                              i.e. greedy one-step information gain, with
                              no notion of loop closure or drift at all.

These intentionally do NOT use registration quality q_t or loop-closure
saliency ell_t for decision-making (matching the thesis's framing that
these are the reactive, non-uncertainty-aware baselines the RL policy is
compared against).
"""

from __future__ import annotations

import numpy as np


class RandomWalkPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=True):
        return self.action_space.sample(), None


class FrontierBasedPolicy:
    """Greedy frontier-seeking using only the local belief patch: turn
    toward whichever half of the patch (left/right) has more "unknown"
    (p close to 0.5) cells, otherwise push forward; dwell periodically to
    refresh the local map."""

    def __init__(self, action_space, dwell_every: int = 12):
        self.action_space = action_space
        self.dwell_every = dwell_every
        self._t = 0

    def predict(self, obs, deterministic=True):
        self._t += 1
        patch = obs["patch"] if "patch" in obs else obs[0]["patch"]
        belief = patch[0] if patch.ndim == 3 else patch
        h, w = belief.shape
        unknown = np.abs(belief - 0.5) < 0.15
        left_unknown = unknown[:, : w // 2].sum()
        right_unknown = unknown[:, w // 2:].sum()

        if self._t % self.dwell_every == 0:
            return 5, None  # periodic dwell-and-scan to refresh loop-closure database
        if left_unknown > right_unknown * 1.2:
            return 3, None  # yaw left toward more unknown space
        if right_unknown > left_unknown * 1.2:
            return 4, None  # yaw right
        return 1, None  # push forward into the frontier

    def reset(self):
        self._t = 0


class NextBestViewPolicy:
    """One-step information-gain heuristic: prefer the action whose
    forward-facing half of the patch currently holds the most uncertainty
    (variance), falling back to a turn if forward is already resolved."""

    def __init__(self, action_space, dwell_every: int = 15):
        self.action_space = action_space
        self.dwell_every = dwell_every
        self._t = 0

    def predict(self, obs, deterministic=True):
        self._t += 1
        patch = obs["patch"] if "patch" in obs else obs[0]["patch"]
        uncertainty = patch[1] if patch.ndim == 3 else patch
        h, w = uncertainty.shape
        forward_region = uncertainty[h // 4: 3 * h // 4, w // 2:]
        left_region = uncertainty[:, : w // 2]
        right_region = uncertainty[:, w // 2:]

        if self._t % self.dwell_every == 0:
            return 5, None
        scores = {1: forward_region.mean(), 3: left_region.mean(), 4: right_region.mean()}
        best_action = max(scores, key=scores.get)
        return best_action, None

    def reset(self):
        self._t = 0
