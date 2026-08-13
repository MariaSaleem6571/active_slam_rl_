"""
Multi-objective reward (thesis section 5.7):

    r_t = w_cov * Delta V_t^unc + w_cons * Delta I_t
          - w_safe * c_prox,t + w_loop * b_lc,t

  * Delta V_t^unc : reduction in map entropy this step (uncertainty resolved)
  * Delta I_t     : information gain from a validated loop closure this step
                    (reduction in accumulated pose-covariance trace)
  * c_prox,t      : proximity-to-obstacle penalty (collision risk)
  * b_lc,t        : flat bonus on a validated loop closure

Change-detection reward shaping (feedback loop D) is folded in as an extra
term that locally rewards resolving high-innovation (eta_t^v) regions,
exactly as described in section 5.4/5.8.

BETA-DECAY EXPLORE/EXPLOIT CYCLING
-----------------------------------
On top of the static weights above, an exponentially-decaying factor

    beta(t) = max(beta_min, beta_initial * exp(-decay_rate * t))

(t = steps since the *last validated loop closure*, not total episode
time) scales exploration reward down and loop-closure-seeking reward up
the longer the vehicle goes without closing a loop, then resets to
beta_initial the moment a closure validates -- automatically cycling the
policy's incentive between "go explore" (beta high, just after a
closure) and "go find a loop closure" (beta low, it's been a while)
without any hand-coded mode switch. `sim_env.py` owns the `t` clock (it
resets it on `loop_closure_validated`); this module only consumes the
resulting `beta` value.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardWeights:
    w_cov: float = 400.0    # entropy_delta is a small per-voxel quantity; scaled up to be comparable to other terms
    w_cons: float = 2.0
    w_safe: float = 4.0
    w_loop: float = 5.0
    w_change: float = 50.0  # feedback loop D: reward for resolving change-flagged voxels
    w_loiter: float = 0.15  # penalty per consecutive near-stationary step (anti reward-hacking)
    beta_initial: float = 1.0    # explore/exploit weight right after a loop closure (or episode start)
    beta_decay_rate: float = 0.02  # per step since the last validated closure
    beta_min: float = 0.15       # floor -- exploration reward never fully vanishes


@dataclass
class RewardBreakdown:
    coverage_term: float
    consistency_term: float
    safety_term: float
    loop_bonus_term: float
    change_term: float
    loiter_term: float
    beta: float
    total: float


def compute_beta(steps_since_last_closure: int, local_unknown_fraction: float = 0.0,
                  weights: RewardWeights = RewardWeights()) -> float:
    """beta = max(time-decayed value, a geometry-driven floor).

    time_component = max(beta_min, beta_initial * exp(-decay_rate * t))
        decays with elapsed time since the last validated closure, as
        before.

    geometry_floor = beta_min + local_unknown_fraction * (beta_initial - beta_min)
        `local_unknown_fraction` in [0, 1] is how much of the vehicle's
        immediate surroundings (the same local map crop the state encoder
        sees) is still unresolved. In a wide, largely-unexplored area this
        stays high, which keeps beta from decaying below what the local
        geometry still warrants -- exploration credit doesn't vanish just
        because the clock ran out, if there's obviously still a lot of
        unexplored space right here. In a narrow, already-mostly-mapped
        corridor this is low, so beta falls back to (or below) the pure
        time-decay behavior, letting the transition to loop-closure-
        seeking happen sooner. Taking the max of the two, rather than
        e.g. averaging them, is deliberate: either signal alone is enough
        to justify staying in "explore" mode; only when *both* the clock
        has run out *and* the local area is well-mapped does beta actually
        fall through to the floor.
    """
    import math
    time_component = weights.beta_initial * math.exp(-weights.beta_decay_rate * steps_since_last_closure)
    geometry_floor = weights.beta_min + local_unknown_fraction * (weights.beta_initial - weights.beta_min)
    beta = max(weights.beta_min, time_component, geometry_floor)
    return min(beta, weights.beta_initial)


def compute_reward(entropy_delta: float, info_gain: float, proximity_cost: float,
                    loop_closure_validated: bool, change_voxels_resolved: float,
                    stationary_streak: int = 0, beta: float = 1.0,
                    weights: RewardWeights = RewardWeights()) -> RewardBreakdown:
    # beta high (just closed a loop) -> full exploration credit.
    # beta low (long time since a closure) -> exploration credit fades,
    # and resolving change-flagged regions (which is how new loop-closure
    # candidates actually get found and confirmed) is worth more instead.
    coverage_term = weights.w_cov * entropy_delta * beta
    consistency_term = weights.w_cons * info_gain
    safety_term = -weights.w_safe * proximity_cost
    loop_bonus_term = weights.w_loop * (1.0 if loop_closure_validated else 0.0)
    urgency = 1.0 - beta
    change_term = weights.w_change * change_voxels_resolved * (1.0 + urgency)
    # Grows with consecutive near-stationary steps (dwell/revisit-in-place),
    # so that sitting still stops being a free lunch even before its other
    # per-step reward sources decay to zero -- a second, independent
    # safeguard against the loitering exploit alongside the loop-closure
    # gating in sim_env.py.
    loiter_term = -weights.w_loiter * max(0, stationary_streak - 3)

    total = (coverage_term + consistency_term + safety_term + loop_bonus_term
             + change_term + loiter_term)
    return RewardBreakdown(
        coverage_term=coverage_term,
        consistency_term=consistency_term,
        safety_term=safety_term,
        loop_bonus_term=loop_bonus_term,
        change_term=change_term,
        loiter_term=loiter_term,
        beta=beta,
        total=total,
    )


class AdaptiveDecayController:
    """Self-tunes beta_decay_rate from experience during training, rather
    than leaving it a fixed hand-picked constant forever.

    PPO's gradients never touch the reward function -- reward is just a
    scalar number the environment hands back, not a differentiable
    function of any network's parameters -- so decay_rate genuinely can't
    be "learned" the same way the policy's weights are. What it *can* be
    is adaptive: this class tracks a running estimate of how many steps
    actually elapse between validated loop closures, and sets decay_rate
    so beta's decay timescale matches what's actually being observed.
    If closures are happening quickly in practice, decay can speed up
    without hurting exploration; if they're rare, decay slows down so the
    vehicle isn't pushed toward loop-closure-seeking behavior that rarely
    pays off.

    This is deliberately a single, persistent object shared across
    episodes within one training run (see sim_env.py -- it's created once
    in ActiveSlamEnv.__init__, not recreated on every reset()), since the
    whole point is accumulating a running statistic *across* episodes.
    """

    def __init__(self, initial_decay_rate: float = 0.02, smoothing: float = 0.05,
                 min_rate: float = 0.002, max_rate: float = 0.2):
        self.decay_rate = initial_decay_rate
        self.smoothing = smoothing
        self.min_rate = min_rate
        self.max_rate = max_rate
        # Seed the running interval estimate from the initial rate, so the
        # very first update() nudges gently rather than jumping wildly.
        self._ema_interval = 1.0 / max(initial_decay_rate, 1e-6)
        self.n_updates = 0

    def update(self, steps_since_last_closure_at_validation: int) -> float:
        """Call once, at the moment a closure validates, with how many
        steps elapsed since the previous validated closure. Returns the
        (possibly updated) decay_rate."""
        interval = max(1.0, float(steps_since_last_closure_at_validation))
        self._ema_interval = (1 - self.smoothing) * self._ema_interval + self.smoothing * interval
        target_rate = 1.0 / self._ema_interval
        self.decay_rate = float(min(self.max_rate, max(self.min_rate, target_rate)))
        self.n_updates += 1
        return self.decay_rate
