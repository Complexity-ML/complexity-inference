"""
Mu Dynamics
===========

Core implementation of Mu-guided equilibrium-seeking dynamics.

The Mu (μ) represents an equilibrium point that the system seeks.
All tensor operations use soft-clamping around this equilibrium
for numerical stability and smooth behavior.

Key Concepts:
- μ (mu): Equilibrium point, typically 0.0
- Soft clamping: tanh-based smooth bounds (differentiable)
- Velocity tracking: Rate of change monitoring
- β (beta): Adaptation rate [0, 2]

Example:
    >>> x = torch.randn(100)
    >>> x_stable = mu_clamp(x, mu=0.0, clamp_min=-10, clamp_max=10)
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional, List, Union

import torch


def soft_clamp(
    value: Union[torch.Tensor, float],
    min_val: float,
    max_val: float,
) -> Union[torch.Tensor, float]:
    """
    Soft clamping using tanh for smooth saturation.

    Unlike hard clamping (torch.clamp), this provides smooth bounds
    that are fully differentiable and avoid sudden discontinuities.

    The mapping is:
        center = (max + min) / 2
        range = (max - min) / 2
        output = tanh((x - center) / range) * range + center

    Args:
        value: Value to clamp (tensor or scalar)
        min_val: Minimum bound
        max_val: Maximum bound

    Returns:
        Soft-clamped value in [min_val, max_val]
    """
    if max_val <= min_val:
        center = (max_val + min_val) / 2
        if isinstance(value, torch.Tensor):
            return torch.full_like(value, center)
        return center

    range_val = (max_val - min_val) * 0.5
    center = (max_val + min_val) * 0.5

    if isinstance(value, torch.Tensor):
        normalized = (value - center) / range_val
        return torch.tanh(normalized) * range_val + center
    else:
        normalized = (value - center) / range_val
        return math.tanh(normalized) * range_val + center


def mu_clamp(
    value: Union[torch.Tensor, float],
    mu: float = 0.0,
    clamp_min: float = -10.0,
    clamp_max: float = 10.0,
) -> Union[torch.Tensor, float]:
    """
    Mu-clamping: Soft clamp deviation from equilibrium point.

    This is the core operation of Mu dynamics. Values are clamped
    based on their deviation from the equilibrium (μ).

    The operation:
        deviation = value - μ
        clamped_deviation = soft_clamp(deviation, clamp_min, clamp_max)
        output = μ + clamped_deviation

    Args:
        value: Value to clamp
        mu: Equilibrium point (default: 0.0)
        clamp_min: Minimum deviation from mu
        clamp_max: Maximum deviation from mu

    Returns:
        Mu-clamped value centered around equilibrium

    Example:
        >>> x = torch.tensor([100.0, -50.0, 5.0])
        >>> mu_clamp(x, mu=0.0, clamp_min=-10, clamp_max=10)
        tensor([10.0, -10.0, 4.9995])  # Extreme values smoothly bounded
    """
    deviation = value - mu
    clamped_deviation = soft_clamp(deviation, clamp_min, clamp_max)

    if isinstance(value, torch.Tensor):
        return mu + clamped_deviation
    return mu + clamped_deviation


def apply_mu_dynamics(
    tensor: torch.Tensor,
    mu: float = 0.0,
    clamp_min: float = -10.0,
    clamp_max: float = 10.0,
    enabled: bool = True,
) -> torch.Tensor:
    """
    Apply Mu dynamics to a tensor (in-place friendly).

    Args:
        tensor: Input tensor
        mu: Equilibrium point
        clamp_min: Minimum deviation
        clamp_max: Maximum deviation
        enabled: Whether to apply (pass-through if False)

    Returns:
        Mu-clamped tensor
    """
    if not enabled:
        return tensor
    return mu_clamp(tensor, mu, clamp_min, clamp_max)


@dataclass
class MuState:
    """
    Mutable state for tracking Mu dynamics over time.

    Used for adaptive control and monitoring.
    """

    current_value: float = 0.0
    target_mu: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0

    # PID error tracking
    integral_error: float = 0.0
    previous_error: float = 0.0

    # Timing
    last_update_time: float = field(default_factory=time.time)

    # History
    value_history: List[float] = field(default_factory=list)
    max_history: int = 20


class MuDynamics:
    """
    Mu Dynamics Controller.

    Implements equilibrium-seeking behavior with velocity tracking
    and PID control for stable convergence.

    Used for:
    - Adaptive batch sizing
    - Cache pressure management
    - Token velocity monitoring
    - Request priority smoothing

    Example:
        >>> dynamics = MuDynamics(mu=0.0)
        >>> for value in stream:
        ...     stable_value = dynamics.update(value)
    """

    def __init__(
        self,
        mu: float = 0.0,
        clamp_min: float = -10.0,
        clamp_max: float = 10.0,
        velocity_min: float = -10.0,
        velocity_max: float = 10.0,
        beta: float = 1.0,
        ema_alpha: float = 0.3,
        enabled: bool = True,
    ):
        """
        Initialize Mu dynamics controller.

        Args:
            mu: Equilibrium point
            clamp_min: Minimum value deviation from mu
            clamp_max: Maximum value deviation from mu
            velocity_min: Minimum velocity
            velocity_max: Maximum velocity
            beta: Adaptation rate [0, 2]
            ema_alpha: EMA smoothing factor
            enabled: Whether dynamics are active
        """
        self.mu = mu
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.velocity_min = velocity_min
        self.velocity_max = velocity_max
        self.beta = beta
        self.ema_alpha = ema_alpha
        self.enabled = enabled

        self.state = MuState(target_mu=mu)

    def update(self, new_value: float, dt: Optional[float] = None) -> float:
        """
        Update dynamics with a new observed value.

        Args:
            new_value: New observed value
            dt: Time delta (computed automatically if None)

        Returns:
            Mu-clamped value after dynamics update
        """
        if not self.enabled:
            return new_value

        current_time = time.time()
        if dt is None:
            dt = current_time - self.state.last_update_time

        if dt <= 0:
            return new_value

        # Compute velocity
        if self.state.current_value != 0:
            raw_velocity = (new_value - self.state.current_value) / dt
            self.state.velocity = soft_clamp(
                raw_velocity, self.velocity_min, self.velocity_max
            )

        # EMA smoothing
        self.state.current_value = (
            self.ema_alpha * new_value +
            (1 - self.ema_alpha) * self.state.current_value
        )

        # Track history
        self.state.value_history.append(new_value)
        if len(self.state.value_history) > self.state.max_history:
            self.state.value_history.pop(0)

        self.state.last_update_time = current_time

        # Return mu-clamped value
        return mu_clamp(
            self.state.current_value,
            self.mu,
            self.clamp_min,
            self.clamp_max,
        )

    def get_equilibrium_force(self) -> float:
        """
        Compute force towards equilibrium (spring + damping).

        Returns:
            Force magnitude
        """
        deviation = self.state.current_value - self.mu
        spring_force = -self.beta * deviation
        damping_force = -0.3 * self.state.velocity
        total_force = spring_force + damping_force
        return soft_clamp(total_force, -2.0, 2.0)

    def should_intervene(self) -> bool:
        """Check if intervention is needed for stability."""
        if not self.enabled:
            return False

        # Velocity at limits
        if abs(self.state.velocity) >= self.velocity_max * 0.9:
            return True

        # Far from equilibrium
        if abs(self.state.current_value - self.mu) > self.clamp_max * 0.8:
            return True

        return False

    def reset(self):
        """Reset dynamics state."""
        self.state = MuState(target_mu=self.mu)

    def get_stats(self) -> dict:
        """Get dynamics statistics."""
        return {
            "enabled": self.enabled,
            "mu": self.mu,
            "current_value": self.state.current_value,
            "velocity": self.state.velocity,
            "equilibrium_force": self.get_equilibrium_force(),
            "should_intervene": self.should_intervene(),
        }
