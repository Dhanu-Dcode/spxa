r"""State-Space Model abstractions and FilteredTrajectory container."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class FilteredTrajectory:
    """Trajectory of filtered latent states, covariances, and innovations."""

    times: np.ndarray
    filtered_states: np.ndarray  # Shape (n_steps, dim_state)
    filtered_covariances: np.ndarray  # Shape (n_steps, dim_state, dim_state)
    innovations: np.ndarray  # Shape (n_steps, dim_obs)
    log_likelihood: float
    state_names: Optional[List[str]] = None

    def confidence_bounds(
        self, state_idx: int = 0, z: float = 1.96
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute asymptotic confidence intervals: state +/- z * sqrt(var)."""
        means = self.filtered_states[:, state_idx]
        vars_ = self.filtered_covariances[:, state_idx, state_idx]
        stds = np.sqrt(np.maximum(1e-12, vars_))
        return means - z * stds, means + z * stds


class StateSpaceModel(abc.ABC):
    """Abstract non-linear continuous-discrete state-space model:

    State equation:       x_{k} = f(x_{k-1}, dt) + w_k,   w_k ~ N(0, Q(x_{k-1}, dt))
    Observation equation: y_{k} = h(x_{k}) + v_k,         v_k ~ N(0, R)
    """

    @property
    @abc.abstractmethod
    def dim_state(self) -> int:
        """Dimension of latent state vector x."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dim_obs(self) -> int:
        """Dimension of observation vector y."""
        raise NotImplementedError

    @abc.abstractmethod
    def transition(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Nonlinear state transition f(x, dt)."""
        raise NotImplementedError

    @abc.abstractmethod
    def transition_covariance(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Process noise covariance matrix Q(x, dt)."""
        raise NotImplementedError

    @abc.abstractmethod
    def observation(self, state: np.ndarray) -> np.ndarray:
        """Nonlinear measurement function h(x)."""
        raise NotImplementedError

    @abc.abstractmethod
    def observation_covariance(self) -> np.ndarray:
        """Measurement noise covariance matrix R."""
        raise NotImplementedError
