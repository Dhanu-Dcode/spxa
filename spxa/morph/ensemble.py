"""
Dynamic model ensemble and Bayesian model mixing.

Combines multiple candidate stochastic models into an adaptive ensemble:

    P_ens = ∑_{k=1}^K w_k P_k

with support for dynamic Bayesian updates based on realized observations:
    w_k(t+1) ∝ w_k(t) · exp(-η · L_k(y_{t+1}))

References:
  Hoeting, J. A. et al. (1999), Bayesian model averaging: a tutorial;
  Cesa-Bianchi, N. & Lugosi, G. (2006), Prediction, Learning, and Games.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.morph.barycenter import ModelBarycenter
from spxa.story.narrator import CompositionNode, NodeKind


class ModelEnsemble(Process):
    r"""
    An adaptive ensemble of stochastic processes.

    Parameters
    ----------
    models :
        List of candidate Process instances.
    weights :
        Initial prior weights (default uniform).
    learning_rate :
        Learning rate :math:`\eta > 0` for exponential loss updates.
    """

    def __init__(
        self,
        models: Sequence[Process],
        weights: Optional[Sequence[float]] = None,
        learning_rate: float = 1.0,
    ) -> None:
        if len(models) < 1:
            raise ValueError("Must provide at least one model")

        self.models = list(models)
        k = len(models)

        if weights is None:
            self.weights = np.ones(k, dtype=np.float64) / k
        else:
            w = np.asarray(weights, dtype=np.float64)
            self.weights = w / np.sum(w)

        self.learning_rate = float(learning_rate)

        _node = CompositionNode(
            kind=NodeKind.MORPH,
            name=f"ModelEnsemble(K={k})",
            children=[m._node for m in models],
            math_note=f"Ensemble mixture with weights {np.round(self.weights, 3).tolist()}",
            reference="Hoeting et al. (1999); Cesa-Bianchi & Lugosi (2006)",
        )

        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=_node)

    def update_weights(self, losses: Sequence[float]) -> np.ndarray:
        r"""
        Update ensemble weights using exponential loss update:

        .. math::
            w_k \leftarrow w_k \exp(-\eta L_k) / \sum_j w_j \exp(-\eta L_j)
        """
        loss_arr = np.asarray(losses, dtype=np.float64)
        if len(loss_arr) != len(self.models):
            raise ValueError("Number of loss values must match number of models")

        unnorm = self.weights * np.exp(-self.learning_rate * (loss_arr - np.min(loss_arr)))
        self.weights = unnorm / np.sum(unnorm)
        return self.weights

    def to_barycenter(self, geometry: str = "triplet") -> ModelBarycenter:
        """Convert current ensemble into a Fréchet barycenter process."""
        return ModelBarycenter(self.models, weights=self.weights, geometry=geometry)

    def _triplet(self) -> LevyTriplet:
        return self.to_barycenter()._triplet()

    def _properties(self) -> ProcessProperties:
        return self.to_barycenter()._properties()

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        mode: str = "mixture",
    ) -> np.ndarray:
        r"""
        Simulate ensemble sample paths.

        Parameters
        ----------
        mode :
            ``"mixture"``: sample model index :math:`k \sim \text{Cat}(\mathbf{w})` per path,
            ``"barycenter"``: simulate averaged barycenter process.
        """
        if rng is None:
            rng = np.random.default_rng()

        if mode == "barycenter":
            return self.to_barycenter().simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng)

        # Mixture mode: draw model indices
        k_indices = rng.choice(len(self.models), size=n_paths, p=self.weights)
        paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)

        for k_idx, model in enumerate(self.models):
            mask = k_indices == k_idx
            count = int(np.sum(mask))
            if count > 0:
                sub_paths = model.simulate(n_steps=n_steps, n_paths=count, T=T, rng=rng)
                paths[mask, :] = sub_paths

        return paths
