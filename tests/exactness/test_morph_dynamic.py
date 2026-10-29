"""
Tests for dynamic schedules, state feedback, and stochastic model morphing.
"""

import math
import numpy as np

from spxa.morph.dynamic_morph import (
    CosineSchedule,
    DynamicMorphedProcess,
    LinearSchedule,
    SigmoidalSchedule,
    SmoothstepSchedule,
    StateThresholdSchedule,
)
from spxa.zoo.levy.brownian import BrownianMotion


def test_deterministic_schedules():
    lin = LinearSchedule(0.0, 1.0)
    assert lin(0.0) == 0.0
    assert lin(0.5) == 0.5
    assert lin(1.0) == 1.0

    cos = CosineSchedule(0.0, 1.0)
    assert cos(0.0) == 0.0
    assert math.isclose(cos(0.5), 0.5, rel_tol=1e-9)
    assert math.isclose(cos(1.0), 1.0, rel_tol=1e-9)

    step = SmoothstepSchedule(0.0, 1.0, order=2)
    assert step(0.0) == 0.0
    assert math.isclose(step(0.5), 0.5, rel_tol=1e-9)
    assert math.isclose(step(1.0), 1.0, rel_tol=1e-9)

    sig = SigmoidalSchedule(t_mid=0.5, rate=10.0)
    assert math.isclose(sig(0.5), 0.5, rel_tol=1e-9)
    assert sig(0.0) < 0.1
    assert sig(1.0) > 0.9


def test_state_threshold_schedule():
    thresh = StateThresholdSchedule(x_crit=100.0, delta=5.0)
    assert math.isclose(thresh(0.0, 100.0), 0.5, rel_tol=1e-9)
    assert thresh(0.0, 80.0) < 0.05
    assert thresh(0.0, 120.0) > 0.95


def test_dynamic_morphed_process_simulation():
    bm0 = BrownianMotion(mu=0.0, sigma=0.1)
    bm1 = BrownianMotion(mu=0.5, sigma=0.4)

    sched = CosineSchedule(0.0, 1.0)
    proc = DynamicMorphedProcess(bm0, bm1, schedule=sched, x0=10.0)

    rng = np.random.default_rng(42)
    paths, alpha_paths = proc.simulate(
        n_steps=40, n_paths=30, T=1.0, rng=rng, return_alpha_paths=True
    )

    assert paths.shape == (30, 41)
    assert alpha_paths.shape == (30, 41)
    assert np.all(paths[:, 0] == 10.0)
    assert math.isclose(alpha_paths[0, 0], 0.0, abs_tol=1e-7)
    assert math.isclose(alpha_paths[0, -1], 1.0, abs_tol=1e-7)


def test_stochastic_alpha_boundedness():
    bm0 = BrownianMotion(mu=0.0, sigma=0.1)
    bm1 = BrownianMotion(mu=0.2, sigma=0.3)

    sched = LinearSchedule(0.0, 1.0)
    proc = DynamicMorphedProcess(
        bm0,
        bm1,
        schedule=sched,
        x0=1.0,
        stochastic_alpha=True,
        kappa_alpha=3.0,
        alpha_bar=0.5,
        sigma_alpha=0.3,
    )

    rng = np.random.default_rng(123)
    paths, alpha_paths = proc.simulate(
        n_steps=50, n_paths=40, T=1.0, rng=rng, return_alpha_paths=True
    )

    assert np.all(alpha_paths >= 0.0)
    assert np.all(alpha_paths <= 1.0)
