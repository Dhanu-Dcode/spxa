r"""
Exactness tests for CompoundPoisson process.

Each test verifies a closed-form analytical result against a published formula
or exact mathematical theorem (Kingman 1993, Cont & Tankov 2004, Sato 1999).
These tests must never be skipped or marked xfail.
"""

import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.zoo.levy import (
    BrownianMotion,
    CompoundPoisson,
    DiscreteJump,
    DoubleExponentialJump,
    ExponentialJump,
    NormalJump,
    UniformJump,
)


@pytest.mark.exactness
def test_cp_exactness_level() -> None:
    """CompoundPoisson carries ExactnessLevel.EXACT."""
    cp = CompoundPoisson(rate=2.0, jump_dist=NormalJump(mean=0.0, std=1.0))
    assert cp.exactness == ExactnessLevel.EXACT


@pytest.mark.exactness
def test_cp_triplet_diffusion_zero() -> None:
    """CompoundPoisson is a pure-jump process with σ² = 0."""
    cp = CompoundPoisson(rate=3.5, jump_dist=NormalJump(mean=0.5, std=1.2))
    assert cp.triplet.sigma_sq == 0.0


@pytest.mark.exactness
def test_cp_triplet_total_mass_and_finite_activity() -> None:
    """
    Lévy measure total mass equals arrival rate λ < ∞ (finite activity).
    Cont & Tankov (2004), Proposition 3.4.
    """
    rate = 4.25
    cp = CompoundPoisson(rate=rate, jump_dist=NormalJump(mean=0.0, std=1.0))
    assert cp.triplet.nu.total_mass() == pytest.approx(rate)
    assert cp.triplet.nu.is_finite_activity() is True


@pytest.mark.exactness
def test_cp_triplet_drift_normal_jump() -> None:
    """
    Canonical drift b = λ · E[Y · 𝟙_{|Y|≤1}].
    For symmetric normal jumps (mean=0), b = 0.
    """
    cp = CompoundPoisson(rate=2.0, jump_dist=NormalJump(mean=0.0, std=1.0))
    assert cp.triplet.b == pytest.approx(0.0, abs=1e-12)


@pytest.mark.exactness
def test_cp_triplet_drift_exponential_jump() -> None:
    """
    For Exp(β) jumps, b = λ/β · [1 - (1+β)e^{-β}].
    """
    rate = 2.0
    beta = 1.5
    cp = CompoundPoisson(rate=rate, jump_dist=ExponentialJump(rate=beta))
    expected_b = rate * (1.0 / beta) * (1.0 - (1.0 + beta) * np.exp(-beta))
    assert cp.triplet.b == pytest.approx(expected_b, rel=1e-10)


@pytest.mark.exactness
def test_cp_triplet_drift_uniform_jump() -> None:
    """
    For Uniform(-0.5, 0.5) jumps, b = 0 by symmetry.
    For Uniform(0.0, 2.0), truncation region is [0, 1], so E[Y 1_{|Y|<=1}] = 1/4.
    """
    cp_symm = CompoundPoisson(rate=3.0, jump_dist=UniformJump(low=-0.5, high=0.5))
    assert cp_symm.triplet.b == pytest.approx(0.0, abs=1e-12)

    cp_asymm = CompoundPoisson(rate=4.0, jump_dist=UniformJump(low=0.0, high=2.0))
    # E[Y 1_{|Y|<=1}] = \int_0^1 (y / 2) dy = 1/4. Drift = 4 * 1/4 = 1.
    assert cp_asymm.triplet.b == pytest.approx(1.0, rel=1e-10)


@pytest.mark.exactness
def test_cp_triplet_drift_discrete_jump() -> None:
    """
    For discrete jumps, b = λ · ∑_{|v_k| ≤ 1} p_k v_k.
    """
    rate = 2.0
    # Values: -2.0 (|v|>1, excluded), 0.5 (included), 1.0 (included)
    values = [-2.0, 0.5, 1.0]
    probs = [0.2, 0.5, 0.3]
    cp = CompoundPoisson(rate=rate, jump_dist=DiscreteJump(values=values, probs=probs))
    expected_b = rate * (0.5 * 0.5 + 0.3 * 1.0)
    assert cp.triplet.b == pytest.approx(expected_b, rel=1e-10)


@pytest.mark.exactness
def test_cp_char_func_at_zero() -> None:
    """φ(0) = 1 for any compound Poisson process."""
    cp = CompoundPoisson(rate=2.0, jump_dist=NormalJump(mean=0.3, std=0.8))
    assert cp.char_func(0.0) == pytest.approx(1.0 + 0j)


@pytest.mark.exactness
def test_cp_char_func_modulus_leq_one() -> None:
    """|φ(u)| ≤ 1 for all real u."""
    cp = CompoundPoisson(rate=3.0, jump_dist=ExponentialJump(rate=2.0))
    u_grid = np.linspace(-10, 10, 50)
    vals = cp.char_func(u_grid)
    assert np.all(np.abs(vals) <= 1.0 + 1e-10)


@pytest.mark.exactness
def test_cp_char_func_time_additivity() -> None:
    """
    φ(u; t1 + t2) = φ(u; t1) · φ(u; t2) (independent stationary increments).
    """
    cp = CompoundPoisson(rate=1.8, jump_dist=DoubleExponentialJump(p=0.4, eta1=2.0, eta2=1.5))
    u = 1.25
    val_t1 = cp.char_func(u, t=1.0)
    val_t2 = cp.char_func(u, t=2.0)
    val_t3 = cp.char_func(u, t=3.0)
    assert val_t3 == pytest.approx(val_t1 * val_t2, rel=1e-10)


@pytest.mark.exactness
def test_cp_char_func_matches_triplet_char_exp() -> None:
    """
    Analytic characteristic function matches the Lévy–Khintchine formula
    evaluated via the triplet.
    """
    cp = CompoundPoisson(rate=2.5, jump_dist=NormalJump(mean=0.2, std=0.5))
    u = 1.0
    val_analytic = cp.char_func(u, t=1.0)
    val_triplet = cp.triplet.char_exp(u, t=1.0)
    assert val_analytic == pytest.approx(val_triplet, rel=1e-5)


@pytest.mark.exactness
def test_cp_cumulants_gaussian_jump() -> None:
    """
    κ_n(X_1) = λ · E[Y^n] for Gaussian jumps Y ~ N(μ, σ²).
      κ_1 = λμ
      κ_2 = λ(μ² + σ²)
      κ_3 = λ(μ³ + 3μσ²)
      κ_4 = λ(μ⁴ + 6μ²σ² + 3σ⁴)
    """
    rate = 2.5
    mu = 0.4
    sigma = 0.8
    cp = CompoundPoisson(rate=rate, jump_dist=NormalJump(mean=mu, std=sigma))
    kappas = cp.cumulants(order=4)

    expected_k1 = rate * mu
    expected_k2 = rate * (mu**2 + sigma**2)
    expected_k3 = rate * (mu**3 + 3 * mu * sigma**2)
    expected_k4 = rate * (mu**4 + 6 * (mu**2) * (sigma**2) + 3 * (sigma**4))

    assert kappas[1] == pytest.approx(expected_k1, rel=1e-10)
    assert kappas[2] == pytest.approx(expected_k2, rel=1e-10)
    assert kappas[3] == pytest.approx(expected_k3, rel=1e-10)
    assert kappas[4] == pytest.approx(expected_k4, rel=1e-10)


@pytest.mark.exactness
def test_cp_cumulants_exponential_jump() -> None:
    """
    κ_n(X_1) = λ · n! / β^n for Exp(β) jumps.
    """
    rate = 3.0
    beta = 2.0
    cp = CompoundPoisson(rate=rate, jump_dist=ExponentialJump(rate=beta))
    kappas = cp.cumulants(order=5)

    import math

    for n in range(1, 6):
        expected_kn = rate * math.factorial(n) / (beta**n)
        assert kappas[n] == pytest.approx(expected_kn, rel=1e-10)


@pytest.mark.exactness
def test_cp_cumulant_time_scaling() -> None:
    """κ_n(X_t) = t · κ_n(X_1) for all orders n."""
    cp = CompoundPoisson(rate=2.0, jump_dist=UniformJump(low=-1.0, high=2.0))
    t = 2.5
    for n in range(1, 5):
        k_t = cp.cumulant_exact(n, t=t)
        k_1 = cp.cumulant_exact(n, t=1.0)
        assert k_t == pytest.approx(t * k_1, rel=1e-10)


@pytest.mark.exactness
def test_cp_is_martingale_iff_zero_mean_jump() -> None:
    """
    A compound Poisson process is a martingale iff E[Y] = 0.
    Applebaum (2009), Theorem 2.4.5.
    """
    cp_zero_mean = CompoundPoisson(rate=2.0, jump_dist=NormalJump(mean=0.0, std=1.0))
    assert cp_zero_mean.properties.is_martingale is True

    cp_pos_mean = CompoundPoisson(rate=2.0, jump_dist=NormalJump(mean=0.5, std=1.0))
    assert cp_pos_mean.properties.is_martingale is False

    cp_exp = CompoundPoisson(rate=2.0, jump_dist=ExponentialJump(rate=1.0))
    assert cp_exp.properties.is_martingale is False


@pytest.mark.exactness
def test_cp_is_subordinator_iff_positive_jumps() -> None:
    """
    A compound Poisson process is a subordinator iff jump sizes are strictly positive.
    """
    cp_exp = CompoundPoisson(rate=2.0, jump_dist=ExponentialJump(rate=1.5))
    assert cp_exp.properties.is_subordinator is True

    cp_norm = CompoundPoisson(rate=2.0, jump_dist=NormalJump(mean=2.0, std=0.5))
    assert cp_norm.properties.is_subordinator is False

    cp_discrete_pos = CompoundPoisson(rate=1.0, jump_dist=DiscreteJump([1.0, 2.0, 3.5]))
    assert cp_discrete_pos.properties.is_subordinator is True

    cp_discrete_mixed = CompoundPoisson(rate=1.0, jump_dist=DiscreteJump([-1.0, 2.0]))
    assert cp_discrete_mixed.properties.is_subordinator is False


@pytest.mark.exactness
def test_cp_string_and_deterministic_initialization() -> None:
    """Convenience constructors (string names, deterministic jump scalars)."""
    cp_norm = CompoundPoisson(rate=2.0, jump_dist="normal", mean=0.1, std=0.9)
    assert isinstance(cp_norm.jump_dist, NormalJump)
    assert cp_norm.jump_dist.mean == 0.1

    cp_exp = CompoundPoisson(rate=2.0, jump_dist="exponential", jump_rate=1.5)
    assert isinstance(cp_exp.jump_dist, ExponentialJump)
    assert cp_exp.jump_dist.rate == 1.5

    cp_det = CompoundPoisson(rate=2.0, jump_dist=3.5)
    assert isinstance(cp_det.jump_dist, DiscreteJump)
    assert cp_det.cumulant_exact(1) == pytest.approx(2.0 * 3.5)


@pytest.mark.exactness
def test_cp_addition_triplet_with_bm() -> None:
    """
    (CompoundPoisson + BrownianMotion).triplet sums correctly:
      b = b_{CP} + b_{BM}
      σ² = 0 + σ²_{BM}
      ν = ν_{CP} + 0
    Sato (1999), Proposition 11.10.
    """
    cp = CompoundPoisson(rate=2.0, jump_dist=NormalJump(mean=0.0, std=1.0))
    bm = BrownianMotion(mu=0.5, sigma=1.2)
    z = cp + bm

    assert z.exactness == ExactnessLevel.EXACT
    assert z.triplet.sigma_sq == pytest.approx(1.44, rel=1e-10)
    assert z.triplet.b == pytest.approx(0.5, rel=1e-10)
    assert z.cumulants(2)[2] == pytest.approx(2.0 * 1.0 + 1.44, rel=1e-10)


@pytest.mark.exactness
def test_cp_simulate_mean() -> None:
    """Empirical mean converges to λ · T · E[Y]."""
    rng = np.random.default_rng(42)
    rate = 3.0
    jump_mean = 0.5
    cp = CompoundPoisson(rate=rate, jump_dist=NormalJump(mean=jump_mean, std=1.0))
    paths = cp.simulate(n_steps=50, n_paths=15_000, T=1.0, rng=rng)
    assert paths[:, 0].sum() == 0.0
    empirical_mean = paths[:, -1].mean()
    expected_mean = rate * jump_mean
    assert empirical_mean == pytest.approx(expected_mean, abs=0.05)


@pytest.mark.exactness
def test_cp_simulate_variance() -> None:
    """Empirical variance converges to λ · T · E[Y²]."""
    rng = np.random.default_rng(42)
    rate = 2.0
    mu = 0.2
    sigma = 0.8
    cp = CompoundPoisson(rate=rate, jump_dist=NormalJump(mean=mu, std=sigma))
    paths = cp.simulate(n_steps=50, n_paths=15_000, T=1.0, rng=rng)
    empirical_var = paths[:, -1].var()
    expected_var = rate * (mu**2 + sigma**2)
    assert empirical_var == pytest.approx(expected_var, rel=0.06)


@pytest.mark.exactness
def test_cp_subordinator_paths_non_decreasing() -> None:
    """Exponential jump compound Poisson sample paths are monotonically non-decreasing."""
    rng = np.random.default_rng(42)
    cp = CompoundPoisson(rate=3.0, jump_dist=ExponentialJump(rate=2.0))
    paths = cp.simulate(n_steps=20, n_paths=200, T=1.0, rng=rng)
    diffs = np.diff(paths, axis=1)
    assert np.all(diffs >= 0.0)


@pytest.mark.exactness
def test_cp_jump_distributions_methods_and_tails() -> None:
    """Test jump distribution densities, tails, moments, and representations."""
    # NormalJump
    nj = NormalJump(mean=0.5, std=1.0)
    assert repr(nj).startswith("NormalJump")
    assert nj.density(0.5) > 0.0
    assert 0.0 < nj.tail(1.0) < 1.0

    # ExponentialJump
    ej = ExponentialJump(rate=2.0)
    assert repr(ej).startswith("ExponentialJump")
    assert ej.density(np.array([-1.0, 1.0]))[0] == 0.0
    assert ej.density(np.array([-1.0, 1.0]))[1] > 0.0
    assert ej.tail(-0.5) == 1.0
    assert ej.tail(1.0) == pytest.approx(np.exp(-2.0))

    # DoubleExponentialJump
    dej = DoubleExponentialJump(p=0.4, eta1=2.0, eta2=1.5)
    assert repr(dej).startswith("DoubleExponentialJump")
    assert dej.density(1.0) > 0.0
    assert dej.density(-1.0) > 0.0
    assert dej.tail(0.5) > 0.0
    assert dej.tail(-0.5) > 0.0

    # UniformJump
    uj = UniformJump(low=-1.0, high=2.0)
    assert repr(uj).startswith("UniformJump")
    assert uj.density(0.0) == pytest.approx(1.0 / 3.0)
    assert uj.density(5.0) == 0.0
    assert uj.tail(3.0) == 0.0
    assert uj.tail(-2.0) == 1.0

    # DiscreteJump
    dj = DiscreteJump(values=[-1.0, 2.0], probs=[0.3, 0.7])
    assert repr(dj).startswith("DiscreteJump")
    with pytest.raises(NotImplementedError):
        dj.density(0.0)
    assert dj.tail(0.0) == pytest.approx(0.7)


@pytest.mark.exactness
def test_cp_invalid_parameters_raise() -> None:
    """Test error handling for invalid process and jump distribution inputs."""
    with pytest.raises(ValueError, match="positive"):
        CompoundPoisson(rate=-1.0)
    with pytest.raises(ValueError, match="std must be positive"):
        NormalJump(mean=0.0, std=-0.5)
    with pytest.raises(ValueError, match="rate must be positive"):
        ExponentialJump(rate=0.0)
    with pytest.raises(ValueError, match="low must be < high"):
        UniformJump(low=2.0, high=1.0)
    with pytest.raises(ValueError, match="p must be in"):
        DoubleExponentialJump(p=1.5)
    with pytest.raises(ValueError, match="values cannot be empty"):
        DiscreteJump(values=[])
    with pytest.raises(ValueError, match="Unknown jump distribution name"):
        CompoundPoisson(rate=1.0, jump_dist="unsupported_name")
