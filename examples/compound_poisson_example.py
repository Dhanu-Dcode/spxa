"""
Compound Poisson process demonstration in spxa.

Run with: python examples/compound_poisson_example.py
"""

import numpy as np

from spxa.zoo.levy import (
    BrownianMotion,
    CompoundPoisson,
    DiscreteJump,
    DoubleExponentialJump,
    ExponentialJump,
    NormalJump,
    UniformJump,
)

print("=" * 65)
print("spxa — Compound Poisson Process & Jump-Diffusion Demo")
print("=" * 65)

# 1. Normal Jump Compound Poisson
cp_norm = CompoundPoisson(rate=3.0, jump_dist=NormalJump(mean=0.2, std=0.8))
print(f"\n1. Gaussian Jumps: {cp_norm}")
print(f"   Exactness      : {cp_norm.exactness.name}")
print(f"   Triplet drift b: {cp_norm.triplet.b:.4f}")
print(f"   Triplet sigma² : {cp_norm.triplet.sigma_sq:.4f}")
print(f"   Total jump mass: {cp_norm.triplet.nu.total_mass():.4f}")
print(f"   Cumulants κ_1..4: {cp_norm.cumulants(4)}")

# 2. Subordinator via Exponential Jumps
cp_exp = CompoundPoisson(rate=2.0, jump_dist=ExponentialJump(rate=1.5))
print(f"\n2. Exponential Jumps (Subordinator): {cp_exp}")
print(f"   Is subordinator: {cp_exp.properties.is_subordinator}")
print(f"   Is martingale  : {cp_exp.properties.is_martingale}")
print(f"   Exact mean     : {cp_exp.cumulant_exact(1):.4f}")
print(f"   Exact variance : {cp_exp.cumulant_exact(2):.4f}")

# 3. Double-Exponential (Kou) and Uniform jumps
cp_kou = CompoundPoisson(
    rate=2.5, jump_dist=DoubleExponentialJump(p=0.4, eta1=5.0, eta2=3.0)
)
print(f"\n3. Double Exponential Jumps: {cp_kou}")
print(f"   Cumulants κ_1..3: {cp_kou.cumulants(3)}")

# 4. Algebraic Composition: Jump-Diffusion Z = BM + CompoundPoisson
bm = BrownianMotion(mu=0.05, sigma=0.2)
z = bm + cp_norm
print(f"\n4. Algebraic Composition (Merton-style Jump-Diffusion):")
print(f"   Process        : {z}")
print(f"   Z exactness    : {z.exactness.name}")
print(f"   Z triplet b    : {z.triplet.b:.4f}")
print(f"   Z triplet σ²   : {z.triplet.sigma_sq:.4f}")

# 5. Monte Carlo Simulation Check
rng = np.random.default_rng(42)
t = 1.0
paths = z.simulate(n_steps=100, n_paths=10_000, T=t, rng=rng)
emp_mean = paths[:, -1].mean()
emp_var = paths[:, -1].var()
theo_mean = z.cumulants(2)[1] * t
theo_var = z.cumulants(2)[2] * t

print(f"\n5. Monte Carlo Verification (10,000 paths at T={t}):")
print(f"   Empirical mean : {emp_mean:.4f}  (Theory: {theo_mean:.4f})")
print(f"   Empirical var  : {emp_var:.4f}  (Theory: {theo_var:.4f})")

# 6. Composition Story Trace
print("\n6. Composition Trace:")
print(z.__story__())
