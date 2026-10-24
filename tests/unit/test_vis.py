"""
Unit tests for visualization module spxa.vis.

Tests Lévy density inspection and plotting, 3D implied volatility surface generation
and smile comparison, and composition DAG ASCII, Mermaid, DOT, and Matplotlib plotting.
"""

import os
import tempfile
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from spxa import (
    BrownianMotion,
    CGMY,
    CompoundPoisson,
    GammaProcess,
    HestonProcess,
    KouJumpDiffusion,
    MertonJumpDiffusion,
    NormalJump,
    VarianceGamma,
)
from spxa.core.exceptions import ExactnessError
from spxa.vis import (
    compute_iv_surface,
    dag_to_ascii,
    dag_to_dot,
    dag_to_mermaid,
    estimate_blumenthal_getoor_index,
    inspect_levy_triplet,
    plot_dag,
    plot_iv_surface,
    plot_levy_comparison,
    plot_levy_density,
    plot_smile_comparison,
    plot_term_structure,
)


class TestLevyInspector:
    """Tests for spxa.vis.levy_inspector."""

    def test_inspect_finite_activity_merton(self):
        merton = MertonJumpDiffusion(rate=1.5, jump_mean=-0.05, jump_std=0.1)
        res = inspect_levy_triplet(merton)

        assert res.process_name == "MertonJumpDiffusion"
        assert res.is_finite_activity is True
        assert np.isclose(res.total_jump_rate, 1.5)
        # Brownian component present => infinite variation
        assert res.has_gaussian_component is True
        assert res.is_finite_variation is False
        assert res.blumenthal_getoor_index == 0.0
        assert res.has_jump_component is True

        summary = res.summary()
        assert "Lévy Triplet Diagnostic" in summary
        assert "Finite Activity" in summary
        assert "MertonJumpDiffusion" in summary

    def test_inspect_finite_activity_kou(self):
        kou = KouJumpDiffusion(rate=2.0)
        res = inspect_levy_triplet(kou)

        assert res.is_finite_activity is True
        assert np.isclose(res.total_jump_rate, 2.0)
        assert res.has_gaussian_component is True
        assert res.is_finite_variation is False
        assert res.blumenthal_getoor_index == 0.0

    def test_inspect_infinite_activity_vg(self):
        vg = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.05)
        res = inspect_levy_triplet(vg)

        assert res.is_finite_activity is False
        assert np.isinf(res.total_jump_rate)
        # Pure jump VG has σ² = 0 and finite variation
        assert res.has_gaussian_component is False
        assert res.is_finite_variation is True
        assert res.blumenthal_getoor_index == 0.0
        assert res.singularity_order == 1.0

    def test_inspect_infinite_activity_cgmy(self):
        cgmy = CGMY(C=1.0, G=2.0, M=3.0, Y=0.6)
        res = inspect_levy_triplet(cgmy)

        assert res.is_finite_activity is False
        assert np.isinf(res.total_jump_rate)
        assert res.has_gaussian_component is False
        # Y = 0.6 < 1.0 => finite variation
        assert res.is_finite_variation is True
        assert np.isclose(res.blumenthal_getoor_index, 0.6)
        assert np.isclose(res.singularity_order, 1.6)

    def test_inspect_pure_brownian(self):
        bm = BrownianMotion(mu=0.05, sigma=0.2)
        res = inspect_levy_triplet(bm)

        assert res.has_gaussian_component is True
        assert res.is_finite_variation is False
        assert res.blumenthal_getoor_index == 0.0

    def test_inspect_non_levy_raises(self):
        heston = HestonProcess(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, s0=100.0)
        with pytest.raises(ExactnessError):
            inspect_levy_triplet(heston)

    def test_estimate_blumenthal_getoor_index(self):
        cgmy = CGMY(C=1.0, G=2.0, M=3.0, Y=0.4)
        beta, s = estimate_blumenthal_getoor_index(cgmy)
        assert np.isclose(beta, 0.4)
        assert np.isclose(s, 1.4)

    def test_plot_levy_density_save(self):
        vg = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.05)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fig, axes = plot_levy_density(vg, n_points=100, save_path=tmp_path)
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 1000
            assert len(axes) == 2
            plt.close(fig)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_plot_levy_comparison_save(self):
        m = MertonJumpDiffusion()
        k = KouJumpDiffusion()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fig, axes = plot_levy_comparison(
                {"Merton": m, "Kou": k}, n_points=80, save_path=tmp_path
            )
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 1000
            assert len(axes) == 2
            plt.close(fig)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestSurface:
    """Tests for spxa.vis.surface."""

    def test_compute_iv_surface_merton(self):
        merton = MertonJumpDiffusion()
        strikes = np.linspace(90, 110, 5)
        maturities = np.array([0.5, 1.0, 1.5])

        Kg, Tg, IVg = compute_iv_surface(merton, strikes, maturities, S0=100.0)

        assert Kg.shape == (3, 5)
        assert Tg.shape == (3, 5)
        assert IVg.shape == (3, 5)
        assert np.all(np.isfinite(IVg))
        assert np.all(IVg > 0.05)
        assert np.all(IVg < 1.0)

    def test_compute_iv_surface_heston(self):
        heston = HestonProcess(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, s0=100.0)
        strikes = np.linspace(90, 110, 5)
        maturities = np.array([0.5, 1.0])

        Kg, Tg, IVg = compute_iv_surface(heston, strikes, maturities, S0=100.0)
        assert Kg.shape == (2, 5)
        assert np.all(np.isfinite(IVg))
        assert np.all(IVg > 0.01)

    def test_plot_iv_surface(self):
        merton = MertonJumpDiffusion()
        strikes = np.linspace(90, 110, 5)
        maturities = np.array([0.5, 1.0])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fig, ax = plot_iv_surface(
                merton, strikes=strikes, maturities=maturities, save_path=tmp_path
            )
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 1000
            plt.close(fig)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_plot_smile_comparison(self):
        m = MertonJumpDiffusion()
        k = KouJumpDiffusion()
        strikes = np.linspace(90, 110, 7)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fig, ax = plot_smile_comparison(
                {"Merton": m, "Kou": k}, strikes=strikes, T=1.0, save_path=tmp_path
            )
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 1000
            plt.close(fig)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_plot_term_structure(self):
        m = MertonJumpDiffusion()
        maturities = np.linspace(0.2, 1.5, 5)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fig, ax = plot_term_structure(
                m, maturities=maturities, strike=100.0, save_path=tmp_path
            )
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 1000
            plt.close(fig)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_invalid_process_raises(self):
        class Dummy:
            pass

        with pytest.raises(ValueError, match="does not implement"):
            compute_iv_surface(Dummy(), [100.0], [1.0])


class TestStoryDAG:
    """Tests for spxa.vis.story_dag."""

    def test_dag_to_ascii_single(self):
        bm = BrownianMotion(mu=0.05, sigma=0.2)
        ascii_tree = dag_to_ascii(bm)

        assert "[Primitive]" in ascii_tree
        assert "BrownianMotion" in ascii_tree
        assert "Triplet:" in ascii_tree

    def test_dag_to_ascii_composed(self):
        bm = BrownianMotion(mu=0.05, sigma=0.2)
        cp = CompoundPoisson(rate=1.0, jump_dist=NormalJump(mean=0.0, std=0.1))
        p = 2.0 * (bm + cp)

        ascii_tree = dag_to_ascii(p)
        assert "[Scaling]" in ascii_tree
        assert "[Addition]" in ascii_tree
        assert "[Primitive] BrownianMotion" in ascii_tree
        assert "[Primitive] CompoundPoisson" in ascii_tree
        assert "├──" in ascii_tree or "└──" in ascii_tree

    def test_dag_to_mermaid(self):
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        gam = GammaProcess(a=1.0, b=2.0)
        vg = bm @ gam

        mermaid_str = dag_to_mermaid(vg)
        assert "graph TB" in mermaid_str
        assert "Subordination" in mermaid_str
        assert "-->" in mermaid_str
        assert "classDef" in mermaid_str

    def test_dag_to_dot(self):
        bm = BrownianMotion(mu=0.0, sigma=1.0)
        cp = CompoundPoisson(rate=1.0, jump_dist=NormalJump(mean=0.0, std=0.1))
        sum_proc = bm + cp

        dot_str = dag_to_dot(sum_proc)
        assert "digraph CompositionDAG" in dot_str
        assert "Addition" in dot_str
        assert "->" in dot_str

    def test_plot_dag(self):
        bm = BrownianMotion(mu=0.05, sigma=0.2)
        cp = CompoundPoisson(rate=1.0, jump_dist=NormalJump(mean=0.0, std=0.1))
        p = 1.5 * (bm + cp)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fig, ax = plot_dag(p, save_path=tmp_path)
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 1000
            plt.close(fig)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_extract_node_invalid_type(self):
        with pytest.raises(TypeError, match="Expected a Process instance or CompositionNode"):
            dag_to_ascii(12345)
