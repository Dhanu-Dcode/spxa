import pathlib

THEORY = pathlib.Path("spxa/zoo/beyond/_mmpp_theory.py")
MMPP = pathlib.Path("spxa/zoo/beyond/mmpp.py")
TESTS = pathlib.Path("tests/unit/test_mmpp_spxa.py")

BLOCK = '''def _stirling2_table(order):
    """Stirling numbers of the second kind S(n, k) for 0 <= k <= n <= order."""
    s = [[0] * (order + 1) for _ in range(order + 1)]
    s[0][0] = 1
    for i in range(1, order + 1):
        for k in range(1, i + 1):
            s[i][k] = k * s[i - 1][k] + s[i - 1][k - 1]
    return s


def count_cumulants(Q, rates, horizon, order, *, initial_state=0, initial_probabilities=None):
    """Return ``{1: kappa_1, ..., order: kappa_order}`` for the count N(horizon).

    Factorial moments solve ``u_k' = u_k Q + k u_{k-1} diag(rates)``; one block
    matrix exponential gives all of them. They are converted to cumulants through
    the factorial-moment generating series (a logarithm) and Stirling numbers.
    Cancellation costs roughly ``(order - 1) * log10(1 + max(rates) * horizon)``
    digits of the 16 available, so requests that would lose more than 9 digits
    raise ``ValueError`` instead of returning noise. Use a shorter horizon or a
    lower order.
    """
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError('order must be a positive integer')
    order = int(order)
    off, exits, lam, T, p0 = _validated_parameters(
        Q, rates, horizon, initial_state, initial_probabilities, 1, 1
    )
    n = lam.size
    if T == 0:
        return {k: 0.0 for k in range(1, order + 1)}
    digits_lost = (order - 1) * math.log10(1.0 + float(lam.max()) * T)
    if digits_lost > 9.0:
        raise ValueError(
            f'cumulants up to order {order} are ill-conditioned for max(rates)*horizon = '
            f'{float(lam.max()) * T:.3g} (about {digits_lost:.0f} of 16 digits would be lost); '
            'use a shorter horizon or a lower order'
        )
    p = np.eye(n)[initial_state] if p0 is None else p0
    generator = off.copy()
    np.fill_diagonal(generator, -exits)
    big = np.zeros(((order + 1) * n, (order + 1) * n), dtype=float)
    d = np.diag(lam)
    for k in range(order + 1):
        big[k * n:(k + 1) * n, k * n:(k + 1) * n] = generator.T
        if k >= 1:
            big[k * n:(k + 1) * n, (k - 1) * n:k * n] = k * d
    start = np.zeros((order + 1) * n)
    start[:n] = p
    out = expm(big * T) @ start
    # factorial moments F_k = E[N(N-1)...(N-k+1)], F_0 = 1
    fm = [float(np.sum(out[k * n:(k + 1) * n])) for k in range(order + 1)]
    # series G(s) = sum F_k s^k / k!; factorial cumulants are k! [s^k] log G(s)
    g = [fm[k] / math.factorial(k) for k in range(order + 1)]
    h = [0.0] * (order + 1)  # h = log G, from G' = G h'
    for k in range(1, order + 1):
        acc = k * g[k]
        for j in range(1, k):
            acc -= j * h[j] * g[k - j]
        h[k] = acc / (k * g[0])
    fcum = [h[k] * math.factorial(k) for k in range(order + 1)]
    s2 = _stirling2_table(order)
    return {m: float(sum(s2[m][k] * fcum[k] for k in range(1, m + 1)))
            for m in range(1, order + 1)}
'''

METHOD = '''    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        """Cumulants kappa_1..kappa_order of N(1), via block matrix exponential.

        Accurate to roughly 1e-16 * (1 + max(rates))**(order - 1); requests that
        would lose more than 9 digits raise ValueError (see ``count_cumulants``).
        """
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("order must be a positive integer")
        return count_cumulants(
            self.Q, self.rates, 1.0, order,
            initial_state=self.initial_state,
            initial_probabilities=self.initial_probabilities,
        )

'''

TESTS_ADD = '''def test_higher_order_cumulants_match_high_precision_reference():
    # Reference: 60-digit mpmath derivatives of log E[exp(theta N_1)] at theta = 0.
    p = MarkovModulatedPoissonProcess(
        [[-1, 1], [2, -2]], [4, 10], initial_probabilities=[2 / 3, 1 / 3]
    )
    ref = {
        1: 6.0,
        2: 9.64406589932065,
        3: 21.3728566911684,
        4: 47.4672936132737,
        5: -20.005738768914,
        6: -1449.73336947646,
    }
    got = p.cumulants(order=6)
    for k, v in ref.items():
        assert got[k] == pytest.approx(v, rel=1e-9)


def test_constant_rate_all_cumulants_equal_poisson():
    p = MarkovModulatedPoissonProcess([[-3, 3], [2, -2]], [5, 5])
    got = p.cumulants(order=8)
    assert all(got[k] == pytest.approx(5.0, rel=1e-8) for k in range(1, 9))


def test_ill_conditioned_cumulants_raise():
    p = MarkovModulatedPoissonProcess([[-1, 1], [2, -2]], [400, 1000])
    with pytest.raises(ValueError, match="ill-conditioned"):
        p.cumulants(order=6)
    assert set(p.cumulants(order=2)) == {1, 2}


def test_cumulants_invalid_order():
    p = MarkovModulatedPoissonProcess([[-1, 1], [2, -2]], [4, 10])
    for bad in (0, -1, True):
        with pytest.raises(ValueError):
            p.cumulants(order=bad)
'''

th = THEORY.read_text(encoding="utf-8")
if "count_cumulants" in th:
    raise SystemExit("already applied")
assert "import numpy as np\n" in th
th = th.replace("import numpy as np\n", "import math\n\nimport numpy as np\n", 1)
THEORY.write_text(th.rstrip("\n") + "\n\n\n" + BLOCK, encoding="utf-8")

mm = MMPP.read_text(encoding="utf-8")
a = mm.index("    def _moment_propagation_cumulants")
b = mm.index("    def simulate_events(")
mm = mm[:a] + METHOD + mm[b:]
old = "from ._mmpp_theory import count_moments\n"
assert old in mm, "import line not found in mmpp.py"
mm = mm.replace(old, "from ._mmpp_theory import count_cumulants, count_moments\n", 1)
MMPP.write_text(mm, encoding="utf-8")

te = TESTS.read_text(encoding="utf-8")
TESTS.write_text(te.rstrip("\n") + "\n\n\n" + TESTS_ADD, encoding="utf-8")
print("done")
