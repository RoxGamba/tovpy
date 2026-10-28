"""
Tests for the ODE backends (tovpy/solvers.py) and the TOV solve built on them.

Covers
  - make_solver with backend names, a ready solver, an unknown backend or method
  - the TOV equations, the same r.h.s. from numpy and jax.numpy
  - scipy and jax agree on M, R, C and the Love numbers, for a piecewise
    polytrope and for a tabulated EOS with linear interpolation
  - solve and solve_many agree
  - accuracy against a tight-tolerance reference at the default tolerances
  - a failed integration comes back as nan, never as a finite star
"""

import os

import numpy as np
import pytest

from tovpy.eos import EOS, EOSPiecewisePolytropic
from tovpy.solvers import JaxSolver, ODESolver, ScipySolver, make_solver
from tovpy.tov import TOV, TOVEquations


def _jax_available():
    try:
        import jax  # noqa: F401
        import diffrax  # noqa: F401

        return True
    except ImportError:
        return False


needs_jax = pytest.mark.skipif(
    not _jax_available(), reason="jax and diffrax not installed"
)
BACKENDS = ["scipy", pytest.param("jax", marks=needs_jax)]

EOS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eos")


@pytest.fixture(scope="module")
def sly():
    """SLy piecewise polytrope."""
    return EOSPiecewisePolytropic("SLy")


@pytest.fixture(scope="module")
def sfho():
    """SFHo with crust, tabulated (e, p) in cgs and interpolated linearly in
    log space, the way catalogue tables are used."""
    table = np.loadtxt(os.path.join(EOS_DIR, "SFHo_25-Sept-2017.rns"), skiprows=1)
    return EOS("tabular", name="from_ndarray", data=table[:, :2], interp_kind="linear")


PC = 1e-9  # geometric units, a ~2 Msun SLy star
PCS = np.logspace(-10, -9, 6)


def _love(result):
    """M, R, C, k2, h2, j2 from a solve with leven=lodd=[2]."""
    M, R, C, k, h, j = result
    return np.array([M, R, C, k[2], h[2], j[2]])


class TestMakeSolver:
    def test_scipy(self):
        assert isinstance(make_solver("scipy"), ScipySolver)

    @needs_jax
    def test_jax(self):
        assert isinstance(make_solver("jax"), JaxSolver)

    def test_ready_solver_used_as_is(self):
        s = ScipySolver()
        assert make_solver(s) is s

    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown ODE backend"):
            make_solver("numba")

    def test_method_forwarded(self):
        assert make_solver("scipy", method="RK45").method == "RK45"

    @needs_jax
    def test_jax_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            make_solver("jax", method="LSODA")

    def test_jax_missing_fails_at_construction(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def no_jax(name, *args, **kwargs):
            if name in ("jax", "diffrax"):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_jax)
        with pytest.raises(ImportError, match="needs jax and diffrax"):
            JaxSolver()


class TestEquations:
    @needs_jax
    def test_numpy_and_jax_agree(self, sly):
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        tov = TOV(eos=sly, leven=[2, 3], lodd=[2])
        y0, h0, _ = tov._TOV__initial_data(PC, dh_fact=-1e-3)
        dy_np = tov.equations(h0, y0, tov._tables, np)
        dy_jax = tov.equations(
            h0, jnp.asarray(y0), tuple(map(jnp.asarray, tov._tables)), jnp
        )
        np.testing.assert_allclose(np.asarray(dy_jax), dy_np, rtol=1e-12)

    def test_hashable(self):
        assert hash(TOVEquations((2,), ())) == hash(TOVEquations((2,), ()))


TIGHT = dict(ode_rtol=1e-10, ode_atol=1e-10)


class TestBackendAgreement:
    """The backends and solve/solve_many solve the same equations, so at tight
    tolerance they agree to far better than the tolerance. At the default
    tolerances they differ by about the integration error, as any two step
    sequences do."""

    @needs_jax
    @pytest.mark.parametrize("eos_name", ["sly", "sfho"])
    def test_scipy_and_jax(self, eos_name, request):
        eos = request.getfixturevalue(eos_name)
        pcs = PCS if eos_name == "sfho" else [PC]
        out = {
            b: np.array(
                [
                    _love(r)
                    for r in TOV(
                        eos=eos, leven=[2], lodd=[2], ode_backend=b, **TIGHT
                    ).solve_many(pcs)
                ]
            )
            for b in ("scipy", "jax")
        }
        np.testing.assert_allclose(out["jax"], out["scipy"], rtol=2e-6)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_solve_and_solve_many(self, backend, sfho):
        tov = TOV(eos=sfho, leven=[2], lodd=[2], ode_backend=backend, **TIGHT)
        one = np.array([_love(tov.solve(pc)) for pc in PCS])
        many = np.array([_love(r) for r in tov.solve_many(PCS)])
        np.testing.assert_allclose(many, one, rtol=1e-6)

    def test_trajectory_ends_at_surface(self, sly):
        tov = TOV(eos=sly)
        M, R, C = tov.solve(PC)
        assert tov.sol.t[-1] == 0.0
        assert tov.sol.y[tov.var["r"], -1] == R
        assert tov.sol.y[tov.var["nu"], -1] == pytest.approx(np.log(1 - 2 * M / R))


class TestAccuracy:
    """At the default tolerances, Lambda of a tabulated EOS stays within 1e-3
    of a 1e-11 reference. Lambda is what a loose tolerance gets wrong first."""

    @pytest.fixture(scope="class")
    def reference(self, sfho):
        tov = TOV(
            eos=sfho, leven=[2], ode_backend="scipy", ode_rtol=1e-11, ode_atol=1e-11
        )
        return np.array(
            [tov.Compute_Lambda(2, r[3][2], r[2]) for r in tov.solve_many(PCS)]
        )

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_lambda(self, backend, sfho, reference):
        tov = TOV(eos=sfho, leven=[2], ode_backend=backend)
        lam = np.array(
            [tov.Compute_Lambda(2, r[3][2], r[2]) for r in tov.solve_many(PCS)]
        )
        np.testing.assert_allclose(lam, reference, rtol=1e-3)


class _NeverReaches(ODESolver):
    """A backend whose integrations all stop short of the surface."""

    def integrate(
        self, rhs, args, t0, t1, y0, first_step, rtol, atol, trajectory=False
    ):
        y = np.asarray(y0, dtype=float)
        return t0, y, False, None


class TestFailure:
    def test_stopping_short_gives_nan(self, sly):
        tov = TOV(eos=sly, leven=[2], lodd=[2], ode_backend=_NeverReaches())
        assert np.all(np.isnan(_love(tov.solve(PC))))
        assert all(np.all(np.isnan(_love(r))) for r in tov.solve_many(PCS))

    @needs_jax
    def test_jax_max_steps(self, sly):
        tov = TOV(eos=sly, leven=[2], ode_backend=JaxSolver(max_steps=5))
        M, R, C, k, h = tov.solve_many([PC])[0]
        assert np.isnan(M) and np.isnan(k[2])


class TestDefaults:
    def test_tolerances(self, sly):
        tov = TOV(eos=sly)
        assert (tov.ode_rtol, tov.ode_atol) == (1e-8, 1e-8)

    def test_methods(self, sly):
        assert TOV(eos=sly).solver.method == "DOP853"

    @needs_jax
    def test_jax_method(self, sly):
        assert TOV(eos=sly, ode_backend="jax").solver.method == "Dopri5"


class TestTabularVectorised:
    """An array gives the same values as the same points one by one, below,
    inside and above the table."""

    @pytest.mark.parametrize(
        "method, lo, hi",
        [
            ("Pressure_Of_PseudoEnthalpy", -30, 1),
            ("EnergyDensity_Of_PseudoEnthalpy", -30, 1),
            ("PseudoEnthalpy_Of_Pressure", -30, -5),
            ("EnergyDensity_Of_Pressure", -30, -5),
            ("EnergyDensityDeriv_Of_Pressure", -30, -5),
        ],
    )
    def test_array_equals_scalar(self, sfho, method, lo, hi):
        f = getattr(sfho, method)
        x = np.logspace(lo, hi, 400)
        np.testing.assert_array_equal(f(x), [f(v) for v in x])
        assert np.isscalar(f(x[200])) or np.ndim(f(x[200])) == 0


class TestLoveSmallCompactness:
    """The l = 2 Love number at small compactness, where the closed form
    cancels catastrophically."""

    @staticmethod
    def _closed_form(c, y, log=np.log):
        nk = (1 - 2 * c) ** 2 * (2 + 2 * c * (y - 1) - y)
        dk = (
            2 * c * (6 - 3 * y + 3 * c * (5 * y - 8))
            + 4 * c**3 * (13 - 11 * y + c * (3 * y - 2) + 2 * c**2 * (1 + y))
            + 3 * (1 - 2 * c) ** 2 * (2 - y + 2 * c * (y - 1)) * log(1 - 2 * c)
        )
        return 8 * c**5 / 5 * nk / dk

    @pytest.mark.parametrize("y", [0.5, 1.5, 2.5])
    def test_exact(self, y):
        mp = pytest.importorskip("mpmath")
        from tovpy.tov import _k2_small_compactness

        mp.mp.dps = 50
        for c in (1e-6, 1e-4, 1e-3, 1e-2, 0.03):
            exact = float(self._closed_form(mp.mpf(c), mp.mpf(y), log=mp.log))
            assert _k2_small_compactness(c, y) == pytest.approx(exact, rel=1e-13)

    @pytest.mark.parametrize("y", [0.5, 1.5, 2.5])
    def test_joins_closed_form(self, y):
        from tovpy.tov import _k2_small_compactness

        # at c = 0.03 the closed form is still good to ~3e-10
        c = 0.03
        assert _k2_small_compactness(c, y) == pytest.approx(
            self._closed_form(c, y), rel=1e-8
        )

    @pytest.mark.parametrize("y", [0.5, 1.5, 2.5])
    def test_newtonian_limit(self, y):
        from tovpy.tov import _k2_small_compactness

        # k2 -> (2 - y) / (2 (y + 3)) with a correction linear in c
        k = _k2_small_compactness(1e-8, y)
        assert k == pytest.approx((2 - y) / (2 * (y + 3)), rel=1e-6)
