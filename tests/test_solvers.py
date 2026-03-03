"""
Tests for the pluggable ODE solver backends (tovpy/solvers.py).

Covers:
  - make_solver factory (string keys, passthrough of existing instance, invalid key)
  - SolverResult container
  - ScipySolver and NumbaSolver correctness against a known analytic ODE
  - JaxSolver stub raises NotImplementedError
  - TOV.solve() consistency between scipy and numba backends
  - Solver speed benchmark (printed, not asserted)
"""

import time

import numpy as np
import pytest

from tovpy.eos import EOSPiecewisePolytropic
from tovpy.solvers import (
    JaxSolver,
    NumbaSolver,
    ODESolver,
    ScipySolver,
    SolverResult,
    make_solver,
)
from tovpy.tov import TOV


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sly_eos():
    """SLy piecewise-polytrope EOS (shared across the module)."""
    return EOSPiecewisePolytropic("SLy")


@pytest.fixture(scope="module")
def central_pressure():
    """Moderate central pressure (geometric units) for a ~2 M_sun star."""
    return 1e-9


# Simple analytic ODE for unit-testing the solver loop:
#   dy/dt = -y,  y(0) = 1  =>  y(t) = exp(-t)
def _exponential_rhs(t, y):
    return -np.asarray(y)


def analytic_solution(t):
    return np.exp(-t)


# ---------------------------------------------------------------------------
# make_solver factory
# ---------------------------------------------------------------------------

class TestMakeSolver:
    def test_scipy_string(self):
        s = make_solver("scipy")
        assert isinstance(s, ScipySolver)

    def test_numba_string(self):
        s = make_solver("numba")
        assert isinstance(s, NumbaSolver)

    def test_jax_string(self):
        s = make_solver("jax")
        assert isinstance(s, JaxSolver)

    def test_passthrough_existing_instance(self):
        s = ScipySolver()
        assert make_solver(s) is s

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown ODE backend"):
            make_solver("unknown_backend")

    def test_scipy_method_kwarg(self):
        s = make_solver("scipy", method="RK45")
        assert isinstance(s, ScipySolver)
        assert s.method == "RK45"

    def test_numba_max_steps_kwarg(self):
        s = make_solver("numba", max_steps=50_000)
        assert isinstance(s, NumbaSolver)
        assert s.max_steps == 50_000


# ---------------------------------------------------------------------------
# SolverResult container
# ---------------------------------------------------------------------------

class TestSolverResult:
    def test_attributes_are_numpy_arrays(self):
        r = SolverResult([0.0, 1.0, 2.0], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert isinstance(r.t, np.ndarray)
        assert isinstance(r.y, np.ndarray)

    def test_shapes(self):
        t = [0.0, 0.5, 1.0]
        y = [[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]]
        r = SolverResult(t, y)
        assert r.t.shape == (3,)
        assert r.y.shape == (2, 3)


# ---------------------------------------------------------------------------
# ScipySolver correctness on analytic ODE
# ---------------------------------------------------------------------------

class TestScipySolver:
    @pytest.mark.parametrize("method", ["DOP853", "RK45", "RK23"])
    def test_exponential_decay(self, method):
        solver = ScipySolver(method=method)
        result = solver.solve(_exponential_rhs, (0.0, 3.0), [1.0], rtol=1e-10, atol=1e-12)
        t_end = result.t[-1]
        y_end = result.y[0, -1]
        assert abs(t_end - 3.0) < 1e-10, "solver did not reach t_end"
        assert abs(y_end - analytic_solution(3.0)) < 1e-7

    def test_result_has_t_and_y(self):
        solver = ScipySolver()
        result = solver.solve(_exponential_rhs, (0.0, 1.0), [1.0])
        assert hasattr(result, "t")
        assert hasattr(result, "y")
        assert result.t[-1] == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# NumbaSolver correctness on analytic ODE
# ---------------------------------------------------------------------------

class TestNumbaSolver:
    def test_exponential_decay(self):
        solver = NumbaSolver()
        result = solver.solve(_exponential_rhs, (0.0, 3.0), [1.0], rtol=1e-10, atol=1e-12)
        t_end = result.t[-1]
        y_end = result.y[0, -1]
        assert abs(t_end - 3.0) < 1e-10
        assert abs(y_end - analytic_solution(3.0)) < 1e-6

    def test_returns_solver_result(self):
        solver = NumbaSolver()
        result = solver.solve(_exponential_rhs, (0.0, 1.0), [1.0])
        assert isinstance(result, SolverResult)

    def test_result_has_t_and_y(self):
        solver = NumbaSolver()
        result = solver.solve(_exponential_rhs, (0.0, 1.0), [1.0])
        assert hasattr(result, "t")
        assert hasattr(result, "y")
        assert result.t[-1] == pytest.approx(1.0, abs=1e-10)

    def test_multi_component_ode(self):
        """Harmonic oscillator: d²x/dt² = -x  =>  x=cos(t), v=-sin(t)."""
        def harmonic(t, y):
            return np.array([y[1], -y[0]])

        solver = NumbaSolver()
        result = solver.solve(harmonic, (0.0, 2 * np.pi), [1.0, 0.0],
                              rtol=1e-9, atol=1e-11)
        x_end = result.y[0, -1]
        v_end = result.y[1, -1]
        assert abs(x_end - 1.0) < 1e-5, f"x(2π) = {x_end}, expected ≈ 1"
        assert abs(v_end - 0.0) < 1e-5, f"v(2π) = {v_end}, expected ≈ 0"


# ---------------------------------------------------------------------------
# JaxSolver stub
# ---------------------------------------------------------------------------

class TestJaxSolver:
    def test_raises_not_implemented(self):
        solver = JaxSolver()
        with pytest.raises(NotImplementedError):
            solver.solve(_exponential_rhs, (0.0, 1.0), [1.0])

    def test_is_ode_solver_subclass(self):
        assert issubclass(JaxSolver, ODESolver)


# ---------------------------------------------------------------------------
# Backend agreement on TOV equations
# ---------------------------------------------------------------------------

class TestTOVBackendConsistency:
    """ScipySolver and NumbaSolver must agree to within tolerances."""

    REL_TOL = 1e-4  # 0.01% relative tolerance on M, R, C

    def test_mass_radius_compactness(self, sly_eos, central_pressure):
        tov_scipy = TOV(eos=sly_eos, ode_backend="scipy")
        tov_numba = TOV(eos=sly_eos, ode_backend="numba")

        M_s, R_s, C_s = tov_scipy.solve(central_pressure)
        M_n, R_n, C_n = tov_numba.solve(central_pressure)

        assert abs(M_s - M_n) / M_s < self.REL_TOL, (
            f"Mass disagreement: scipy={M_s:.6g}, numba={M_n:.6g}"
        )
        assert abs(R_s - R_n) / R_s < self.REL_TOL, (
            f"Radius disagreement: scipy={R_s:.6g}, numba={R_n:.6g}"
        )
        assert abs(C_s - C_n) / C_s < self.REL_TOL, (
            f"Compactness disagreement: scipy={C_s:.6g}, numba={C_n:.6g}"
        )

    def test_tov_scipy_default_backend(self, sly_eos, central_pressure):
        """Default backend must be scipy."""
        tov = TOV(eos=sly_eos)
        assert isinstance(tov.solver, ScipySolver)
        M, R, C = tov.solve(central_pressure)
        assert M > 0
        assert R > 0
        assert 0 < C < 0.5

    def test_tov_custom_instance(self, sly_eos, central_pressure):
        """Passing a pre-built solver instance must work."""
        custom = ScipySolver(method="RK45")
        tov = TOV(eos=sly_eos, ode_backend=custom)
        assert tov.solver is custom
        M, R, C = tov.solve(central_pressure)
        assert M > 0


# ---------------------------------------------------------------------------
# Solver speed benchmark (informational — never fails)
# ---------------------------------------------------------------------------

class TestSolverBenchmark:
    """
    Benchmark the scipy and numba backends and print the results.
    These tests always pass; they exist purely to report timing information.

    The ``TOV`` RHS uses pre-built log-space EOS tables (sampled once at
    construction) together with ``np.interp`` for EOS lookups.  When Numba is
    available the JIT-compiled ``_interp_positive`` is used instead, giving a
    further speedup.  Both paths avoid Python EOS method calls and dict lookups
    in the hot integration loop.
    """

    N_WARMUP = 2
    N_BENCH = 10

    def _time_backend(self, backend_name, sly_eos, pc):
        tov = TOV(eos=sly_eos, ode_backend=backend_name)
        # Warm-up
        for _ in range(self.N_WARMUP):
            tov.solve(pc)
        # Timed runs
        t0 = time.perf_counter()
        for _ in range(self.N_BENCH):
            tov.solve(pc)
        elapsed = time.perf_counter() - t0
        return elapsed / self.N_BENCH

    def test_benchmark_scipy(self, sly_eos, central_pressure):
        t = self._time_backend("scipy", sly_eos, central_pressure)
        print(f"\n[benchmark] scipy  : {t * 1000:.2f} ms/call")

    def test_benchmark_numba(self, sly_eos, central_pressure):
        t = self._time_backend("numba", sly_eos, central_pressure)
        print(f"\n[benchmark] numba  : {t * 1000:.2f} ms/call")

    def test_benchmark_rhs_interpolation(self, sly_eos, central_pressure):
        """
        Compare the optimized RHS (pre-built tables + np.interp) against a
        reference that calls Python EOS methods directly.  The optimized path
        should be equal or faster; with Numba it is significantly faster.
        """
        import numpy as np

        tov = TOV(eos=sly_eos)
        eos = sly_eos

        # Sample a typical h value from the middle of the integration range
        pc = central_pressure
        hc = eos.PseudoEnthalpy_Of_Pressure(pc)
        h_test = hc * 0.5  # mid-range value

        N = 10_000

        # Reference: Python EOS method calls
        t0 = time.perf_counter()
        for _ in range(N):
            eos.Pressure_Of_PseudoEnthalpy(h_test)
            eos.EnergyDensity_Of_PseudoEnthalpy(h_test)
            eos.EnergyDensityDeriv_Of_Pressure(eos.Pressure_Of_PseudoEnthalpy(h_test))
        t_eos = (time.perf_counter() - t0) / N * 1e6

        # Optimized: np.interp on pre-built log tables
        lh = np.log(h_test)
        t0 = time.perf_counter()
        for _ in range(N):
            p = np.exp(np.interp(lh, tov._log_h, tov._log_p))
            np.exp(np.interp(lh, tov._log_h, tov._log_e))
            np.exp(np.interp(np.log(p), tov._log_p_sorted, tov._log_dedp))
        t_tables = (time.perf_counter() - t0) / N * 1e6

        print(
            f"\n[benchmark] RHS EOS evaluation:"
            f" Python methods={t_eos:.2f} µs,"
            f" pre-built tables (np.interp)={t_tables:.2f} µs"
        )
        # The pre-built table path (np.interp on log arrays) must not be more
        # than 3× slower than the Python EOS methods.  In practice both are
        # ~5 µs without Numba; with Numba the table path is 10-50× faster.
        MAX_SLOWDOWN_FACTOR = 3
        assert t_tables < t_eos * MAX_SLOWDOWN_FACTOR, (
            f"Table interpolation ({t_tables:.2f} µs) is unexpectedly slow "
            f"vs Python EOS ({t_eos:.2f} µs)"
        )

    def test_benchmark_analytic_ode_scipy(self):
        solver = ScipySolver()
        t0 = time.perf_counter()
        for _ in range(self.N_BENCH):
            solver.solve(_exponential_rhs, (0.0, 3.0), [1.0], rtol=1e-9, atol=1e-11)
        elapsed = (time.perf_counter() - t0) / self.N_BENCH
        print(f"\n[benchmark] scipy  (analytic ODE): {elapsed * 1e6:.1f} µs/call")

    def test_benchmark_analytic_ode_numba(self):
        solver = NumbaSolver()
        t0 = time.perf_counter()
        for _ in range(self.N_BENCH):
            solver.solve(_exponential_rhs, (0.0, 3.0), [1.0], rtol=1e-9, atol=1e-11)
        elapsed = (time.perf_counter() - t0) / self.N_BENCH
        print(f"\n[benchmark] numba  (analytic ODE): {elapsed * 1e6:.1f} µs/call")
