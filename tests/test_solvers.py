"""
Tests for the pluggable ODE solver backends (tovpy/solvers.py) focused on
TOV equation solving correctness, including tidal (Love) parameters.

Covers:
  - make_solver factory (string keys, passthrough, invalid key, kwargs)
  - JaxSolver (diffrax backend) -- correctness and backend agreement
  - TOV.solve() M/R/C consistency between scipy, numba, and jax backends
  - Even-parity tidal parameters (k[2], h[2]) -- validity and backend agreement
  - Odd-parity tidal parameters (j[2]) -- validity and backend agreement
  - Solver speed benchmarks for scipy, numba, and jax backends
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
    make_solver,
)
from tovpy.tov import TOV


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _jax_diffrax_available():
    try:
        import jax       # noqa: F401
        import diffrax   # noqa: F401
        return True
    except ImportError:
        return False


jax_available = pytest.mark.skipif(
    not _jax_diffrax_available(),
    reason="jax and diffrax not installed",
)


@pytest.fixture(scope="module")
def sly_eos():
    """SLy piecewise-polytrope EOS (shared across the module)."""
    return EOSPiecewisePolytropic("SLy")


@pytest.fixture(scope="module")
def central_pressure():
    """Moderate central pressure (geometric units) for a ~2 M_sun star."""
    return 1e-9


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
# JaxSolver backend (diffrax)
# ---------------------------------------------------------------------------

class TestJaxSolver:
    def test_is_ode_solver_subclass(self):
        assert issubclass(JaxSolver, ODESolver)

    @jax_available
    def test_mass_radius_scipy_agreement(self, sly_eos, central_pressure):
        """JAX (diffrax) and scipy must agree to 0.01% on M, R, C."""
        REL_TOL = 1e-4
        tov_scipy = TOV(eos=sly_eos, ode_backend="scipy")
        tov_jax   = TOV(eos=sly_eos, ode_backend="jax")

        M_s, R_s, C_s = tov_scipy.solve(central_pressure)
        M_j, R_j, C_j = tov_jax.solve(central_pressure)

        assert abs(M_s - M_j) / M_s < REL_TOL, (
            f"Mass disagreement: scipy={M_s:.6g}, jax={M_j:.6g}"
        )
        assert abs(R_s - R_j) / R_s < REL_TOL, (
            f"Radius disagreement: scipy={R_s:.6g}, jax={R_j:.6g}"
        )
        assert abs(C_s - C_j) / C_s < REL_TOL, (
            f"Compactness disagreement: scipy={C_s:.6g}, jax={C_j:.6g}"
        )


# ---------------------------------------------------------------------------
# Backend agreement on TOV equations (M, R, C)
# ---------------------------------------------------------------------------

class TestTOVBackendConsistency:
    """ScipySolver and NumbaSolver must agree to within tolerances."""

    REL_TOL = 1e-4  # 0.01% relative tolerance

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
        """Default backend must be scipy and produce a physical star."""
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
# Even-parity tidal parameters (k[2], h[2])
# ---------------------------------------------------------------------------

class TestEvenTidal:
    """Tests for the ell=2 even-parity Love/shape numbers."""

    REL_TOL = 2e-3  # 0.2% — consistent with ode_atol/rtol=1e-6

    @pytest.fixture(scope="class")
    def even_tov_scipy(self, sly_eos):
        return TOV(eos=sly_eos, leven=[2], ode_backend="scipy")

    @pytest.fixture(scope="class")
    def even_tov_numba(self, sly_eos):
        return TOV(eos=sly_eos, leven=[2], ode_backend="numba")

    def test_even_tidal_scipy_physical(self, even_tov_scipy, central_pressure):
        """k[2] and h[2] must be positive and in physically reasonable range."""
        M, R, C, k, h = even_tov_scipy.solve(central_pressure)
        assert M > 0 and R > 0
        assert k[2] > 0, f"Even Love number k[2]={k[2]} must be positive"
        assert h[2] > 0, f"Shape number h[2]={h[2]} must be positive"
        # For typical NS k2 is O(0.01-0.15); h2 is O(0.1-10)
        assert k[2] < 1.0, f"k[2]={k[2]} seems unphysically large"

    def test_even_tidal_backend_consistency(
        self, even_tov_scipy, even_tov_numba, central_pressure
    ):
        """scipy and numba must agree on k[2] and h[2]."""
        M_s, R_s, C_s, k_s, h_s = even_tov_scipy.solve(central_pressure)
        M_n, R_n, C_n, k_n, h_n = even_tov_numba.solve(central_pressure)

        assert abs(k_s[2] - k_n[2]) / abs(k_s[2]) < self.REL_TOL, (
            f"k[2] disagreement: scipy={k_s[2]:.6g}, numba={k_n[2]:.6g}"
        )
        assert abs(h_s[2] - h_n[2]) / abs(h_s[2]) < self.REL_TOL, (
            f"h[2] disagreement: scipy={h_s[2]:.6g}, numba={h_n[2]:.6g}"
        )


# ---------------------------------------------------------------------------
# Odd-parity tidal parameters (j[2])
# ---------------------------------------------------------------------------

class TestOddTidal:
    """Tests for the ell=2 odd-parity Love numbers."""

    REL_TOL = 2e-3  # 0.2% — consistent with ode_atol/rtol=1e-6

    @pytest.fixture(scope="class")
    def odd_tov_scipy(self, sly_eos):
        return TOV(eos=sly_eos, lodd=[2], ode_backend="scipy")

    @pytest.fixture(scope="class")
    def odd_tov_numba(self, sly_eos):
        return TOV(eos=sly_eos, lodd=[2], ode_backend="numba")

    def test_odd_tidal_scipy_physical(self, odd_tov_scipy, central_pressure):
        """j[2] must be nonzero and in physically reasonable magnitude range."""
        M, R, C, j = odd_tov_scipy.solve(central_pressure)
        assert M > 0 and R > 0
        assert j[2] != 0, "Odd Love number j[2] must be nonzero"
        assert abs(j[2]) < 1.0, f"|j[2]|={abs(j[2])} seems unphysically large"

    def test_odd_tidal_backend_consistency(
        self, odd_tov_scipy, odd_tov_numba, central_pressure
    ):
        """scipy and numba must agree on j[2]."""
        M_s, R_s, C_s, j_s = odd_tov_scipy.solve(central_pressure)
        M_n, R_n, C_n, j_n = odd_tov_numba.solve(central_pressure)

        assert abs(j_s[2] - j_n[2]) / abs(j_s[2]) < self.REL_TOL, (
            f"j[2] disagreement: scipy={j_s[2]:.6g}, numba={j_n[2]:.6g}"
        )


# ---------------------------------------------------------------------------
# Combined even + odd tidal parameters
# ---------------------------------------------------------------------------

class TestCombinedTidal:
    """TOV with both even and odd perturbations active simultaneously."""

    @pytest.fixture(scope="class")
    def combined_tov(self, sly_eos):
        return TOV(eos=sly_eos, leven=[2], lodd=[2], ode_backend="scipy")

    def test_combined_returns_all_outputs(self, combined_tov, central_pressure):
        """With both leven and lodd set, solve() returns (M, R, C, k, h, j)."""
        result = combined_tov.solve(central_pressure)
        assert len(result) == 6, (
            f"Expected 6-tuple (M,R,C,k,h,j), got {len(result)}-tuple"
        )
        M, R, C, k, h, j = result
        assert M > 0 and R > 0
        assert k[2] > 0
        assert h[2] > 0
        assert j[2] != 0


# ---------------------------------------------------------------------------
# Solver speed benchmark (informational -- never fails)
# ---------------------------------------------------------------------------

class TestSolverBenchmark:
    """
    Benchmark all three backends on the full TOV equations and print the
    results.  These tests always pass; they exist to report timing information.
    """

    N_WARMUP = 2
    N_BENCH = 10

    def _time_backend(self, backend_name, sly_eos, pc):
        tov = TOV(eos=sly_eos, ode_backend=backend_name)
        for _ in range(self.N_WARMUP):
            tov.solve(pc)
        t0 = time.perf_counter()
        for _ in range(self.N_BENCH):
            tov.solve(pc)
        return (time.perf_counter() - t0) / self.N_BENCH

    def test_benchmark_scipy(self, sly_eos, central_pressure):
        t = self._time_backend("scipy", sly_eos, central_pressure)
        print(f"\n[benchmark] scipy  : {t * 1000:.2f} ms/call")

    def test_benchmark_numba(self, sly_eos, central_pressure):
        t = self._time_backend("numba", sly_eos, central_pressure)
        print(f"\n[benchmark] numba  : {t * 1000:.2f} ms/call")

    @pytest.mark.skipif(not _jax_diffrax_available(), reason="jax/diffrax not installed")
    def test_benchmark_jax(self, sly_eos, central_pressure):
        t = self._time_backend("jax", sly_eos, central_pressure)
        print(f"\n[benchmark] jax    : {t * 1000:.2f} ms/call")

    @pytest.mark.skipif(not _jax_diffrax_available(), reason="jax/diffrax not installed")
    def test_benchmark_jax_sequence(self, sly_eos):
        """JAX sequence benchmark: solving for varying central pressures must be
        fast after JIT warmup (no re-tracing per call)."""
        pc_array = np.logspace(-12, -9, 20)
        tov = TOV(eos=sly_eos, ode_backend="jax")
        # Warmup: first call triggers JIT compilation
        tov.solve(pc_array[0])
        tov.solve(pc_array[-1])

        t0 = time.perf_counter()
        for pc in pc_array:
            tov.solve(pc)
        elapsed = time.perf_counter() - t0
        ms_per_solve = elapsed / len(pc_array) * 1000
        print(
            f"\n[benchmark] jax seq: {elapsed:.3f}s total, "
            f"{ms_per_solve:.2f} ms/solve ({len(pc_array)} pressures)"
        )
        # After JIT compilation, JAX solves with varying inputs should average
        # no more than 50 ms each (well below the ~800 ms re-tracing penalty).
        assert ms_per_solve < 50, (
            f"JAX sequence too slow: {ms_per_solve:.1f} ms/solve; "
            f"expected <50 ms after JIT compilation"
        )

    @pytest.mark.skipif(not _jax_diffrax_available(), reason="jax/diffrax not installed")
    def test_benchmark_jax_eos_change(self, sly_eos):
        """JAX EOS-change benchmark: switching EOS must reuse the compiled
        kernel (no recompilation) as long as the structural layout matches."""
        eos_names = ["SLy", "AP1", "AP2", "AP3", "AP4",
                     "MPA1", "WFF1", "WFF2", "ENG", "MS1"]
        pc_array = np.logspace(-12, -9, 20)

        # Warmup: compile the kernel with the first EOS
        tov_warmup = TOV(eos=sly_eos, ode_backend="jax")
        tov_warmup.solve(pc_array[0])
        tov_warmup.solve(pc_array[-1])

        # Benchmark subsequent EOS — should all hit the cached kernel
        times = {}
        for name in eos_names[1:]:
            eos = EOSPiecewisePolytropic(name)
            tov = TOV(eos=eos, ode_backend="jax")
            t0 = time.perf_counter()
            for pc in pc_array:
                tov.solve(pc)
            times[name] = time.perf_counter() - t0

        total = sum(times.values())
        ms_per_solve = total / (len(eos_names[1:]) * len(pc_array)) * 1000
        print(
            f"\n[benchmark] jax EOS change: {total:.3f}s total, "
            f"{ms_per_solve:.2f} ms/solve ({len(eos_names[1:])} EOS × "
            f"{len(pc_array)} pressures)"
        )
        for name, t in times.items():
            print(f"  {name}: {t:.3f}s ({t/len(pc_array)*1000:.1f} ms/solve)")

        # After warmup, switching EOS should not trigger recompilation.
        # 50 ms/solve is generous (actual ~5 ms); catches the ~800 ms
        # re-tracing regression.
        assert ms_per_solve < 50, (
            f"JAX EOS change too slow: {ms_per_solve:.1f} ms/solve; "
            f"expected <50 ms (compiled kernel should be reused across EOS)"
        )

    def test_benchmark_rhs_interpolation(self, sly_eos, central_pressure):
        """
        Compare the optimized RHS (pre-built tables + np.interp) against a
        reference that calls Python EOS methods directly.  The optimized path
        should be equal or faster; with Numba it is significantly faster.
        """
        tov = TOV(eos=sly_eos)
        eos = sly_eos
        pc = central_pressure
        hc = eos.PseudoEnthalpy_Of_Pressure(pc)
        h_test = hc * 0.5

        N = 10_000

        t0 = time.perf_counter()
        for _ in range(N):
            eos.Pressure_Of_PseudoEnthalpy(h_test)
            eos.EnergyDensity_Of_PseudoEnthalpy(h_test)
            eos.EnergyDensityDeriv_Of_Pressure(eos.Pressure_Of_PseudoEnthalpy(h_test))
        t_eos = (time.perf_counter() - t0) / N * 1e6

        lh = np.log(h_test)
        t0 = time.perf_counter()
        for _ in range(N):
            p = np.exp(np.interp(lh, tov._log_h, tov._log_p))
            np.exp(np.interp(lh, tov._log_h, tov._log_e))
            np.exp(np.interp(np.log(p), tov._log_p_sorted, tov._log_dedp))
        t_tables = (time.perf_counter() - t0) / N * 1e6

        print(
            f"\n[benchmark] RHS EOS evaluation:"
            f" Python methods={t_eos:.2f} us,"
            f" pre-built tables (np.interp)={t_tables:.2f} us"
        )
        MAX_SLOWDOWN_FACTOR = 3
        assert t_tables < t_eos * MAX_SLOWDOWN_FACTOR, (
            f"Table interpolation ({t_tables:.2f} us) is unexpectedly slow "
            f"vs Python EOS ({t_eos:.2f} us)"
        )
