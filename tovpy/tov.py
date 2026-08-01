"""
Copyright (C) 2024 Sebastiano Bernuzzi and others

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from scipy.special import factorial2, gamma, hyp2f1, poch
from math import comb, prod

from .solvers import JaxSolver, make_solver

# ---------------------------------------------------------------------------
# Optional Numba JIT — transparent no-op fallback if not installed
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit

    _NUMBA_AVAILABLE = True
except ImportError:

    def _njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def decorator(fn):
            return fn

        return decorator

    _NUMBA_AVAILABLE = False


@_njit(cache=True)
def _interp_positive(val, x_table, y_table):
    """
    Log-log linear interpolation on a strictly positive, monotonically
    increasing *x_table* (stored as log values).  Both *val* and all table
    entries must be > 0.
    """
    lv = np.log(val)
    lo, hi = 0, len(x_table) - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if x_table[mid] <= lv:
            lo = mid
        else:
            hi = mid
    t = (lv - x_table[lo]) / (x_table[hi] - x_table[lo])
    return np.exp(y_table[lo] + t * (y_table[hi] - y_table[lo]))


class TOV(object):
    """Class to solve the Tolman-Oppenheimer-Volkov stellar structure
    equations together with even/odd parity stationary bartropic perturbations.

    References
    ----------
    Lindblom, Astrophys. J. 398, 569 (1992)
    Damour & Nagar, Phys. Rev. D 80, 084035 (2009)

    Works in geometric units.

    Reference codes:

    * https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_neutron_star_t_o_v_8c_source.html
    * https://bitbucket.org/bernuzzi/tov/src/master/TOVL.m
    * https://lscsoft.docs.ligo.org/bilby/_modules/bilby/gw/eos/tov_solver.html

    Parameters
    ----------
    eos : EOS
        Equation of state instance.
    leven : list of int
        Multipole indices for even-parity perturbations (Love numbers k, h).
        Values must be > 1; values <= 1 are silently dropped.
    lodd : list of int
        Multipole indices for odd-parity perturbations (Love numbers j).
        Values must be > 1; values <= 1 are silently dropped.
    dhfact : float
        Initial ODE step factor (must be negative, default ``-1e-12``).
    ode_method : str
        Integration method forwarded to ``ScipySolver`` / ``solve_ivp``
        (e.g. ``'DOP853'``, ``'RK45'``).  Ignored for ``'numba'`` and
        ``'jax'`` backends.
    ode_atol : float
        Absolute ODE tolerance (default ``1e-6``).
    ode_rtol : float
        Relative ODE tolerance (default ``1e-6``).
    ode_backend : str or ODESolver
        ODE solver backend.  Accepts:

        * ``'scipy'`` (default) — wraps :func:`scipy.integrate.solve_ivp`;
          ``ode_method`` selects the algorithm.
        * ``'numba'`` — pure-NumPy adaptive Dormand-Prince RK45; independent
          of scipy; structured for future Numba JIT once the EOS layer is
          numba-compatible.
        * ``'jax'`` — JAX/diffrax Dopri5 with a fully XLA-JIT-compiled native
          RHS; ~3× faster than scipy after one-time JIT compilation warmup;
          requires ``pip install "jax[cpu]" diffrax``.
        * A pre-instantiated :class:`~tovpy.solvers.ODESolver` instance
          (e.g. ``make_solver('scipy', method='RK45')``).
    eos_table_points : int
        Number of points to pre-sample the EOS for fast interpolation (default 2000).

    Examples
    --------
    Basic usage (default scipy backend)::

        tov = TOV(eos=eos)
        M, R, C = tov.solve(pc)

    With tidal parameters::

        tov = TOV(eos=eos, leven=[2], lodd=[2])
        M, R, C, k, h, j = tov.solve(pc)

    Alternative backends::

        tov_numba = TOV(eos=eos, ode_backend='numba')
        tov_jax   = TOV(eos=eos, ode_backend='jax')
        tov_rk45  = TOV(eos=eos, ode_backend=make_solver('scipy', method='RK45'))
    """

    def __init__(
        self,
        eos=None,
        leven=[],
        lodd=[],
        dhfact=-1e-12,
        ode_method="DOP853",
        ode_atol=1e-6,
        ode_rtol=1e-6,
        ode_backend="scipy",
        eos_table_points=2000,
    ):

        if not eos:
            raise ValueError("Must provide a EOS")
        self.eos = eos

        leven = np.array(leven)
        lodd = np.array(lodd)

        self.leven = leven[leven > 1]
        self.lodd = lodd[lodd > 1]

        var = self.__buildvars()
        self.nvar = len(var)
        self.var = dict(zip(var, range(self.nvar)))
        self.ivar = {v: k for k, v in self.var.items()}

        if dhfact > 0.0:
            raise ValueError("ODE timestep must be negative")
        self.dhfact = dhfact
        self.ode_method = ode_method
        self.ode_atol = ode_atol
        self.ode_rtol = ode_rtol

        if ode_backend == "scipy":
            self.solver = make_solver("scipy", method=ode_method)
        else:
            self.solver = make_solver(ode_backend)

        self._i_r = self.var["r"]
        self._i_m = self.var["m"]
        self._i_nu = self.var["nu"]
        self._even_idx = [
            (self.var["H{}".format(l)], self.var["dH{}".format(l)]) for l in self.leven
        ]
        self._odd_idx = [
            (self.var["Psi{}".format(l)], self.var["dPsi{}".format(l)])
            for l in self.lodd
        ]

        self._build_eos_tables(n_points=eos_table_points)

        self._update_even = (
            self._apply_even_perts
            if len(self.leven)
            else lambda dy, y, m, r, p, e, dedp, dr_dh, dnu_dr: None
        )
        self._update_odd = (
            self._apply_odd_perts
            if len(self.lodd)
            else lambda dy, y, m, r, p, e, dedp, dr_dh: None
        )

    def __buildvars(self):
        """
        List of varnames
        """
        v = ["r", "m", "nu"]
        for l in self.leven:
            v.append("H{}".format(l))
            v.append("dH{}".format(l))
        for l in self.lodd:
            v.append("Psi{}".format(l))
            v.append("dPsi{}".format(l))
        return v

    def _build_eos_tables(self, n_points=2000):
        """Pre-sample the EOS once at construction and store log-space 1-D arrays
        for use in the fast RHS interpolation path.

        Tables stored (all 1-D, contiguous, float64):
          _log_h        : log(h) values (sorted ascending)
          _log_p        : log(p) values (same ordering as h)
          _log_e        : log(e) values
          _log_p_sorted : log(p) sorted ascending (key for dedp lookup)
          _log_dedp     : log(dedp) values, sorted by ascending log(p)

        Also binds ``self._eos_eval(h) -> (p, e, dedp)`` once, choosing the
        Numba JIT path or the ``np.interp`` path depending on availability.
        """
        # Sample in h, the integration variable, across every value a solve can
        # visit -- so the RHS only ever *interpolates*.
        #
        # This used to sample in p across a fixed 1e-19..1e-8 box (it read
        # `self.eos.p_min`/`p_max`, names no EOS class defines, so the lookup
        # raised AttributeError every time and the box was never a choice
        # anyone made). Whatever box you pick, the solve still runs out to the
        # surface, h -> 0, below any finite floor: measured on a catalogue EOS,
        # every solve asked for h up to 8.5 decades under the grid. Those
        # queries were answered by continuing one straight line in log-log
        # through the grid's two lowest points -- exact only where that region
        # happens to be a true power law, and different between the
        # scipy/numba and jax paths, which is why they could disagree about
        # Mmax by 9%.
        #
        # Bounds, both known in advance rather than guessed:
        #  * top -- a solve starts at h(pc), and pc can reach past the table
        #    into the ultra-relativistic extension, so keep the old 1e-8
        #    headroom.
        #  * bottom -- `solve` integrates down to `|dhfact| * h(pc)` exactly,
        #    never below, so the lowest reachable h is `|dhfact|` times the
        #    smallest h any pc could give. Two further decades for safety.
        #
        # Every sample is taken by *calling the EOS*, so its own low- and
        # high-density power-law continuations get tabulated faithfully
        # instead of being re-derived from the grid's edge slope.
        p_lo, p_hi = 1e-19, 1e-8
        try:
            p_lo = min(p_lo, float(self.eos.p_min) * 1.001)
            p_hi = max(p_hi, float(self.eos.p_max) * 0.999)
        except AttributeError:
            # Analytic EOSs (polytropes) are unbounded and define neither.
            pass

        h_hi = float(self.eos.PseudoEnthalpy_Of_Pressure(p_hi))
        h_lo = (
            float(self.eos.PseudoEnthalpy_Of_Pressure(p_lo)) * abs(self.dhfact) * 1e-2
        )

        h_arr = np.logspace(np.log10(h_lo), np.log10(h_hi), n_points)
        p_arr = np.array([self.eos.Pressure_Of_PseudoEnthalpy(h) for h in h_arr])
        e_arr = np.array([self.eos.EnergyDensity_Of_PseudoEnthalpy(h) for h in h_arr])
        dedp_arr = np.array([self.eos.EnergyDensityDeriv_Of_Pressure(p) for p in p_arr])

        valid = (h_arr > 0) & (e_arr > 0) & (dedp_arr > 0) & (p_arr > 0)
        h_arr = h_arr[valid]
        p_arr = p_arr[valid]
        e_arr = e_arr[valid]
        dedp_arr = dedp_arr[valid]

        idx = np.argsort(h_arr)
        h_arr = h_arr[idx]
        p_arr = p_arr[idx]
        e_arr = e_arr[idx]
        dedp_arr = dedp_arr[idx]

        self._log_h = np.ascontiguousarray(np.log(h_arr))
        self._log_p = np.ascontiguousarray(np.log(p_arr))
        self._log_e = np.ascontiguousarray(np.log(e_arr))

        # dedp keyed by pressure
        p_idx = np.argsort(p_arr)
        self._log_p_sorted = np.ascontiguousarray(np.log(p_arr[p_idx]))
        self._log_dedp = np.ascontiguousarray(np.log(dedp_arr[p_idx]))

        log_h = self._log_h
        log_p = self._log_p
        log_e = self._log_e
        log_p_sorted = self._log_p_sorted
        log_dedp = self._log_dedp

        if _NUMBA_AVAILABLE:

            def _eos_eval(h):
                p = _interp_positive(h, log_h, log_p)
                e = _interp_positive(h, log_h, log_e)
                dedp = _interp_positive(p, log_p_sorted, log_dedp)
                return p, e, dedp

        else:

            def _eos_eval(h):
                lh = np.log(h)
                p = np.exp(np.interp(lh, log_h, log_p))
                e = np.exp(np.interp(lh, log_h, log_e))
                dedp = np.exp(np.interp(np.log(p), log_p_sorted, log_dedp))
                return p, e, dedp

        self._eos_eval = _eos_eval

    def _get_jax_rhs(self):
        """Return a native JAX RHS, building and caching it on first call.

        Returns ``None`` if JAX is not installed.  The returned function has
        signature ``jax_rhs(h, y, eos_tables)`` where *eos_tables* is a tuple
        of five ``jnp`` arrays ``(log_h, log_p, log_e, log_p_sorted,
        log_dedp)``.  Passing the EOS data as an argument (rather than closing
        over it) allows a single JIT-compiled kernel to be reused across
        different EOS instances that share the same structural layout and table
        size — see :meth:`_get_jax_structural_key`.
        """
        if not hasattr(self, "_jax_rhs_cached"):
            self._jax_rhs_cached = self._build_jax_rhs()
        return self._jax_rhs_cached

    def _get_jax_eos_tables(self):
        """Return the pre-built EOS tables as a tuple of ``jnp`` arrays.

        Returns ``None`` if JAX is not installed.
        """
        try:
            import jax.numpy as jnp
        except ImportError:
            return None
        return (
            jnp.array(self._log_h),
            jnp.array(self._log_p),
            jnp.array(self._log_e),
            jnp.array(self._log_p_sorted),
            jnp.array(self._log_dedp),
        )

    def _get_jax_structural_key(self):
        """Return a hashable key identifying the JAX computational structure.

        Two TOV instances with the same structural key produce identical XLA
        computation graphs, allowing the JIT-compiled kernel to be shared even
        when the underlying EOS differs.

        The key includes the state-vector dimension, the perturbation layout,
        and the EOS table size (JAX requires matching array shapes).
        """
        return (
            self.nvar,
            tuple(int(l) for l in self.leven),
            tuple(int(l) for l in self.lodd),
            len(self._log_h),
        )

    def _build_jax_rhs(self):
        """Build a native JAX / XLA RHS using ``jnp`` operations throughout.

        The returned callable ``jax_rhs(h, y, eos_tables) -> jnp.ndarray``
        mirrors :meth:`__tov_rhs` but replaces all NumPy operations with their
        JAX equivalents so that diffrax can JIT-compile the entire integration
        loop in a single XLA kernel.

        EOS tables are passed as an argument (not closed over) so that TOV
        instances with different EOS but the same structural layout share one
        compiled kernel.

        * EOS interpolation uses ``jnp.interp`` on the caller-supplied tables.
        * Perturbation loops are *statically unrolled* at Python level
          (compile-time constants) — JAX sees no Python control flow.
        * Array updates use ``dy.at[i].set(v)`` (JAX immutable semantics).

        Returns ``None`` if JAX is not installed.
        """
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            return None

        jax.config.update("jax_enable_x64", True)

        i_r, i_m, i_nu = self._i_r, self._i_m, self._i_nu
        nvar = self.nvar
        even_idx = list(self._even_idx)
        leven_l = list(self.leven)
        odd_idx = list(self._odd_idx)
        lodd_l = list(self.lodd)
        pi = float(np.pi)

        def jax_rhs(h, y, eos_tables):
            log_h, log_p, log_e, log_p_sorted, log_dedp = eos_tables

            # EOS evaluation via log-space interpolation
            lh = jnp.log(h)
            p = jnp.exp(jnp.interp(lh, log_h, log_p))
            e = jnp.exp(jnp.interp(lh, log_h, log_e))
            dedp = jnp.exp(jnp.interp(jnp.log(p), log_p_sorted, log_dedp))

            r = y[i_r]
            m = y[i_m]
            dr_dh = -r * (r - 2.0 * m) / (m + 4.0 * pi * r**3 * p)
            dm_dh = 4.0 * pi * r**2 * e * dr_dh
            dnu_dr = 2.0 * (m + 4.0 * pi * r**3 * p) / (r * (r - 2.0 * m))
            dy = jnp.zeros(nvar, dtype=jnp.float64)
            dy = dy.at[i_r].set(dr_dh)
            dy = dy.at[i_m].set(dm_dh)
            dy = dy.at[i_nu].set(dnu_dr * dr_dh)

            # Even perturbations (Python loop unrolled at JAX trace time)
            for l_val, (iH, idH) in zip(leven_l, even_idx):
                Lam = float(l_val * (l_val + 1))
                div_r = 1.0 / r
                div_r2 = div_r**2
                exp_lam = 1.0 / (1.0 - 2.0 * m * div_r)
                C1_e = 2.0 / r + exp_lam * (2.0 * m * div_r2 + 4.0 * pi * r * (p - e))
                C0_e = -(dnu_dr**2) + exp_lam * (
                    -Lam * div_r2 + 4.0 * pi * (5.0 * e + 9.0 * p + (e + p) * dedp)
                )
                H = y[iH]
                dH = y[idH]
                dy = dy.at[iH].set(dH * dr_dh)
                dy = dy.at[idH].set(-(C0_e * H + C1_e * dH) * dr_dh)

            # Odd perturbations (Python loop unrolled at JAX trace time)
            for l_val, (iPsi, idPsi) in zip(lodd_l, odd_idx):
                Lam = float(l_val * (l_val + 1))
                div_r = 1.0 / r
                div_r2 = div_r**2
                div_r3 = div_r * div_r2
                r3 = r * r * r
                exp_lam = 1.0 / (1.0 - 2.0 * m * div_r)
                C1_o = exp_lam * (2.0 * m + 4.0 * pi * r3 * (p - e)) * div_r2
                C0_o = exp_lam * (-Lam * div_r2 + 6.0 * m * div_r3 - 4.0 * pi * (e - p))
                Psi = y[iPsi]
                dPsi = y[idPsi]
                dy = dy.at[iPsi].set(dPsi * dr_dh)
                dy = dy.at[idPsi].set(-(C0_o * Psi + C1_o * dPsi) * dr_dh)

            return dy

        return jax_rhs

    def __pert_even(self, ell, m, r, p, e, dedp, dnu_dr=[]):
        """
        Eq.(27-29) of Damour & Nagar, Phys. Rev. D 80, 084035 (2009)
        https://arxiv.org/abs/0906.0096
        Note only C0 depends on ell: return an array of values
        """
        r2 = r**2
        r3 = r * r2
        div_r = 1.0 / r
        div_r2 = div_r**2
        exp_lam = 1.0 / (1.0 - 2.0 * m * div_r)
        if not dnu_dr:
            dnu2 = (2.0 * (m + 4.0 * np.pi * r3 * p) / (r * (r - 2.0 * m))) ** 2
        else:
            dnu2 = dnu_dr**2
        C1 = 2.0 / r + exp_lam * (2 * m * div_r2 + 4 * np.pi * r * (p - e))
        C0 = np.zeros(max(ell) + 1)
        for l in ell:
            Lam = l * (l + 1)
            C0[l] = -dnu2
            C0[l] += exp_lam * (
                -Lam * div_r2 + 4 * np.pi * (5 * e + 9 * p + (e + p) * dedp)
            )
        return C1, C0

    def __pert_odd(self, ell, m, r, p, e, dedp):
        """
        Eq.(31) of Damour & Nagar, Phys. Rev. D 80, 084035 (2009)
        https://arxiv.org/abs/0906.0096
        Note only C0 depends on ell: return an array of values
        """
        r2 = r**2
        r3 = r * r2
        div_r = 1.0 / r
        div_r2 = div_r**2
        div_r3 = div_r * div_r2
        exp_lam = 1.0 / (1.0 - 2.0 * m * div_r)
        C1 = exp_lam * (2 * m + 4 * np.pi * r3 * (p - e)) * div_r2
        C0 = np.zeros(max(ell) + 1)
        for l in ell:
            Lam = l * (l + 1)
            C0[l] = exp_lam * (-Lam * div_r2 + 6 * m * div_r3 - 4 * np.pi * (e - p))
        return C1, C0

    def _apply_even_perts(self, dy, y, m, r, p, e, dedp, dr_dh, dnu_dr):
        """Apply even-parity perturbation equations to derivative vector *dy*."""
        C1, C0 = self.__pert_even(self.leven, m, r, p, e, dedp, dnu_dr)
        for l, (iH, idH) in zip(self.leven, self._even_idx):
            H = y[iH]
            dH = y[idH]
            dy[iH] = dH * dr_dh
            dy[idH] = -(C0[l] * H + C1 * dH) * dr_dh

    def _apply_odd_perts(self, dy, y, m, r, p, e, dedp, dr_dh):
        """Apply odd-parity perturbation equations to derivative vector *dy*."""
        C1, C0 = self.__pert_odd(self.lodd, m, r, p, e, dedp)
        for l, (iPsi, idPsi) in zip(self.lodd, self._odd_idx):
            Psi = y[iPsi]
            dPsi = y[idPsi]
            dy[iPsi] = dPsi * dr_dh
            dy[idPsi] = -(C0[l] * Psi + C1 * dPsi) * dr_dh

    def __tov_rhs(self, h, y):
        """ODE r.h.s. for TOV equations with pseudo-enthalpy independent variable.

        Implements Eqs. (5) and (6) of Lindblom, Astrophys. J. 398, 569 (1992),
        and Eqs. (18), (27), (28) of Damour & Nagar, Phys. Rev. D 80, 084035 (2009).
        """
        dy = np.zeros_like(y)
        r = y[self._i_r]
        m = y[self._i_m]
        p, e, dedp = self._eos_eval(h)
        dr_dh = -r * (r - 2.0 * m) / (m + 4.0 * np.pi * r**3 * p)
        dm_dh = 4.0 * np.pi * r**2 * e * dr_dh
        dnu_dr = 2.0 * (m + 4.0 * np.pi * r**3 * p) / (r * (r - 2.0 * m))
        dy[self._i_r] = dr_dh
        dy[self._i_m] = dm_dh
        dy[self._i_nu] = dnu_dr * dr_dh
        self._update_even(dy, y, m, r, p, e, dedp, dr_dh, dnu_dr)
        self._update_odd(dy, y, m, r, p, e, dedp, dr_dh)
        return dy

    def __initial_data(self, pc, dh_fact=-1e-12):
        """Set initial data for the TOV ODE using the pseudo-enthalpy formalism.

        Lindblom (1992), Astrophys. J. 398, 569 — Eqs. (7) and (8).
        """
        y = np.zeros(self.nvar)
        ec = self.eos.EnergyDensity_Of_Pressure(pc)
        hc = self.eos.PseudoEnthalpy_Of_Pressure(pc)
        dedp_c = self.eos.EnergyDensityDeriv_Of_Pressure(pc)
        dhdp_c = 1.0 / (ec + pc)
        dedh_c = dedp_c / dhdp_c
        # Honour dh_fact. This read `dh = -1e-12 * hc`, hardcoding the default
        # and silently ignoring the argument, so `TOV(dhfact=...)` had no
        # effect on where the integration started or stopped -- and any test
        # varying it was really comparing a value against itself.
        dh = dh_fact * hc
        h0 = hc + dh
        h1 = 0.0 - dh
        r0 = np.sqrt(-3.0 * dh / (2.0 * np.pi * (ec + 3.0 * pc)))
        m0 = 4.0 * np.pi * r0**3 * ec / 3.0
        r0 *= 1.0 + 0.25 * dh * (ec - 3.0 * pc - 0.6 * dedh_c) / (ec + 3.0 * pc)
        m0 *= 1.0 + 0.6 * dh * dedh_c / ec
        y[self._i_r] = r0
        y[self._i_m] = m0
        y[self._i_nu] = 0.0
        a0 = 1.0
        if len(self.leven):
            for l, (iH, idH) in zip(self.leven, self._even_idx):
                y[iH] = a0 * r0**l
                y[idH] = a0 * l * r0 ** (l - 1)
        if len(self.lodd):
            for l, (iPsi, idPsi) in zip(self.lodd, self._odd_idx):
                y[iPsi] = a0 * r0 ** (l + 1)
                y[idPsi] = a0 * (l + 1) * r0**l
        return y, h0, h1

    def solve(self, pc):
        """Solve the TOV equations for a given central pressure *pc*.

        Lindblom (1992), Astrophys. J. 398, 569.
        """
        y, h0, h1 = self.__initial_data(pc, dh_fact=self.dhfact)
        sol = self.solver.solve(
            self.__tov_rhs,
            [h0, h1],
            y,
            first_step=abs(self.dhfact),
            rtol=self.ode_rtol,
            atol=self.ode_atol,
        )

        # Final Euler step to the surface
        y = sol.y[:, -1]
        dy = self.__tov_rhs(sol.t[-1], y)
        y[:] -= dy[:] * h1
        np.append(sol.y, y)
        M, R, C = self.__compute_mass_radius(y)
        # Match to Schwarzschild exterior
        sol.y[self._i_nu, :] += np.log(1.0 - (2.0 * M) / R) - sol.y[self._i_nu, -1]

        self.sol = sol
        if len(self.leven):
            k, h = {}, {}
            for l, (iH, idH) in zip(self.leven, self._even_idx):
                yyl = R * y[idH] / y[iH]
                k[l] = self.__compute_Love_even(l, C, yyl)
                h[l] = self.__compute_shape(l, C, yyl)
        if len(self.lodd):
            j = {}
            for l, (iPsi, idPsi) in zip(self.lodd, self._odd_idx):
                yyl = R * y[idPsi] / y[iPsi]
                j[l] = self.__compute_Love_odd(l, C, yyl)

        if len(self.leven) and len(self.lodd):
            return M, R, C, k, h, j
        elif len(self.leven) and not len(self.lodd):
            return M, R, C, k, h
        elif not len(self.leven) and len(self.lodd):
            return M, R, C, j
        else:
            return M, R, C

    # ------------------------------------------------------------------
    # SPEED: run solve() for many central pressures in parallel
    # ------------------------------------------------------------------
    def solve_parallel(self, pc_array, n_jobs=-1):
        """
        Solve TOV for an array of central pressures in parallel.

        Requires joblib (pip install joblib). Falls back to serial if unavailable.

        Parameters
        ----------
        pc_array : array-like of central pressures
        n_jobs   : number of parallel workers; -1 = all CPUs

        Returns
        -------
        list of solve() return values, one per entry in pc_array.

        Example
        -------
        results = tov.solve_parallel(pc_array)
        M_arr = np.array([r[0] for r in results])
        R_arr = np.array([r[1] for r in results])
        """
        if not _JOBLIB_AVAILABLE:
            return [self.solve(pc) for pc in pc_array]
        return Parallel(n_jobs=n_jobs)(_delayed(self.solve)(pc) for pc in pc_array)

    # ------------------------------------------------------------------
    # SPEED: one batched XLA call for many central pressures
    # ------------------------------------------------------------------
    def _solve_batched_jax(self, pc_array):
        """Integrate every central pressure in *pc_array* in a single
        ``jax.vmap``-ed XLA call.

        Returns ``(y_final, h1)`` -- the state vector at the end of each
        integration, shape ``(len(pc_array), nvar)``, and the matching
        end-of-integration pseudo-enthalpies -- or ``None`` if the batched
        path is unavailable (backend is not ``'jax'``, jax/diffrax missing,
        or this TOV has no native JAX RHS), in which case the caller should
        fall back to looping over :meth:`solve`.

        Only the ODE integration is batched. The per-star algebra that
        follows it (Euler step to the surface, Love numbers) is cheap and
        stays in a Python loop in :meth:`solve_many`, so the higher-multipole
        formulas need no vectorisation.
        """
        if not isinstance(self.solver, JaxSolver):
            return None
        try:
            import jax
            import jax.numpy as jnp
            import diffrax as dx
        except ImportError:
            return None

        jax_rhs = self._get_jax_rhs()
        eos_tables = self._get_jax_eos_tables()
        struct_key = self._get_jax_structural_key()
        if jax_rhs is None or eos_tables is None or struct_key is None:
            return None

        if not JaxSolver._x64_set:
            jax.config.update("jax_enable_x64", True)
            JaxSolver._x64_set = True

        initial = [self.__initial_data(pc, dh_fact=self.dhfact) for pc in pc_array]
        y0 = jnp.array(np.stack([np.asarray(i[0], dtype=float) for i in initial]))
        h0 = jnp.array(np.array([float(i[1]) for i in initial]))
        h1 = np.array([float(i[2]) for i in initial])

        rtol, atol = self.ode_rtol, self.ode_atol
        max_steps = self.solver.max_steps
        # Same cache discipline as JaxSolver.solve: key on the computation's
        # structure so one compiled kernel is reused across EOSs *and* across
        # calls, and add the batch size, which XLA bakes in as a shape.
        jit_key = ("_vmap", struct_key, self.nvar, rtol, atol, max_steps, len(pc_array))
        cache = JaxSolver._jit_solve_cache
        if jit_key not in cache:
            _jrhs = jax_rhs
            _dt0 = -abs(self.dhfact)

            def _one(t0_, t1_, y0_, tables_):
                sol = dx.diffeqsolve(
                    dx.ODETerm(lambda t, y, args: _jrhs(t, y, args)),
                    dx.Dopri5(),
                    t0=t0_,
                    t1=t1_,
                    dt0=_dt0,
                    y0=y0_,
                    args=tables_,
                    stepsize_controller=dx.PIDController(rtol=rtol, atol=atol),
                    saveat=dx.SaveAt(t1=True),
                    max_steps=max_steps,
                    # A single bad central pressure in the batch must not take
                    # the whole call down; it comes back as NaN instead and is
                    # filtered by the caller, matching the per-sample tolerance
                    # of the serial path.
                    throw=False,
                )
                return sol.ys[-1]

            cache[jit_key] = jax.jit(jax.vmap(_one, in_axes=(0, 0, 0, None)))

        y_final = np.asarray(
            cache[jit_key](h0, jnp.array(h1), y0, eos_tables), dtype=float
        )
        return y_final, h1

    def _nan_result(self):
        """A :meth:`solve`-shaped return value with every entry ``nan``, used
        to report a single failed central pressure without aborting a batch."""
        nan = float("nan")
        k = {int(l): nan for l in self.leven}
        h = {int(l): nan for l in self.leven}
        j = {int(l): nan for l in self.lodd}
        if len(self.leven) and len(self.lodd):
            return (nan, nan, nan, k, h, j)
        elif len(self.leven):
            return (nan, nan, nan, k, h)
        elif len(self.lodd):
            return (nan, nan, nan, j)
        return (nan, nan, nan)

    def solve_many(self, pc_array):
        """Solve the TOV equations for an array of central pressures.

        Same return values as :meth:`solve`, as a list with one entry per
        central pressure. On the ``'jax'`` backend every integration runs in
        a single batched XLA kernel, which is roughly 10x faster than looping
        over :meth:`solve` for a typical 30-point mass-radius sequence; on
        every other backend this *is* that loop, so results are unchanged.

        Unlike :meth:`solve` this does not populate ``self.sol``: only the
        final state of each integration is retained, not the full trajectory.

        Parameters
        ----------
        pc_array : array-like of central pressures

        Returns
        -------
        list of :meth:`solve` return values, one per entry in *pc_array*.
        Entries whose integration failed contain ``nan``.
        """
        pc_array = np.atleast_1d(np.asarray(pc_array, dtype=float))

        batched = self._solve_batched_jax(pc_array)
        if batched is None:
            # Serial fallback. One unsolvable central pressure must not abort
            # the rest of the sequence, so that a caller sweeping a range gets
            # the same per-point tolerance the batched path gives it for free
            # (diffrax returns nan there rather than raising).
            results = []
            for pc in pc_array:
                try:
                    results.append(self.solve(pc))
                except Exception:
                    results.append(self._nan_result())
            return results
        y_final, h1_all = batched

        results = []
        for y, h1 in zip(y_final, h1_all):
            y = np.array(y, dtype=float, copy=True)
            # Final Euler step to the surface, as in solve(). The integration
            # ends at h1, so that is the point at which the RHS is evaluated.
            dy = self.__tov_rhs(h1, y)
            y -= dy * h1
            M, R, C = self.__compute_mass_radius(y)

            if len(self.leven):
                k, h = {}, {}
                for l, (iH, idH) in zip(self.leven, self._even_idx):
                    yyl = R * y[idH] / y[iH]
                    k[l] = self.__compute_Love_even(l, C, yyl)
                    h[l] = self.__compute_shape(l, C, yyl)
            if len(self.lodd):
                j = {}
                for l, (iPsi, idPsi) in zip(self.lodd, self._odd_idx):
                    yyl = R * y[idPsi] / y[iPsi]
                    j[l] = self.__compute_Love_odd(l, C, yyl)

            if len(self.leven) and len(self.lodd):
                results.append((M, R, C, k, h, j))
            elif len(self.leven):
                results.append((M, R, C, k, h))
            elif len(self.lodd):
                results.append((M, R, C, j))
            else:
                results.append((M, R, C))
        return results

    def __compute_legendre(self, c, l):
        """
        Computes Legendre function values returning Pl2(x), Ql2(x) and their derivatives at x = 1/c -1
        """
        x = 1 / c - 1
        L = np.linspace(0, l - 1, l)
        nP = -prod((2 * l - 1) / 2 - L) / gamma(l) * 2**l * l * (l - 1)
        nQ = gamma(l) / factorial2(2 * l + 1) * (l + 1) * (l + 2)

        Pl2 = 0
        dPl2 = 0
        for i in np.linspace(2, l, l - 2 + 1, dtype=int):
            Pl2 = Pl2 + gamma(i) / gamma(i - 2) * comb(l, i) * prod(
                (l + i - 1) / 2 - L
            ) / gamma(l) * x ** (i - 2)
            dPl2 = dPl2 + gamma(i) / gamma(i - 2) * comb(l, i) * prod(
                (l + i - 1) / 2 - L
            ) / gamma(l) * (i - 2) * x ** (i - 3)

        dPl2 = 2**l * (-2 * x) * Pl2 / nP + 2**l * (1 - x**2) * dPl2 / nP
        Pl2 = 2**l * (1 - x**2) * Pl2 / nP

        Ql2 = (
            1
            / nQ
            * np.sqrt(np.pi)
            / 2 ** (l + 1)
            * gamma(l + 3)
            / gamma(l + 3 / 2)
            * (x**2 - 1)
            / x ** (l + 3)
            * hyp2f1((l + 3) / 2, (l + 4) / 2, l + 3 / 2, 1 / x**2)
        )
        dQl2 = (
            1
            / nQ
            * np.sqrt(np.pi)
            / 2 ** (l + 1)
            * gamma(l + 3)
            / gamma(l + 3 / 2)
            * (
                2
                * x ** (-2 - l)
                * hyp2f1((l + 3) / 2, (l + 4) / 2, l + 3 / 2, 1 / x**2)
                + (-3 - l)
                * x ** (-4 - l)
                * (-1 + x**2)
                * hyp2f1((l + 3) / 2, (l + 4) / 2, l + 3 / 2, 1 / x**2)
                - (
                    2
                    * ((l + 3) / 2)
                    * ((l + 4) / 2)
                    * x ** (-6 - l)
                    * (-1 + x**2)
                    * hyp2f1((l + 3) / 2 + 1, (l + 4) / 2 + 1, l + 3 / 2 + 1, 1 / x**2)
                    / (l + 3 / 2)
                )
            )
        )
        return Pl2, dPl2, Ql2, dQl2

    def __compute_psi(self, c, l):
        x = 1 / c
        CoefficientP = (
            poch(5, l - 2)
            / poch(2 - l, l - 2)
            / poch(3 + l, l - 2)
            * gamma(l - 2)
            * 2 ** (l - 2)
        )
        CoefficientQ = -1 / (l + 2)
        psiP = x**3 * hyp2f1(2 - l, 3 + l, 5, x / 2) * CoefficientP
        psiQ = (
            -(l + 2)
            * x ** (-1 - l)
            * (
                (1 + l) * x * hyp2f1(-1 + l, 2 + l, 2 + 2 * l, 2 / x)
                + (-1 + l) * hyp2f1(l, 3 + l, 3 + 2 * l, 2 / x)
            )
            / (1 + l)
            * CoefficientQ
        )
        dPsiP = 3 * x**2 * hyp2f1(2 - l, 3 + l, 5, x / 2) - 1 / 10 * (
            -6 + l + l**2
        ) * x**3 * hyp2f1(3 - l, 4 + l, 6, x / 2)
        dPsiP = dPsiP * CoefficientP
        dPsiQ = (
            1
            / (1 + l)
            / (3 + 2 * l)
            * (2 + l)
            * x ** (-3 - l)
            * (
                l
                * (3 + 5 * l + 2 * l**2)
                * x**2
                * hyp2f1(-1 + l, 2 + l, 2 + 2 * l, 2 / x)
                + (-1 + l)
                * (
                    (3 + 2 * l) ** 2 * x * hyp2f1(l, 3 + l, 3 + 2 * l, 2 / x)
                    + 2 * l * (3 + l) * hyp2f1(1 + l, 4 + l, 4 + 2 * l, 2 / x)
                )
            )
        )
        dPsiQ = dPsiQ * CoefficientQ
        return psiP, dPsiP, psiQ, dPsiQ

    def __compute_mass_radius(self, y):
        """
        Compute mass, radius, & compactness
        """
        R = y[self._i_r]
        M = y[self._i_m]
        return M, R, M / R

    def Compute_baryon_mass(self, sol):
        """
        Compute baryon mass
        """
        r = sol.y[self._i_r, :]
        m = sol.y[self._i_m, :]
        # e = self.EOSEnergyDensityOfPseudoEnthalpyGeometerized(sol.t,self.eos)
        e = np.array(
            [
                self.eos.EnergyDensity_Of_PseudoEnthalpy(sol.t[i])
                for i in range(len(sol.t))
            ]
        )
        return np.trapz(4 * np.pi * r**2.0 * e / np.sqrt(1 - 2 * m / r), r)

    def Compute_proper_radius(self, sol):
        """
        Compute proper radius
        """
        r = sol.y[self._i_r, :]
        m = sol.y[self._i_m, :]
        return np.trapz(r, 1.0 / np.sqrt((1 - 2 * m / r)), r)

    def __compute_Love_odd(self, ell, c, y):
        """
        Compute odd parity Love numbers given
        * the multipolar index ell
        * the compactness c
        * the ratio y = R Psi(R)'/Psi(R)
        Eq.(61) of Damour & Nagar, Phys. Rev. D 80 084035 (2009)
        """
        c2 = c**2
        c3 = c * c2
        c4 = c * c3
        c5 = c * c4
        j = 0.0
        if ell == 2:
            nj = 96 * c5 * (-1 + 2 * c) * (-3 + y)
            dj = 5.0 * (
                2
                * c
                * (
                    9
                    + 3 * c * (-3 + y)
                    + 2 * c2 * (-3 + y)
                    + 2 * c3 * (-3 + y)
                    - 3 * y
                    + 12 * c4 * (1 + y)
                )
                + 3 * (-1 + 2 * c) * (-3 + y) * np.log(1 - 2 * c)
            )
            j = nj / dj
        else:
            PsiP, dPsiP, PsiQ, dPsiQ = self.__compute_psi(c, ell)
            factor = -(c ** (2 * ell + 1))
            j = factor * (dPsiP - c * y * PsiP) / (dPsiQ - c * y * PsiQ)
        return j

    def __compute_Love_even(self, ell, c, y):
        """
        Compute even parity Love numbers given
        * the multipolar index ell
        * the compactness c
        * the ratio y = R H(R)'/H(R)
        Eq.(49) of Damour & Nagar, Phys. Rev. D 80 084035 (2009)
        """
        c2 = c**2
        c3 = c * c2
        c4 = c * c3
        c5 = c * c4
        c6 = c * c5
        c7 = c * c6
        c8 = c * c7
        c9 = c * c8
        c10 = c * c9
        c11 = c * c10
        c13 = c2 * c11
        c15 = c2 * c13
        c17 = c2 * c15
        k = 0.0
        if ell < 2:
            return k
        if ell == 2:
            nk = (1 - 2 * c) ** 2 * (2 + 2 * c * (y - 1) - y)
            dk = (
                2 * c * (6 - 3 * y + 3 * c * (5 * y - 8))
                + 4 * c3 * (13 - 11 * y + c * (3 * y - 2) + 2 * c2 * (1 + y))
                + 3 * (1 - 2 * c) ** 2 * (2 - y + 2 * c * (y - 1)) * np.log(1 - 2 * c)
            )
            k = 8 * c5 / 5 * nk / dk
        elif ell == 3:
            nk = (1 - 2 * c) ** 2 * (-3 - 3 * c * (-2 + y) + 2 * c2 * (-1 + y) + y)
            dk = 2 * c * (
                15 * (-3 + y)
                + 4 * c5 * (1 + y)
                - 45 * c * (-5 + 2 * y)
                - 20 * c3 * (-9 + 7 * y)
                + 2 * c4 * (-2 + 9 * y)
                + 5 * c2 * (-72 + 37 * y)
            ) - 15 * (1 - 2 * c) ** 2 * (
                -3 - 3 * c * (-2 + y) + 2 * c2 * (-1 + y) + y
            ) * np.log(
                1.0 / (1 - 2 * c)
            )
            k = 8 * c7 / 7 * nk / dk
        elif ell == 4:
            nk = (1 - 2 * c) ** 2 * (
                -7 * (-4 + y)
                + 28 * c * (-3 + y)
                - 34 * c2 * (-2 + y)
                + 12 * c3 * (-1 + y)
            )
            dk = 2 * c * (
                c2 * (5360 - 1910 * y)
                + c4 * (1284 - 996 * y)
                - 105 * (-4 + y)
                + 8 * c6 * (1 + y)
                + 105 * c * (-24 + 7 * y)
                + 40 * c3 * (-116 + 55 * y)
                + c5 * (-8 + 68 * y)
            ) - 15 * (1 - 2 * c) ** 2 * (
                -7 * (-4 + y)
                + 28 * c * (-3 + y)
                - 34 * c2 * (-2 + y)
                + 12 * c3 * (-1 + y)
            ) * np.log(
                1.0 / (1 - 2 * c)
            )
            k = 32 * c9 / 147 * nk / dk
        elif ell == 5:
            nk = (
                32
                * (1 - 2 * c) ** 2
                * c11
                * (
                    3 * (-5 + y)
                    - 15 * c * (-4 + y)
                    + 26 * c2 * (-3 + y)
                    - 18 * c3 * (-2 + y)
                    + 4 * c4 * (-1 + y)
                )
            )
            dk = 99.0 * (
                2
                * c
                * (
                    315 * (-5 + y)
                    + 8 * c7 * (1 + y)
                    - 315 * c * (-35 + 8 * y)
                    + 4 * c6 * (-2 + 27 * y)
                    - 56 * c5 * (-60 + 47 * y)
                    - 210 * c3 * (-170 + 57 * y)
                    + 105 * c2 * (-278 + 75 * y)
                    + 56 * c4 * (-345 + 158 * y)
                )
                - 105
                * (1 - 2 * c) ** 2
                * (
                    3 * (-5 + y)
                    - 15 * c * (-4 + y)
                    + 26 * c2 * (-3 + y)
                    - 18 * c3 * (-2 + y)
                    + 4 * c4 * (-1 + y)
                )
                * np.log(1.0 / (1 - 2 * c))
            )
            k = nk / dk
        elif ell == 6:
            nk = (
                1024
                * (1 - 2 * c) ** 2
                * c13
                * (
                    -33 * (-6 + y)
                    + 198 * c * (-5 + y)
                    - 444 * c2 * (-4 + y)
                    + 456 * c3 * (-3 + y)
                    - 208 * c4 * (-2 + y)
                    + 32 * c5 * (-1 + y)
                )
            )
            dk = 14157.0 * (
                2
                * c
                * (
                    -3465 * (-6 + y)
                    + 32 * c8 * (1 + y)
                    + 10395 * c * (-16 + 3 * y)
                    + 16 * c7 * (-2 + 39 * y)
                    + 2016 * c5 * (-122 + 55 * y)
                    - 64 * c6 * (-457 + 362 * y)
                    - 210 * c2 * (-2505 + 541 * y)
                    + 210 * c3 * (-3942 + 1015 * y)
                    - 84 * c4 * (-7917 + 2567 * y)
                )
                - 105
                * (1 - 2 * c) ** 2
                * (
                    -33 * (-6 + y)
                    + 198 * c * (-5 + y)
                    - 444 * c2 * (-4 + y)
                    + 456 * c3 * (-3 + y)
                    - 208 * c4 * (-2 + y)
                    + 32 * c5 * (-1 + y)
                )
                * np.log(1.0 / (1 - 2 * c))
            )
            k = nk / dk
        elif ell == 7:
            nk = (
                1024
                * (1 - 2 * c) ** 2
                * c15
                * (
                    143 * (-7 + y)
                    - 1001 * c * (-6 + y)
                    + 2750 * c2 * (-5 + y)
                    - 3740 * c3 * (-4 + y)
                    + 2600 * c4 * (-3 + y)
                    - 848 * c5 * (-2 + y)
                    + 96 * c6 * (-1 + y)
                )
            )
            dk = 20449.0 * (
                2
                * c
                * (
                    45045 * (-7 + y)
                    + 160 * c9 * (1 + y)
                    - 45045 * c * (-63 + 10 * y)
                    + 80 * c8 * (-2 + 53 * y)
                    - 432 * c7 * (-651 + 521 * y)
                    - 4620 * c3 * (-4333 + 902 * y)
                    + 1155 * c2 * (-9028 + 1621 * y)
                    + 96 * c6 * (-33964 + 15203 * y)
                    + 126 * c4 * (-168858 + 42239 * y)
                    - 84 * c5 * (-144545 + 45971 * y)
                )
                - 315
                * (1 - 2 * c) ** 2
                * (
                    143 * (-7 + y)
                    - 1001 * c * (-6 + y)
                    + 2750 * c2 * (-5 + y)
                    - 3740 * c3 * (-4 + y)
                    + 2600 * c4 * (-3 + y)
                    - 848 * c5 * (-2 + y)
                    + 96 * c6 * (-1 + y)
                )
                * np.log(1.0 / (1 - 2 * c))
            )
            k = nk / dk
        elif ell == 8:
            nk = (
                16
                * (1 - 2 * c) ** 2
                * (
                    2
                    * c
                    * (
                        c
                        * (
                            2
                            * c
                            * (
                                2
                                * c
                                * (
                                    -737 * (-4 + y)
                                    + 374 * c * (-3 + y)
                                    - 92 * c2 * (-2 + y)
                                    + 8 * c**3 * (-1 + y)
                                )
                                + 1573 * (-5 + y)
                            )
                            - 1859 * (-6 + y)
                        )
                        + 572 * (-7 + y)
                    )
                    - 143 * (-8 + y)
                )
            )
            dk = 286 * c * (
                4
                * (
                    90090
                    + c
                    * (
                        -900900
                        + c
                        * (
                            3768765
                            + c
                            * (
                                -8528520
                                + c
                                * (
                                    11259633
                                    + 2
                                    * c
                                    * (
                                        -4349499
                                        + c
                                        * (
                                            1858341
                                            + 8
                                            * c
                                            * (-47328 + c * (3092 + (-1 + c) * c))
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
                + (-1 + c)
                * (-1 + 2 * c)
                * (
                    -45045
                    + 4
                    * c
                    * (
                        90090
                        + c
                        * (
                            -285285
                            + c
                            * (
                                450450
                                + c
                                * (
                                    -365211
                                    + 4 * c * (34881 + c * (-4887 + 2 * c * (36 + c)))
                                )
                            )
                        )
                    )
                )
                * y
            ) - 45045 * (1 - 2 * c) ** 2 * (
                2
                * c
                * (
                    c
                    * (
                        2
                        * c
                        * (
                            2
                            * c
                            * (
                                -737 * (-4 + y)
                                + 374 * c * (-3 + y)
                                - 92 * c2 * (-2 + y)
                                + 8 * c**3 * (-1 + y)
                            )
                            + 1573 * (-5 + y)
                        )
                        - 1859 * (-6 + y)
                    )
                    + 572 * (-7 + y)
                )
                - 143 * (-8 + y)
            ) * np.log(
                1.0 / (1 - 2 * c)
            )
            k = 256 / 2431 * c17 * nk / dk
        else:
            Pl2, dPl2, Ql2, dQl2 = self.__compute_legendre(c, ell)
            k = (
                -1
                / 2
                * c ** (2 * ell + 1)
                * (dPl2 - c * y * Pl2)
                / (dQl2 - c * y * Ql2)
            )
        return k

    def __compute_shape(self, ell, c, y):
        """
        Compute even shape numbers given
        * the multipolar index ell
        * the compactness c
        * the ratio y = R H(R)'/H(R)
        Eq.(95) of Damour & Nagar, Phys. Rev. D 80 084035 (2009)
        """
        c2 = c**2
        c3 = c * c2
        c4 = c * c3
        c5 = c * c4
        c6 = c * c5
        c7 = c * c6
        c8 = c * c7
        c9 = c * c8
        c10 = c * c9
        c11 = c * c10
        c13 = c2 * c11
        c15 = c2 * c13
        c17 = c2 * c15
        h = 0.0
        if ell < 2:
            return h
        if ell == 2:
            nh = -2 + 6 * c + 2 * c3 * (1 + y) - c2 * (6 + y)
            dh = 2 * c * (
                6
                + c2 * (26 - 22 * y)
                - 3 * y
                + 4 * c4 * (1 + y)
                + 3 * c * (-8 + 5 * y)
                + c3 * (-4 + 6 * y)
            ) - 3 * (1 - 2 * c) ** 2 * (2 + 2 * c * (-1 + y) - y) * np.log(
                1.0 / (1 - 2 * c)
            )
            h = -8 * c5 * nh / dh
        else:
            Pl2, dPl2, Ql2, dQl2 = self.__compute_legendre(c, ell)
            term1 = (1 - 2 * c) / c
            term2 = (
                1
                / (ell - 1)
                / (ell + 2)
                * (
                    2 * c * y
                    + ell * (ell + 1)
                    + 4 * c**2 / (1 - 2 * c)
                    - 2 * (1 - 2 * c)
                )
            )
            factor = (
                c ** (ell + 1) * Pl2 * (1 - (dPl2 / Pl2 - c * y) / (dQl2 / Ql2 - c * y))
            )
            h = (term1 + term2) * factor
        return h

    def Compute_Lambda(self, ell, k, C):
        r"""
        Compute tidal polarizability $\Lambda_\ell$
        from Love numbers and compactness
        Note: Yagi's $\bar{\lambda}_\ell$ is $\Lambda_\ell$
        """
        div = 1.0 / (factorial2(2 * ell - 1) * C ** (2 * ell + 1))
        return 2.0 * k * div
