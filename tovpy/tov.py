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
from numpy import log
from scipy.integrate import solve_ivp
from scipy.special import factorial2, gamma, factorial2, hyp2f1, poch
from math import comb, prod

# ---------------------------------------------------------------------------
# Optional joblib parallelism
# ---------------------------------------------------------------------------
try:
    from joblib import Parallel, delayed as _delayed

    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

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

from numba import njit
import numpy as np
from math import pi


@njit(cache=True, fastmath=True)
def tov_rhs_jit(
    h,
    y,
    log_h,
    log_p,
    log_e,
    log_dedp,
    even_il,
    odd_il,
    even_ell,
    odd_ell,
    has_even,
    has_odd,
    i_r,
    i_m,
    i_nu,
):
    n = y.shape[0]
    dy = np.empty(n)

    # EOS interpolation (log-log)
    p = _interp_positive(h, log_h, log_p)
    e = _interp_positive(h, log_h, log_e)
    dedp = _interp_positive(p, log_p, log_dedp)

    # Basic TOV variables
    r = y[i_r]
    m = y[i_m]

    r2 = r * r
    r3 = r * r2
    denom = m + 4.0 * pi * r3 * p

    dr_dh = -r * (r - 2.0 * m) / denom
    dm_dh = 4.0 * pi * r2 * e * dr_dh
    dnu_dr = 2.0 * denom / (r * (r - 2.0 * m))

    dy[i_r] = dr_dh
    dy[i_m] = dm_dh
    dy[i_nu] = dnu_dr * dr_dh

    # Even perturbations
    if has_even:
        div_r = 1.0 / r
        div_r2 = div_r * div_r
        exp_lam = 1.0 / (1.0 - 2.0 * m * div_r)

        dnu2 = dnu_dr * dnu_dr
        C1 = 2.0 * div_r + exp_lam * (2.0 * m * div_r2 + 4.0 * pi * r * (p - e))

        for k in range(even_ell.shape[0]):
            ell = even_ell[k]
            iH = even_il[2 * k]
            idH = even_il[2 * k + 1]

            Lam = ell * (ell + 1)

            C0 = -dnu2
            C0 += exp_lam * (
                -Lam * div_r2 + 4.0 * pi * (5.0 * e + 9.0 * p + (e + p) * dedp)
            )

            H = y[iH]
            dH = y[idH]

            dy[iH] = dH * dr_dh
            dy[idH] = -(C0 * H + C1 * dH) * dr_dh

    # Odd perturbations
    if has_odd:
        div_r = 1.0 / r
        div_r2 = div_r * div_r
        div_r3 = div_r * div_r2
        exp_lam = 1.0 / (1.0 - 2.0 * m * div_r)

        C1 = exp_lam * (2.0 * m + 4.0 * pi * r3 * (p - e)) * div_r2

        for k in range(odd_ell.shape[0]):
            ell = odd_ell[k]
            iPsi = odd_il[2 * k]
            idPsi = odd_il[2 * k + 1]

            Lam = ell * (ell + 1)

            C0 = exp_lam * (-Lam * div_r2 + 6.0 * m * div_r3 - 4.0 * pi * (e - p))

            Psi = y[iPsi]
            dPsi = y[idPsi]

            dy[iPsi] = dPsi * dr_dh
            dy[idPsi] = -(C0 * Psi + C1 * dPsi) * dr_dh

    return dy


@_njit(cache=True)
def _interp_positive(val, x_table, y_table):
    """
    Linear interpolation in log-log space on a strictly positive,
    monotonically increasing x_table.
    Both val and all table entries must be > 0.
    """
    lv = np.log(val)
    # binary search
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
    """
    Class to solve the Tolman-Oppenheimer-Volkov stellar structure
    equations together with even/odd parity stationary bartropic perturbations

    Lindblom , Astrophys. J. 398 569. (1992)
    Damour & Nagar, Phys. Rev. D 80, 084035 (2009)

    Work in geometric units

    Reference codes:
    * https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_neutron_star_t_o_v_8c_source.html
    * https://bitbucket.org/bernuzzi/tov/src/master/TOVL.m
    * https://lscsoft.docs.ligo.org/bilby/_modules/bilby/gw/eos/tov_solver.html

    """

    def __init__(
        self,
        eos=None,  # EOS instance
        leven=[],  # multipole indexes of even perturbations
        lodd=[],  # multipole indexes of odd perturbations
        dhfact=-1e-12,  # ODE step
        ode_method="DOP853",
        ode_atol=1e-9,
        ode_rtol=1e-9,
    ):

        if not eos:
            raise ValueError("Must provide a EOS")
        self.eos = eos

        leven = np.array(leven)
        lodd = np.array(lodd)

        # Solve perturbation equations for these indexes
        self.leven = leven[leven > 1]
        self.lodd = lodd[lodd > 1]

        # Build variable list
        var = self.__buildvars()
        self.nvar = len(var)
        self.var = dict(zip(var, range(self.nvar)))
        self.ivar = {v: k for k, v in self.var.items()}

        # ------------------------------------------------------------------
        # SPEED: cache integer indices once so the RHS never does dict lookups
        # ------------------------------------------------------------------
        self._i_r = self.var["r"]
        self._i_m = self.var["m"]
        self._i_nu = self.var["nu"]

        # Per-perturbation index pairs: [(i_H, i_dH), ...] and [(i_Psi, i_dPsi), ...]
        self._even_idx = [
            (self.var["H{}".format(l)], self.var["dH{}".format(l)]) for l in self.leven
        ]
        self._odd_idx = [
            (self.var["Psi{}".format(l)], self.var["dPsi{}".format(l)])
            for l in self.lodd
        ]

        # int64 arrays of ell values needed by the JIT RHS
        self._even_ells = np.array(self.leven, dtype=np.int64)
        self._odd_ells = np.array(self.lodd, dtype=np.int64)

        # ODE solver options
        if dhfact > 0.0:
            raise ValueError("ODE timestep must be negative")
        self.dhfact = dhfact
        self.ode_method = ode_method
        self.ode_atol = ode_atol
        self.ode_rtol = ode_rtol

        # Pre-build EOS tables for JIT interpolation (done once at construction)
        self._build_eos_tables()

        if _NUMBA_AVAILABLE:
            # JIT the RHS with pre-built tables and cached indices
            self.__tov_rhs = tov_rhs_jit
        else:
            # Python fallback (slower, but no Numba dependency)
            self.__tov_rhs = self.__tov_rhs_base

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
        """
        Pre-sample the EOS once at construction and store log-space arrays
        for use in the Numba JIT RHS.

        Key fix: h is POSITIVE inside the star (the integrator runs from
        h0 > 0 down to h ~ 0).  We therefore interpolate on h directly
        (not on -h), storing log(h) as the independent variable so that
        _interp_positive can do a standard log-log binary search.

        Tables stored:
          _lh_lp  : log(h) -> log(p)     shape (n,2)
          _lh_le  : log(h) -> log(e)     shape (n,2)
          _lp_ldedp : log(p) -> log(dedp) shape (n,2)
        """
        # Sample pressure from just above the minimum h the EOS supports.
        # We use the EOS's own central-value range by probing a wide span
        # of pressures and keeping only the physically valid portion.
        try:
            p_min = float(self.eos.p_min) * 1.001
            p_max = float(self.eos.p_max) * 0.999
        except AttributeError:
            # Fallback: walk inward from a very small pressure until the
            # EOS returns a positive pseudo-enthalpy.
            p_min = 1e-19
            p_max = 1e-8

        p_arr = np.logspace(np.log10(p_min), np.log10(p_max), n_points)

        # Evaluate EOS; drop any points where h <= 0 (unphysical / below surface)
        h_arr = np.array([self.eos.PseudoEnthalpy_Of_Pressure(p) for p in p_arr])
        e_arr = np.array([self.eos.EnergyDensity_Of_Pressure(p) for p in p_arr])
        dedp_arr = np.array([self.eos.EnergyDensityDeriv_Of_Pressure(p) for p in p_arr])

        valid = (h_arr > 0) & (e_arr > 0) & (dedp_arr > 0) & (p_arr > 0)
        h_arr = h_arr[valid]
        p_arr = p_arr[valid]
        e_arr = e_arr[valid]
        dedp_arr = dedp_arr[valid]

        # Sort by ascending h (needed for binary search)
        idx = np.argsort(h_arr)
        h_arr = h_arr[idx]
        p_arr = p_arr[idx]
        e_arr = e_arr[idx]
        dedp_arr = dedp_arr[idx]

        # Contiguous 1D log-space arrays — no slice allocation on every RHS call
        self._log_h = np.ascontiguousarray(np.log(h_arr))
        self._log_p = np.ascontiguousarray(np.log(p_arr))
        self._log_e = np.ascontiguousarray(np.log(e_arr))
        self._log_dedp = np.ascontiguousarray(np.log(dedp_arr))
        self._log_p_s = self._log_p  # p sorted ascending, reused as dedp key

        # Flat int64 index arrays for JIT RHS
        if len(self.leven):
            ev = []
            for iH, idH in self._even_idx:
                ev += [iH, idH]
            self._even_il = np.array(ev, dtype=np.int64)
        else:
            self._even_il = np.empty(0, dtype=np.int64)
        if len(self.lodd):
            ov = []
            for iP, idP in self._odd_idx:
                ov += [iP, idP]
            self._odd_il = np.array(ov, dtype=np.int64)
        else:
            self._odd_il = np.empty(0, dtype=np.int64)
        self._even_ell_arr = np.array(self.leven, dtype=np.int64)
        self._odd_ell_arr = np.array(self.lodd, dtype=np.int64)

        # Backward-compat aliases for the Python RHS path
        self._lh_lp = np.column_stack([self._log_h, self._log_p])
        self._lh_le = np.column_stack([self._log_h, self._log_e])
        self._lp_ldedp = np.column_stack([self._log_p, self._log_dedp])

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
        exp_lam = 1.0 / (1.0 - 2.0 * m * div_r)  # Eq. (18)
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
        exp_lam = 1.0 / (1.0 - 2.0 * m * div_r)  # Eq. (18)
        C1 = exp_lam * (2 * m + 4 * np.pi * r3 * (p - e)) * div_r2
        C0 = np.zeros(max(ell) + 1)
        for l in ell:
            Lam = l * (l + 1)
            C0[l] = exp_lam * (-Lam * div_r2 + 6 * m * div_r3 - 4 * np.pi * (e - p))
        return C1, C0

    def __tov_rhs_base(self, h, y):
        """
        ODE r.h.s. for TOV equations with pseudo-enthalpy independent variable.
        Implements Eqs. (5) and (6) of Lindblom, Astrophys. J. 398, 569 (1992).
        Also uses Eqs. (7) and (8) [ibid] for inner boundary data, and
        Eqs. (18), (27), (28) of Damour & Nagar, Phys. Rev. D 80, 084035 (2009)
        for the metric perturbation used to obtain the Love number.

        SPEED: uses cached integer indices (_i_r, _i_m, _i_nu, _even_idx,
               _odd_idx) instead of dict lookups on every call.
               When Numba is available, EOS lookups use pre-built log tables
               instead of calling through Python.
        """
        dy = np.zeros_like(y)

        # SPEED: integer index, not dict lookup
        r = y[self._i_r]
        m = y[self._i_m]

        # EOS calls
        if _NUMBA_AVAILABLE:
            # h is positive; interpolate directly on log(h)
            p = _interp_positive(h, self._lh_lp[:, 0], self._lh_lp[:, 1])
            e = _interp_positive(h, self._lh_le[:, 0], self._lh_le[:, 1])
            dedp = _interp_positive(p, self._lp_ldedp[:, 0], self._lp_ldedp[:, 1])
        else:
            p = self.eos.Pressure_Of_PseudoEnthalpy(h)
            e = self.eos.EnergyDensity_Of_PseudoEnthalpy(h)
            dedp = self.eos.EnergyDensityDeriv_Of_Pressure(p)

        # TOV
        dr_dh = -r * (r - 2.0 * m) / (m + 4.0 * np.pi * r**3 * p)
        dm_dh = 4.0 * np.pi * r**2 * e * dr_dh
        dnu_dr = 2.0 * (m + 4.0 * np.pi * r**3 * p) / (r * (r - 2.0 * m))

        # SPEED: integer index, not dict lookup
        dy[self._i_r] = dr_dh
        dy[self._i_m] = dm_dh
        dy[self._i_nu] = dnu_dr * dr_dh

        # Even perturbations
        if len(self.leven) != 0:
            C1, C0 = self.__pert_even(self.leven, m, r, p, e, dedp, dnu_dr)
            # SPEED: pre-cached index pairs, no per-step string formatting
            for l, (iH, idH) in zip(self.leven, self._even_idx):
                H = y[iH]
                dH = y[idH]
                dy[iH] = dH * dr_dh
                dy[idH] = -(C0[l] * H + C1 * dH) * dr_dh

        # Odd perturbations
        if len(self.lodd) != 0:
            C1, C0 = self.__pert_odd(self.lodd, m, r, p, e, dedp)
            # SPEED: pre-cached index pairs, no per-step string formatting
            for l, (iPsi, idPsi) in zip(self.lodd, self._odd_idx):
                Psi = y[iPsi]
                dPsi = y[idPsi]
                dy[iPsi] = dPsi * dr_dh
                dy[idPsi] = -(C0[l] * Psi + C1 * dPsi) * dr_dh

        return dy

    def __initial_data(self, pc, dh_fact=-1e-12, verbose=False):
        """
        Set initial data for the solution of TOV equations using the pseudo-enthalpy formalism introduced in:
        Lindblom (1992) "Determining the Nuclear Equation of State from Neutron-Star Masses and Radii", Astrophys. J. 398 569.
        * input the central pressure
        """
        y = np.zeros(self.nvar)
        # Central values
        ec = self.eos.EnergyDensity_Of_Pressure(pc)
        hc = self.eos.PseudoEnthalpy_Of_Pressure(pc)
        dedp_c = self.eos.EnergyDensityDeriv_Of_Pressure(pc)
        dhdp_c = 1.0 / (ec + pc)
        dedh_c = dedp_c / dhdp_c
        dh = -1e-12 * hc
        h0 = hc + dh
        h1 = 0.0 - dh
        r0 = np.sqrt(-3.0 * dh / (2.0 * np.pi * (ec + 3.0 * pc)))
        m0 = 4.0 * np.pi * r0**3 * ec / 3.0
        # Series expansion for the initial core
        r0 *= 1.0 + 0.25 * dh * (ec - 3.0 * pc - 0.6 * dedh_c) / (
            ec + 3.0 * pc
        )  # second factor Eq. (7) of Lindblom (1992)
        m0 *= (
            1.0 + 0.6 * dh * dedh_c / ec
        )  # second factor of Eq. (8) of Lindblom (1992)

        # SPEED: integer indices
        y[self._i_r] = r0
        y[self._i_m] = m0
        y[self._i_nu] = 0.0

        #  Initial data for the ell-perturbation
        a0 = 1.0
        if len(self.leven) != 0:
            for l, (iH, idH) in zip(self.leven, self._even_idx):
                y[iH] = a0 * r0**l
                y[idH] = a0 * l * r0 ** (l - 1)
        if len(self.lodd) != 0:
            for l, (iPsi, idPsi) in zip(self.lodd, self._odd_idx):
                y[iPsi] = a0 * r0 ** (l + 1)
                y[idPsi] = a0 * (l + 1) * r0**l
        return y, h0, h1

    def solve(self, pc):
        """
        Solves the Tolman-Oppenheimer-Volkov stellar structure equations using the pseudo-enthalpy formalism introduced in:
        Lindblom (1992) "Determining the Nuclear Equation of State from Neutron-Star Masses and Radii", Astrophys. J. 398 569.
        """
        # Initial data
        y, h0, h1 = self.__initial_data(pc, dh_fact=self.dhfact, verbose=True)
        # Fallback: standard scipy solve_ivp
        sol = solve_ivp(
            self.__tov_rhs,
            [h0, h1],
            y,
            first_step=abs(self.dhfact),
            method=self.ode_method,
            rtol=self.ode_rtol,
            atol=self.ode_atol,
        )
        # Take one final Euler step to get to surface
        y = sol.y[:, -1]
        dy = self.__tov_rhs(sol.t[-1], y)
        y[:] -= dy[:] * h1
        np.append(sol.y, y)
        # Mass, Radius & Compactness
        M, R, C = self.__compute_mass_radius(y)
        # Match to Schwarzschild exterior
        sol.y[self._i_nu, :] += np.log(1.0 - (2.0 * M) / R) - sol.y[self._i_nu, -1]
        if _NUMBA_AVAILABLE:
            sol.y[:, 1] = y  # update stub with corrected final state

        self.sol = sol
        if len(self.leven) != 0:
            k, h = {}, {}
            for l, (iH, idH) in zip(self.leven, self._even_idx):
                yyl = R * y[idH] / y[iH]
                k[l] = self.__compute_Love_even(l, C, yyl)
                h[l] = self.__compute_shape(l, C, yyl)
        if len(self.lodd) != 0:
            j = {}
            for l, (iPsi, idPsi) in zip(self.lodd, self._odd_idx):
                yyl = R * y[idPsi] / y[iPsi]
                j[l] = self.__compute_Love_odd(l, C, yyl)

        if len(self.leven) != 0 and len(self.lodd) != 0:
            return M, R, C, k, h, j
        elif len(self.leven) != 0 and len(self.lodd) == 0:
            return M, R, C, k, h
        elif len(self.leven) == 0 and len(self.lodd) != 0:
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
        # SPEED: integer indices
        R = y[self._i_r]
        M = y[self._i_m]
        return M, R, M / R

    def Compute_baryon_mass(self, sol):
        """
        Compute baryon mass
        """
        # SPEED: integer indices
        r = sol.y[self._i_r, :]
        m = sol.y[self._i_m, :]
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
        # SPEED: integer indices
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
                + 3 * (-1 + 2 * c) * (-3 + y) * log(1 - 2 * c)
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
            # https://bitbucket.org/bernuzzi/tov/src/master/ComputeLegendre.m
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
        # elif ell == 3:
        #     nh = -5 + 15*c + 2*c3*(1 + y) - c2*(12 + y)
        #     dh = (5.*(2*c*(15*(-3 + y) + 4*c5*(1 + y) - 45*c*(-5 + 2*y) - 20*c3*(-9 + 7*y) + 2*c4*(-2 + 9*y) + 5*c2*(-72 + 37*y)) - 15*(1 - 2*c)**2*(-3 - 3*c*(-2 + y) + 2*c2*(-1 + y) + y)*np.log(1.0/(1 - 2*c))))
        #     h = 16*c7*nh/dh
        # elif ell == 4:
        #     nh = -9 + 27*c + 2*c3*(1 + y) - c2*(20 + y)
        #     dh = (21.*(2*c*(c2*(5360 - 1910*y) + c4*(1284 - 996*y) - 105*(-4 + y) + 8*c6*(1 + y) + 105*c*(-24 + 7*y)  + 40*c3*(-116 + 55*y) + c5*(-8 + 68*y)) - 15*(1 - 2*c)**2*(-7*(-4 + y) + 28*c*(-3 + y) - 34*c2*(-2 + y) + 12*c3*(-1 + y))*np.log(1.0/(1 - 2*c))))
        #     h = -64*c9*nh/dh
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
