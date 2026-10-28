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

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import factorial2, gamma, hyp2f1, poch
from math import comb, prod

from .solvers import SolverResult, make_solver


def _evaluate(f, x):
    """Evaluate f on the array x, in one call if f takes arrays, else point by
    point."""
    try:
        y = np.asarray(f(x), dtype=float)
        if y.shape == x.shape:
            return y
    except (TypeError, ValueError):
        pass
    return np.array([f(v) for v in x])


def _k2_small_compactness(c, y, terms=30):
    """Even-parity l = 2 Love number for compactness c < 0.03.

    The closed form divides two quantities of order c^5 and loses all precision
    below c ~ 3e-4. Expanding log(1 - 2c) in its denominator as a power series
    makes the terms below c^5 cancel exactly, so c^5 divides out and the result
    is exact to rounding."""
    poly = Polynomial(
        [
            0,
            2 * (6 - 3 * y),
            6 * (5 * y - 8),
            4 * (13 - 11 * y),
            4 * (3 * y - 2),
            8 * (1 + y),
        ]
    )
    q = 3 * Polynomial([1, -4, 4]) * Polynomial([2 - y, 2 * (y - 1)])
    log = Polynomial([0] + [-(2.0**n) / n for n in range(1, terms + 1)])
    dk_over_c5 = Polynomial((poly + q * log).coef[5:])(c)
    nk = (1 - 2 * c) ** 2 * (2 + 2 * c * (y - 1) - y)
    return 8 / 5 * nk / dk_over_c5


@dataclass(frozen=True)
class TOVEquations:
    """r.h.s. of the TOV equations in the pseudo-enthalpy h, with the even- and
    odd-parity stationary perturbations of multipoles ``leven``, ``lodd``. It uses
    only the array module ``xp``, so numpy and jax.numpy share one
    implementation.

    The state is r, m, nu, then (H_l, dH_l) for each even l, then (Psi_l, dPsi_l)
    for each odd l. ``tables`` are the EOS log tables of
    :meth:`TOV._build_eos_tables`.

    Lindblom, Astrophys. J. 398, 569 (1992), Eqs. (5)-(6); Damour & Nagar,
    Phys. Rev. D 80, 084035 (2009), Eqs. (18), (27)-(29), (31).
    """

    leven: tuple = ()
    lodd: tuple = ()

    def __call__(self, h, y, tables, xp):
        log_h, log_p, log_e, log_p_sorted, log_dedp = tables
        lh = xp.log(h)
        p = xp.exp(xp.interp(lh, log_h, log_p))
        e = xp.exp(xp.interp(lh, log_h, log_e))
        dedp = xp.exp(xp.interp(xp.log(p), log_p_sorted, log_dedp))

        r, m = y[0], y[1]
        pi = np.pi
        dr_dh = -r * (r - 2.0 * m) / (m + 4.0 * pi * r**3 * p)
        dnu_dr = 2.0 * (m + 4.0 * pi * r**3 * p) / (r * (r - 2.0 * m))
        exp_lam = 1.0 / (1.0 - 2.0 * m / r)
        dy = [dr_dh, 4.0 * pi * r**2 * e * dr_dh, dnu_dr * dr_dh]

        i = 3
        if self.leven:
            C1 = 2.0 / r + exp_lam * (2.0 * m / r**2 + 4.0 * pi * r * (p - e))
            source = 4.0 * pi * (5.0 * e + 9.0 * p + (e + p) * dedp)
        for l in self.leven:
            C0 = -(dnu_dr**2) + exp_lam * (source - l * (l + 1) / r**2)
            H, dH = y[i], y[i + 1]
            dy += [dH * dr_dh, -(C0 * H + C1 * dH) * dr_dh]
            i += 2
        if self.lodd:
            C1 = exp_lam * (2.0 * m + 4.0 * pi * r**3 * (p - e)) / r**2
        for l in self.lodd:
            C0 = exp_lam * (-l * (l + 1) / r**2 + 6.0 * m / r**3 - 4.0 * pi * (e - p))
            Psi, dPsi = y[i], y[i + 1]
            dy += [dPsi * dr_dh, -(C0 * Psi + C1 * dPsi) * dr_dh]
            i += 2
        return xp.stack(dy)


class TOV(object):
    """Class to solve the Tolman-Oppenheimer-Volkov stellar structure
    equations together with even/odd parity stationary bartropic perturbations.

    References
    ----------
    Lindblom, Astrophys. J. 398, 569 (1992)
    Damour & Nagar, Phys. Rev. D 80, 084035 (2009)

    Works in geometric units.

    Reference codes

    * https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_neutron_star_t_o_v_8c_source.html
    * https://bitbucket.org/bernuzzi/tov/src/master/TOVL.m
    * https://lscsoft.docs.ligo.org/bilby/_modules/bilby/gw/eos/tov_solver.html

    Parameters
    ----------
    eos : EOS
        Equation of state instance.
    leven : list of int
        Multipole indices for even-parity perturbations (Love numbers k, h).
        The solver ignores values <= 1.
    lodd : list of int
        Multipole indices for odd-parity perturbations (Love numbers j).
        The solver ignores values <= 1.
    dhfact : float
        Initial ODE step factor (must be negative, default ``-1e-12``).
    ode_method : str or None
        Integration method. On ``'scipy'``, any ``solve_ivp`` method (default
        ``'DOP853'``). On ``'jax'``, a diffrax solver name or a scipy name, which
        picks the diffrax method of the same order (default ``'Dopri5'``).
        ``Dopri5`` is the fastest on JAX. ``'DOP853'`` (``Dopri8``) takes fewer
        steps but costs about four times as much per star.
    ode_atol : float
        Absolute ODE tolerance (default ``1e-8``).
    ode_rtol : float
        Relative ODE tolerance (default ``1e-8``). At ``1e-6`` the tidal
        deformability of a tabulated EOS with sharp features can be off by tens
        of percent. At ``1e-8`` it stays within ~1e-4.
    ode_backend : str or ODESolver
        ODE solver backend, one of

        * ``'scipy'`` (default), :func:`scipy.integrate.solve_ivp`.
        * ``'jax'``, diffrax, compiled by JAX. :meth:`solve_many` solves all
          central pressures in one kernel. Requires
          ``pip install "jax[cpu]" diffrax``.
        * A pre-instantiated :class:`~tovpy.solvers.ODESolver` instance
          (e.g. ``make_solver('scipy', method='RK45')``).
    eos_table_points : int
        Number of EOS samples the r.h.s. interpolates (default 2000).

    Examples
    --------
    Basic usage (default scipy backend)::

        tov = TOV(eos=eos)
        M, R, C = tov.solve(pc)

    With tidal parameters::

        tov = TOV(eos=eos, leven=[2], lodd=[2])
        M, R, C, k, h, j = tov.solve(pc)

    Alternative backends::

        tov_jax   = TOV(eos=eos, ode_backend='jax')
        tov_rk45  = TOV(eos=eos, ode_backend=make_solver('scipy', method='RK45'))
    """

    def __init__(
        self,
        eos=None,
        leven=[],
        lodd=[],
        dhfact=-1e-12,
        ode_method=None,
        ode_atol=1e-8,
        ode_rtol=1e-8,
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

        self.solver = make_solver(
            ode_backend, **({} if ode_method is None else {"method": ode_method})
        )
        self.equations = TOVEquations(
            tuple(int(l) for l in self.leven), tuple(int(l) for l in self.lodd)
        )

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
        """Sample the EOS once, in log space, over every pseudo-enthalpy a
        solve can reach, so the r.h.s. only interpolates.

        ``self._tables`` holds log h, log p and log e ordered by h, then log p
        and log de/dp ordered by p.
        """
        # A solve with central pressure pc stops at |dhfact| h(pc). The grid runs
        # from h(p_hi), above any central pressure, down to 1e-2 |dhfact| h(p_lo),
        # below every stopping point. Sampling the EOS itself keeps its
        # power-law continuations off the table.
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
        p_arr = _evaluate(self.eos.Pressure_Of_PseudoEnthalpy, h_arr)
        e_arr = _evaluate(self.eos.EnergyDensity_Of_PseudoEnthalpy, h_arr)
        dedp_arr = _evaluate(self.eos.EnergyDensityDeriv_Of_Pressure, p_arr)

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

        p_idx = np.argsort(p_arr)
        self._tables = (
            np.log(h_arr),
            np.log(p_arr),
            np.log(e_arr),
            np.log(p_arr[p_idx]),
            np.log(dedp_arr[p_idx]),
        )

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

        Lindblom (1992), Astrophys. J. 398, 569. Keeps the trajectory, which ends
        at the surface, in ``self.sol``. Returns nan values if the integration
        stops short of the surface.
        """
        y0, h0, h1 = self.__initial_data(pc, dh_fact=self.dhfact)
        h_end, y_end, success, traj = self.solver.integrate(
            self.equations,
            self._tables,
            h0,
            h1,
            y0,
            abs(self.dhfact),
            self.ode_rtol,
            self.ode_atol,
            trajectory=True,
        )
        y_surf, result = self._star(h_end, y_end, success)
        if y_surf is not None:
            M, R = y_surf[self._i_m], y_surf[self._i_r]
            ys = np.column_stack([traj.y, y_surf])
            # Match to the Schwarzschild exterior
            ys[self._i_nu, :] += np.log(1.0 - 2.0 * M / R) - ys[self._i_nu, -1]
            traj = SolverResult(np.append(traj.t, 0.0), ys)
        self.sol = traj
        return result

    def solve_many(self, pc_array):
        """Solve the TOV equations for an array of central pressures.

        Returns one :meth:`solve` result per central pressure, nan where the
        integration stops short of the surface. On ``'jax'`` all central
        pressures run in one compiled kernel. Leaves ``self.sol`` unchanged.

        Parameters
        ----------
        pc_array : array-like of central pressures

        Returns
        -------
        list of :meth:`solve` return values, one per entry in *pc_array*.
        """
        pc_array = np.atleast_1d(np.asarray(pc_array, dtype=float))
        initial = [self.__initial_data(pc, dh_fact=self.dhfact) for pc in pc_array]
        h_end, y_end, success = self.solver.integrate_many(
            self.equations,
            self._tables,
            np.array([i[1] for i in initial]),
            np.array([i[2] for i in initial]),
            np.stack([i[0] for i in initial]),
            abs(self.dhfact),
            self.ode_rtol,
            self.ode_atol,
        )
        return [self._star(*args)[1] for args in zip(h_end, y_end, success)]

    def _star(self, h_end, y, success):
        """Euler step from the end of the integration to the surface, h = 0,
        then M, R, C and the Love numbers.

        Returns
        -------
        tuple
            (state at the surface, :meth:`solve` return value), or
            (None, nan values) if the integration stopped short.
        """
        if not (success and np.all(np.isfinite(y))):
            return None, self._pack(*(np.nan,) * 3)
        y = y - self.equations(h_end, y, self._tables, np) * h_end
        M, R, C = self.__compute_mass_radius(y)
        k = {
            l: self.__compute_Love_even(l, C, R * y[idH] / y[iH])
            for l, (iH, idH) in zip(self.leven, self._even_idx)
        }
        h = {
            l: self.__compute_shape(l, C, R * y[idH] / y[iH])
            for l, (iH, idH) in zip(self.leven, self._even_idx)
        }
        j = {
            l: self.__compute_Love_odd(l, C, R * y[idPsi] / y[iPsi])
            for l, (iPsi, idPsi) in zip(self.lodd, self._odd_idx)
        }
        return y, self._pack(M, R, C, k, h, j)

    def _pack(self, M, R, C, k=None, h=None, j=None):
        """The :meth:`solve` return value. (M, R, C), followed by (k, h) if there
        are even multipoles and j if there are odd ones. Love numbers not given
        are nan."""
        nan = float("nan")
        out = (M, R, C)
        if len(self.leven):
            k = k or {int(l): nan for l in self.leven}
            h = h or {int(l): nan for l in self.leven}
            out += (k, h)
        if len(self.lodd):
            out += (j or {int(l): nan for l in self.lodd},)
        return out

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
        if ell == 2 and c < 0.03:
            return _k2_small_compactness(c, y)
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
