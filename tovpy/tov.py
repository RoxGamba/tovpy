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

import sys, os, shutil
import numpy as np
from numpy import log, exp
import scipy as sp
from scipy.integrate import solve_ivp, odeint
from scipy.special import factorial2, gamma, factorial2, hyp2f1, poch
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, bisect
from math import comb, prod

from . import eos
from . import units
from .eos import EOS
from .units import Units
from .solvers import make_solver

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

    def __init__(self,
                 eos        = None, # EOS instance
                 leven      = [], # multipole indexes of even perturbations 
                 lodd       = [], # multipole indexes of odd perturbations 
                 dhfact     = -1e-12, # ODE step
                 ode_method = 'DOP853',
                 ode_atol   = 1e-6,
                 ode_rtol   = 1e-6,
                 ode_backend = 'scipy'): # ODE solver backend: 'scipy', 'numba', 'jax', or ODESolver instance

        if not eos:
            raise ValueError("Must provide a EOS")
        self.eos = eos

        leven = np.array(leven)
        lodd  = np.array(lodd)

        # Solve perturbation equations for these indexes
        self.leven = leven[leven>1]
        self.lodd  = lodd[lodd>1]

        # Build variable list
        var = self.__buildvars()
        self.nvar = len(var)
        self.var  = dict(zip(var, range(self.nvar)))
        self.ivar = {v: k for k, v in self.var.items()}

        # ODE solver options
        if dhfact > 0.:
            raise ValueError("ODE timestep must be negative")
        self.dhfact = dhfact
        # ode_method is only used by the scipy backend (forwarded to solve_ivp).
        # It is retained for backward compatibility but has no effect on other backends.
        self.ode_method = ode_method        
        self.ode_atol = ode_atol
        self.ode_rtol = ode_rtol

        # Instantiate the solver backend.  When ode_backend is 'scipy' the
        # ode_method parameter selects the integration method; for other
        # backends ode_method is ignored.
        if ode_backend == 'scipy':
            self.solver = make_solver('scipy', method=ode_method)
        else:
            self.solver = make_solver(ode_backend)

        # Cache integer indices for hot-path RHS (avoids dict lookups per step)
        self._i_r   = self.var['r']
        self._i_m   = self.var['m']
        self._i_nu  = self.var['nu']
        self._even_idx = [
            (self.var['H{}'.format(l)], self.var['dH{}'.format(l)])
            for l in self.leven
        ]
        self._odd_idx = [
            (self.var['Psi{}'.format(l)], self.var['dPsi{}'.format(l)])
            for l in self.lodd
        ]

        # Pre-build EOS tables for fast log-log interpolation in the RHS, and
        # bind the EOS evaluation wrapper (dispatched once here, not per step)
        self._build_eos_tables()

        # Bind perturbation update methods — no branching at RHS call time.
        # If no perturbations are requested the method is a no-op.
        self._update_even = (
            self._apply_even_perts if len(self.leven) else lambda dy, y, m, r, p, e, dedp, dr_dh, dnu_dr: None
        )
        self._update_odd = (
            self._apply_odd_perts if len(self.lodd) else lambda dy, y, m, r, p, e, dedp, dr_dh: None
        )

        
    def __buildvars(self):
        """
        List of varnames
        """
        v = ['r','m','nu']
        for l in self.leven:
            v.append('H{}'.format(l))
            v.append('dH{}'.format(l))
        for l in self.lodd:
            v.append('Psi{}'.format(l))
            v.append('dPsi{}'.format(l))
        return v

    def _build_eos_tables(self, n_points=2000):
        """
        Pre-sample the EOS once at construction and store log-space 1-D arrays
        for use in the fast RHS interpolation path.

        Tables stored (all 1-D, contiguous, float64):
          _log_h        : log(h) values (sorted ascending)
          _log_p        : log(p) values (same ordering as h)
          _log_e        : log(e) values
          _log_p_sorted : log(p) sorted ascending (key for dedp lookup)
          _log_dedp     : log(dedp) values, sorted by ascending log(p)

        Also binds ``self._eos_eval(h) -> (p, e, dedp)`` once, choosing the
        Numba JIT path or the ``np.interp`` path depending on availability.
        No branching occurs at RHS call time.
        """
        try:
            p_min = float(self.eos.p_min) * 1.001  # small buffer avoids interpolation boundary issues
            p_max = float(self.eos.p_max) * 0.999  # stay inside the EOS support range
        except AttributeError:
            p_min = 1e-19
            p_max = 1e-8

        p_arr    = np.logspace(np.log10(p_min), np.log10(p_max), n_points)
        h_arr    = np.array([self.eos.PseudoEnthalpy_Of_Pressure(p) for p in p_arr])
        e_arr    = np.array([self.eos.EnergyDensity_Of_Pressure(p)  for p in p_arr])
        dedp_arr = np.array([self.eos.EnergyDensityDeriv_Of_Pressure(p) for p in p_arr])

        valid = (h_arr > 0) & (e_arr > 0) & (dedp_arr > 0) & (p_arr > 0)
        h_arr    = h_arr[valid];    p_arr    = p_arr[valid]
        e_arr    = e_arr[valid];    dedp_arr = dedp_arr[valid]

        idx   = np.argsort(h_arr)
        h_arr = h_arr[idx];  p_arr = p_arr[idx]
        e_arr = e_arr[idx];  dedp_arr = dedp_arr[idx]

        self._log_h    = np.ascontiguousarray(np.log(h_arr))
        self._log_p    = np.ascontiguousarray(np.log(p_arr))
        self._log_e    = np.ascontiguousarray(np.log(e_arr))

        # dedp keyed by pressure (already sorted by h ≈ sorted by p)
        p_idx = np.argsort(p_arr)
        self._log_p_sorted = np.ascontiguousarray(np.log(p_arr[p_idx]))
        self._log_dedp     = np.ascontiguousarray(np.log(dedp_arr[p_idx]))

        # Bind the EOS evaluation function once — no if-else at RHS call time.
        # Capture table references in a closure so the resulting callable is a
        # plain function with no Python attribute access (forward-compatible with
        # JAX once the interpolation is ported to jnp operations).
        log_h = self._log_h;  log_p = self._log_p;  log_e = self._log_e
        log_p_sorted = self._log_p_sorted;  log_dedp = self._log_dedp

        if _NUMBA_AVAILABLE:
            # JIT-compiled O(log n) binary search; no Python overhead per call
            def _eos_eval(h):
                p    = _interp_positive(h, log_h, log_p)
                e    = _interp_positive(h, log_h, log_e)
                dedp = _interp_positive(p, log_p_sorted, log_dedp)
                return p, e, dedp
        else:
            # np.interp is C-implemented; equivalent speed without Numba
            def _eos_eval(h):
                lh   = np.log(h)
                p    = np.exp(np.interp(lh, log_h, log_p))
                e    = np.exp(np.interp(lh, log_h, log_e))
                dedp = np.exp(np.interp(np.log(p), log_p_sorted, log_dedp))
                return p, e, dedp

        self._eos_eval = _eos_eval

    
    def __pert_even(self,ell,m,r,p,e,dedp,dnu_dr=[]):
        """
        Eq.(27-29) of Damour & Nagar, Phys. Rev. D 80, 084035 (2009)
        https://arxiv.org/abs/0906.0096
        Note only C0 depends on ell: return an array of values
        """
        r2       = r**2
        r3       = r * r2
        div_r    = 1.0/r
        div_r2   = div_r**2
        exp_lam  = 1.0 / (1.0 - 2.0 * m * div_r ) # Eq. (18)
        if not dnu_dr:
            dnu2 = (2.0 * (m + 4.0 * np.pi * r3 * p) / (r * (r - 2.0 * m)))**2
        else:
            dnu2 = dnu_dr**2
        C1 = 2.0/r + exp_lam * ( 2*m*div_r2 + 4*np.pi*r*(p-e) ) 
        C0 = np.zeros(max(ell)+1)
        for l in ell:
            Lam = l*(l+1)
            C0[l] = -dnu2
            C0[l] += exp_lam * ( -Lam*div_r2 + 4*np.pi*( 5*e + 9*p + (e + p) * dedp ) ) 
        return C1, C0
                
    def __pert_odd(self,ell,m,r,p,e,dedp):
        """
        Eq.(31) of Damour & Nagar, Phys. Rev. D 80, 084035 (2009)
        https://arxiv.org/abs/0906.0096
        Note only C0 depends on ell: return an array of values
        """
        r2 = r**2
        r3 = r * r2
        div_r = 1.0/r
        div_r2 = div_r**2
        div_r3 = div_r*div_r2
        exp_lam = 1.0 / (1.0 - 2.0 * m * div_r ) # Eq. (18)
        C1 = exp_lam * ( 2*m + 4*np.pi*r3*(p-e) ) * div_r2
        C0 = np.zeros(max(ell)+1)
        for l in ell:
            Lam = l*(l+1)
            C0[l] = exp_lam*( -Lam*div_r2 + 6*m*div_r3 - 4*np.pi*(e-p) )
        return C1, C0

    def _apply_even_perts(self, dy, y, m, r, p, e, dedp, dr_dh, dnu_dr):
        """Apply even-parity perturbation equations to derivative vector *dy*."""
        C1, C0 = self.__pert_even(self.leven, m, r, p, e, dedp, dnu_dr)
        for l, (iH, idH) in zip(self.leven, self._even_idx):
            H  = y[iH]
            dH = y[idH]
            dy[iH]  = dH * dr_dh
            dy[idH] = -(C0[l] * H + C1 * dH) * dr_dh

    def _apply_odd_perts(self, dy, y, m, r, p, e, dedp, dr_dh):
        """Apply odd-parity perturbation equations to derivative vector *dy*."""
        C1, C0 = self.__pert_odd(self.lodd, m, r, p, e, dedp)
        for l, (iPsi, idPsi) in zip(self.lodd, self._odd_idx):
            Psi  = y[iPsi]
            dPsi = y[idPsi]
            dy[iPsi]  = dPsi * dr_dh
            dy[idPsi] = -(C0[l] * Psi + C1 * dPsi) * dr_dh

    def __tov_rhs(self, h, y):
        """
        ODE r.h.s. for TOV equations with pseudo-enthalpy independent variable.
        Implements Eqs. (5) and (6) of Lindblom, Astrophys. J. 398, 569 (1992).
        Also uses Eqs. (7) and (8) [ibid] for inner boundary data, and
        Eqs. (18), (27), (28) of Damour & Nagar, Phys. Rev. D 80, 084035 (2009)
        for the metric perturbation used to obtain the Love number.

        Uses cached integer indices and the pre-bound ``self._eos_eval`` wrapper
        (JIT or ``np.interp``, selected once at construction).
        ``self._update_even`` and ``self._update_odd`` are similarly pre-bound
        to the actual perturbation methods or no-ops, so this function contains
        no branching regardless of the active configuration.

        .. note:: **JAX compatibility**
            Making this RHS JAX-traceable requires restructuring it as a
            standalone pure function (no ``self`` capture, no Python attribute
            access).  The ``_eos_eval`` closure already captures only array
            data and is JAX-portable once its internals are ported to
            ``jnp`` operations.  The ``JaxSolver`` stub documents the remaining
            requirements.
        """
        dy = np.zeros_like(y)
        r = y[self._i_r]
        m = y[self._i_m]
        p, e, dedp = self._eos_eval(h)
        dr_dh  = -r * (r - 2.0 * m) / (m + 4.0 * np.pi * r**3 * p)
        dm_dh  =  4.0 * np.pi * r**2 * e * dr_dh
        dnu_dr =  2.0 * (m + 4.0 * np.pi * r**3 * p) / (r * (r - 2.0 * m))
        dy[self._i_r]  = dr_dh
        dy[self._i_m]  = dm_dh
        dy[self._i_nu] = dnu_dr * dr_dh
        self._update_even(dy, y, m, r, p, e, dedp, dr_dh, dnu_dr)
        self._update_odd(dy, y, m, r, p, e, dedp, dr_dh)
        return dy

    def __initial_data(self,pc,dh_fact=-1e-12,verbose=False):
        """
        Set initial data for the solution of TOV equations using the pseudo-enthalpy formalism introduced in:
        Lindblom (1992) "Determining the Nuclear Equation of State from Neutron-Star Masses and Radii", Astrophys. J. 398 569.
        * input the central pressure
        """
        y = np.zeros(self.nvar)
        # Central values 
        ec     = self.eos.EnergyDensity_Of_Pressure(pc)
        hc     = self.eos.PseudoEnthalpy_Of_Pressure(pc)
        dedp_c = self.eos.EnergyDensityDeriv_Of_Pressure(pc)
        dhdp_c = 1.0 / (ec + pc)
        dedh_c = dedp_c / dhdp_c
        dh = -1e-12 * hc
        h0 = hc + dh
        h1 = 0.0 - dh
        r0 = np.sqrt(-3.0 * dh / (2.0 * np.pi * (ec + 3.0 * pc)))
        m0 = 4.0 * np.pi * r0**3 * ec / 3.0
        # Series expansion for the initial core 
        r0 *= 1.0 + 0.25 * dh * (ec - 3.0 * pc  - 0.6 * dedh_c) / (ec + 3.0 * pc) # second factor Eq. (7) of Lindblom (1992) 
        m0 *= 1.0 + 0.6 * dh * dedh_c / ec # second factor of Eq. (8) of Lindblom (1992) 
        y[self._i_r]  = r0
        y[self._i_m]  = m0
        y[self._i_nu] = 0.0
        #  Initial data for the ell-perturbation
        a0 = 1.0
        if len(self.leven)!= 0:
            for l, (iH, idH) in zip(self.leven, self._even_idx):
                y[iH]  = a0 * r0**l
                y[idH] = a0 * l * r0**(l-1)
        if len(self.lodd)!= 0:
            for l, (iPsi, idPsi) in zip(self.lodd, self._odd_idx):
                y[iPsi]  = a0 * r0**(l+1)
                y[idPsi] = a0 * (l+1) * r0**l
        # if verbose:
        #     print("pc = {:.8e} hc = {:.8e} dh = {:.8e} h0  = {:.8e}".format(pc,hc,dh,h0))
        #     print(y, self.ivar)
        return y, h0, h1
    
    def solve(self,pc):
        """
        Solves the Tolman-Oppenheimer-Volkov stellar structure equations using the pseudo-enthalpy formalism introduced in:
        Lindblom (1992) "Determining the Nuclear Equation of State from Neutron-Star Masses and Radii", Astrophys. J. 398 569.
        """
        # Initial data
        y, h0, h1 = self.__initial_data(pc, dh_fact=self.dhfact, verbose=True)
        # Integrate
        # print("Integrating TOV equations")
        # print("h0 = {:.8e} h1 = {:.8e}".format(h0,h1))
        sol = self.solver.solve(self.__tov_rhs, [h0, h1], y,
                        first_step = abs(self.dhfact),
                        rtol = self.ode_rtol,
                        atol = self.ode_atol)
    
        # Take one final Euler step to get to surface 
        y  = sol.y[:,-1]
        dy = self.__tov_rhs(sol.t[-1],y)
        y[:] -= dy[:] * h1
        np.append(sol.y, y)
        # Mass, Radius & Compactness
        M,R,C = self.__compute_mass_radius(y)
        # Match to Schwarzschild exterior
        sol.y[self._i_nu, :] += np.log(1.0-(2.*M)/R) - sol.y[self._i_nu, -1]

        self.sol = sol
        if len(self.leven) != 0:
            k, h = {}, {}
            for l, (iH, idH) in zip(self.leven, self._even_idx):
                yyl = R * y[idH] / y[iH]
                k[l] = self.__compute_Love_even(l,C,yyl)
                h[l] = self.__compute_shape(l,C,yyl)
        if len(self.lodd) != 0:
            # Odd Love numbers
            j = {}
            for l, (iPsi, idPsi) in zip(self.lodd, self._odd_idx):
                yyl = R * y[idPsi] / y[iPsi]
                j[l] = self.__compute_Love_odd(l,C,yyl)

        if len(self.leven)!= 0 and len(self.lodd)!= 0:
            return M,R,C,k,h,j
        elif len(self.leven)!= 0 and len(self.lodd)== 0:
            return M,R,C,k,h
        elif len(self.leven)== 0 and len(self.lodd)!= 0:
            return M,R,C,j
        else:
            return M,R,C

    def __compute_legendre(self, c, l):
        """
        Computes Legendre function values returning Pl2(x), Ql2(x) and their derivatives at x = 1/c -1
        """
        x = 1/c -1
        L = np.linspace(0,l-1,l)
        nP = -prod((2*l-1)/2-L)/gamma(l) * 2**l * l*(l-1)
        nQ = gamma(l)/factorial2(2*l+1)*(l+1)*(l+2)

        Pl2 = 0
        dPl2 = 0
        for i in np.linspace(2,l,l-2+1,dtype=int):
            Pl2 = Pl2 + gamma(i)/gamma(i-2) * comb(l,i) * prod((l+i-1)/2-L) / gamma(l) * x**(i-2)
            dPl2 = dPl2 + gamma(i)/gamma(i-2) * comb(l,i) * prod((l+i-1)/2-L) / gamma(l) * (i-2) * x**(i-3)
        
        dPl2 = 2**l*(-2*x)*Pl2/nP + 2**l*(1-x**2)*dPl2/nP
        Pl2  = 2**l*(1-x**2)*Pl2/nP

        Ql2  = 1/nQ * np.sqrt(np.pi)/2**(l+1) * gamma(l+3)/gamma(l+3/2) * (x**2-1)/x**(l+3) * hyp2f1((l+3)/2, (l+4)/2,l+3/2,1/x**2)
        dQl2 = 1/nQ * np.sqrt(np.pi)/2**(l+1) * gamma(l+3)/gamma(l+3/2) * (2*x**(-2 - l)*hyp2f1((l+3)/2, (l+4)/2,l+3/2,1/x**2) +\
                                                            (-3 - l)*x**(-4 - l)*(-1 + x**2)*hyp2f1((l+3)/2, (l+4)/2,l+3/2,1/x**2) -\
                                                            (2*((l+3)/2)*((l+4)/2)*x**(-6 - l)*(-1 + x**2)*hyp2f1((l+3)/2+1, (l+4)/2+1,l+3/2+1,1/x**2)/(l+3/2)))
        return Pl2,dPl2,Ql2,dQl2
    
    def __compute_psi(self, c, l):
        x = 1/c
        CoefficientP = poch(5, l-2) / poch (2-l, l-2) / poch(3+l, l-2) * gamma(l-2) * 2 ** (l-2)
        CoefficientQ = -1 / (l+2)
        psiP = x**3 * hyp2f1(2-l, 3+l, 5, x/2) * CoefficientP
        psiQ = - (l+2) * x**(-1-l) * ((1+l) * x * hyp2f1(-1+l,2+l,2+2*l,2/x) + (-1+l)*hyp2f1(l,3+l,3+2*l,2/x) )/(1+l) * CoefficientQ
        dPsiP = 3 * x**2 * hyp2f1(2-l, 3+l, 5, x/2) - 1/10 * (-6 + l + l**2) * x**3 * hyp2f1(3-l, 4+l, 6, x/2)
        dPsiP = dPsiP * CoefficientP
        dPsiQ = 1/(1+l)/(3+2*l) * (2+l) * x**(-3-l) * (
            l*(3+5*l+2*l**2)*x**2*hyp2f1(-1+l, 2+l, 2+2*l, 2/x) +
            (-1+l)*(
                (3+2*l)**2*x*hyp2f1(l, 3+l, 3+2*l, 2/x) +
                2*l*(3+l)*hyp2f1(1+l, 4+l, 4+2*l, 2/x)
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
        return M,R,M/R

    def Compute_baryon_mass(self, sol):
        """
        Compute baryon mass
        """
        r = sol.y[self._i_r, :]
        m = sol.y[self._i_m, :]
        # e = self.EOSEnergyDensityOfPseudoEnthalpyGeometerized(sol.t,self.eos)
        e = np.array([self.eos.EnergyDensity_Of_PseudoEnthalpy(sol.t[i]) for i in range(len(sol.t))])
        return np.trapz( 4*np.pi*r**2.*e/np.sqrt(1-2*m/r), r )

    def Compute_proper_radius(self, sol):
        """
        Compute baryon mass
        """
        r = sol.y[self._i_r, :]
        m = sol.y[self._i_m, :]
        return np.trapz( r, 1./np.sqrt((1-2*m/r)), r )
        
    def __compute_Love_odd(self,ell,c,y):
        """
        Compute odd parity Love numbers given 
        * the multipolar index ell
        * the compactness c
        * the ratio y = R Psi(R)'/Psi(R) 
        Eq.(61) of Damour & Nagar, Phys. Rev. D 80 084035 (2009)
        """
        c2 = c**2
        c3 = c*c2
        c4 = c*c3
        c5 = c*c4
        j = 0.
        if ell == 2:
            nj =  96*c5*(-1 + 2*c)*(-3 + y)
            dj =  5.*(2*c*(9 + 3*c*(-3 + y) + 2*c2*(-3 + y) + 2*c3*(-3 + y) - 3*y + 12*c4*(1 + y)) + 3*(-1 + 2*c)*(-3 + y)*log(1 - 2*c))
            j = nj/dj
        else:
            PsiP, dPsiP, PsiQ, dPsiQ = self.__compute_psi(c,ell)
            factor =  - c ** (2 * ell + 1)
            j = factor * (dPsiP - c * y * PsiP) / (dPsiQ - c * y * PsiQ)
        return j
    
    def __compute_Love_even(self,ell,c,y):
        """
        Compute even parity Love numbers given 
        * the multipolar index ell
        * the compactness c
        * the ratio y = R H(R)'/H(R) 
        Eq.(49) of Damour & Nagar, Phys. Rev. D 80 084035 (2009)
        """
        c2 = c**2
        c3 = c*c2
        c4 = c*c3
        c5 = c*c4
        c6 = c*c5
        c7 = c*c6
        c8 = c*c7
        c9 = c*c8
        c10 = c*c9
        c11 = c*c10
        c13 = c2*c11
        c15 = c2*c13
        c17 = c2*c15
        k = 0.
        if ell < 2: return k
        if ell == 2:
            nk = (1-2*c)**2*(2+2*c*(y-1)-y)
            dk = 2*c*(6-3*y+3*c*(5*y-8))+4*c3*(13-11*y+c*(3*y-2)+2*c2*(1+y)) + 3*(1-2*c)**2*(2-y+2*c*(y-1))*np.log(1-2*c)
            k = 8*c5/5*nk/dk
        elif ell == 3:
            nk = (1 - 2*c)**2*(-3 - 3*c*(-2 + y) + 2*c2*(-1 + y) + y)
            dk = 2*c*(15*(-3 + y) + 4*c5*(1 + y) - 45*c*(-5 + 2*y) - 20*c3*(-9 + 7*y) + 2*c4*(-2 + 9*y) + 5*c2*(-72 + 37*y)) - 15*(1 - 2*c)**2*(-3 - 3*c*(-2 + y) + 2*c**2*(-1 + y) + y)*np.log(1.0/(1 - 2*c))
            k = 8*c7/7*nk/dk
        elif ell == 4:
            nk = (1 - 2*c)**2*(-7*(-4 + y) + 28*c*(-3 + y) - 34*c2*(-2 + y) + 12*c3*(-1 + y))
            dk = (2*c*(c2*(5360 - 1910*y) + c4*(1284 - 996*y) - 105*(-4 + y) + 8*c6*(1 + y) + 105*c*(-24 + 7*y) + 40*c3*(-116 + 55*y) + c5*(-8 + 68*y)) - 15*(1 - 2*c)**2*(-7*(-4 + y) + 28*c*(-3 + y) - 34*c2*(-2 + y) + 12*c3*(-1 + y))*np.log(1.0/(1 - 2*c)))
            k = 32*c9/147*nk/dk
        elif ell == 5:
            nk = (32*(1 - 2*c)**2*c11*(3*(-5 + y) - 15*c*(-4 + y) + 26*c2*(-3 + y) - 18*c3*(-2 + y) + 4*c4*(-1 + y)))
            dk = 99.*(2*c*(315*(-5 + y) + 8*c7*(1 + y) - 315*c*(-35 + 8*y) + 4*c6*(-2 + 27*y) - 56*c5*(-60 + 47*y) - 210*c3*(-170 + 57*y) + 105*c2*(-278 + 75*y) + 56*c4*(-345 + 158*y)) - 105*(1 - 2*c)**2*(3*(-5 + y) - 15*c*(-4 + y) + 26*c2*(-3 + y) - 18*c3*(-2 + y) + 4*c4*(-1 + y))*np.log(1.0/(1 - 2*c)))
            k = nk/dk
        elif ell == 6:
            nk = (1024*(1 - 2*c)**2*c13*(-33*(-6 + y) + 198*c*(-5 + y) - 444*c2*(-4 + y) + 456*c3*(-3 + y) - 208*c4*(-2 + y) + 32*c5*(-1 + y)))
            dk = 14157.*(2*c*(-3465*(-6 + y) + 32*c8*(1 + y) + 10395*c*(-16 + 3*y) + 16*c7*(-2 + 39*y) + 2016*c5*(-122 + 55*y) - 64*c6*(-457 + 362*y) - 210*c2*(-2505 + 541*y)  + 210*c3*(-3942 + 1015*y) - 84*c4*(-7917 + 2567*y)) - 105*(1 - 2*c)**2*(-33*(-6 + y) + 198*c*(-5 + y) - 444*c2*(-4 + y) + 456*c3*(-3 + y) - 208*c4*(-2 + y) + 32*c5*(-1 + y))*np.log(1.0/(1 - 2*c)))
            k = nk/dk
        elif ell == 7:
            nk = 1024*(1 - 2*c)**2*c15*(143*(-7 + y) - 1001*c*(-6 + y) + 2750*c2*(-5 + y) - 3740*c3*(-4 + y) + 2600*c4*(-3 + y) - 848*c5*(-2 + y) + 96*c6*(-1 + y))
            dk = 20449.*(2*c*(45045*(-7 + y) + 160*c9*(1 + y) - 45045*c*(-63 + 10*y) + 80*c8*(-2 + 53*y) - 432*c7*(-651 + 521*y) - 4620*c3*(-4333 + 902*y)  + 1155*c2*(-9028 + 1621*y) + 96*c6*(-33964 + 15203*y) + 126*c4*(-168858 + 42239*y) - 84*c5*(-144545 + 45971*y)) - 315*(1 - 2*c)**2*(143*(-7 + y) - 1001*c*(-6 + y) + 2750*c2*(-5 + y) - 3740*c3*(-4 + y) + 2600*c4*(-3 + y) - 848*c5*(-2 + y) + 96*c6*(-1 + y))*np.log(1.0/(1 - 2*c)))
            k = nk/dk
        elif ell == 8:
            nk = (16*(1 - 2*c)**2*(2*c*(c*(2*c*(2*c*(-737*(-4 + y) + 374*c*(-3 + y) - 92*c2*(-2 + y) + 8*c**3*(-1 + y)) + 1573*(-5 + y)) - 1859*(-6 + y)) + 572*(-7 + y)) - 143*(-8 + y)))
            dk = (286*c*(4*(90090 + c*(-900900 + c*(3768765 + c*(-8528520 + c*(11259633 + 2*c*(-4349499 + c*(1858341 + 8*c*(-47328 + c*(3092 + (-1 + c)*c))))))))) + (-1 + c)*(-1 + 2*c)*(-45045 + 4*c*(90090 + c*(-285285 + c*(450450 + c*(-365211 + 4*c*(34881 + c*(-4887 + 2*c*(36 + c))))))))*y) - 45045*(1 - 2*c)**2*(2*c*(c*(2*c*(2*c*(-737*(-4 + y) + 374*c*(-3 + y) - 92*c2*(-2 + y) + 8*c**3*(-1 + y)) + 1573*(-5 + y)) - 1859*(-6 + y)) + 572*(-7 + y)) - 143*(-8 + y))*np.log(1.0/(1 - 2*c)))
            k = 256/2431 * c17 * nk/dk
        else:
            # https://bitbucket.org/bernuzzi/tov/src/master/ComputeLegendre.m
            Pl2, dPl2, Ql2, dQl2 = self.__compute_legendre(c, ell)
            k = -1/2*c**(2*ell+1)*(dPl2-c*y*Pl2)/(dQl2-c*y*Ql2)
        return k
        
    def __compute_shape(self,ell,c,y):
        """
        Compute even shape numbers given 
        * the multipolar index ell
        * the compactness c
        * the ratio y = R H(R)'/H(R) 
        Eq.(95) of Damour & Nagar, Phys. Rev. D 80 084035 (2009)
        """
        c2 = c**2
        c3 = c*c2
        c4 = c*c3
        c5 = c*c4
        c6 = c*c5
        c7 = c*c6
        c8 = c*c7
        c9 = c*c8
        c10 = c*c9
        c11 = c*c10
        c13 = c2*c11
        c15 = c2*c13
        c17 = c2*c15
        h = 0.
        if ell < 2: return h
        if ell == 2:
            nh = (-2 + 6*c + 2*c3*(1 + y) - c2*(6 + y))
            dh = (2*c*(6 + c2*(26 - 22*y) - 3*y + 4*c4*(1 + y) + 3*c*(-8 + 5*y) + c3*(-4 + 6*y)) - 3*(1 - 2*c)**2*(2 + 2*c*(-1 + y) - y)*np.log(1.0/(1 - 2*c)))
            h = -8*c5*nh/dh
        # elif ell == 3:
        #     nh = -5 + 15*c + 2*c3*(1 + y) - c2*(12 + y)
        #     dh = (5.*(2*c*(15*(-3 + y) + 4*c5*(1 + y) - 45*c*(-5 + 2*y) - 20*c3*(-9 + 7*y) + 2*c4*(-2 + 9*y) + 5*c2*(-72 + 37*y)) - 15*(1 - 2*c)**2*(-3 - 3*c*(-2 + y) + 2*c2*(-1 + y) + y)*np.log(1.0/(1 - 2*c))))
        #     h = 16*c7*nh/dh
        # elif ell == 4:
        #     nh = -9 + 27*c + 2*c3*(1 + y) - c2*(20 + y)
        #     dh = (21.*(2*c*(c2*(5360 - 1910*y) + c4*(1284 - 996*y) - 105*(-4 + y) + 8*c6*(1 + y) + 105*c*(-24 + 7*y)  + 40*c3*(-116 + 55*y) + c5*(-8 + 68*y)) - 15*(1 - 2*c)**2*(-7*(-4 + y) + 28*c*(-3 + y) - 34*c2*(-2 + y) + 12*c3*(-1 + y))*np.log(1.0/(1 - 2*c))))
        #     h = -64*c9*nh/dh
        else:
            Pl2, dPl2, Ql2, dQl2 = self.__compute_legendre(c,ell)
            term1 = (1-2*c)/c
            term2 = 1/(ell-1)/(ell+2) * (2*c*y + ell*(ell+1) + 4*c**2/(1-2*c) - 2*(1-2*c))
            factor = c**(ell+1)*Pl2 * (1-(dPl2/Pl2-c*y)/(dQl2/Ql2-c*y))
            h = (term1 + term2) * factor
        return h
    
    def Compute_Lambda(self,ell,k,C):
        r"""
        Compute tidal polarizability $\Lambda_\ell$
        from Love numbers and compactness
        Note: Yagi's $\bar{\lambda}_\ell$ is $\Lambda_\ell$
        """
        div = 1.0/(factorial2(2*ell-1)*C**(2*ell+1))
        return 2.*k*div


