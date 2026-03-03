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

"""ODE solver backends for the TOV solver.

Provides a pluggable solver interface supporting:

  - ``'scipy'`` : standard :func:`scipy.integrate.solve_ivp` (default)
  - ``'numba'`` : explicit adaptive RK45 integrator, structured to allow
                  future numba JIT compilation once the EOS layer is
                  numba-compatible.
  - ``'jax'``   : JAX-based solver (future perspective, not yet implemented).

Usage example::

    from tovpy.tov import TOV
    from tovpy.solvers import make_solver

    tov = TOV(eos=my_eos, ode_backend='scipy')          # default
    tov = TOV(eos=my_eos, ode_backend='numba')          # numba backend
    tov = TOV(eos=my_eos, ode_backend=make_solver('scipy', method='RK45'))
"""

from abc import ABC, abstractmethod
import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class SolverResult:
    """Lightweight result container with the same ``.t`` / ``.y`` interface
    as :class:`scipy.integrate.OdeSolution`.

    Parameters
    ----------
    t : array_like, shape (n,)
        Time / independent-variable values.
    y : array_like, shape (nvar, n)
        Solution values; ``y[:, i]`` is the state at ``t[i]``.
    """

    def __init__(self, t, y):
        self.t = np.asarray(t)
        self.y = np.asarray(y)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class ODESolver(ABC):
    """Abstract base class for TOV ODE solver backends.

    All concrete backends must implement :meth:`solve`.
    """

    @abstractmethod
    def solve(self, rhs, t_span, y0, first_step=None, rtol=1e-9, atol=1e-9, **kwargs):
        """Integrate ``dy/dt = rhs(t, y)`` from ``t_span[0]`` to ``t_span[1]``.

        Parameters
        ----------
        rhs : callable
            RHS function with signature ``rhs(t, y) -> array_like``.
        t_span : (float, float)
            Integration interval ``(t0, t1)``.
        y0 : array_like
            Initial state vector.
        first_step : float, optional
            Magnitude of the initial step size.
        rtol : float
            Relative tolerance.
        atol : float
            Absolute tolerance.
        **kwargs
            Extra keyword arguments (ignored by backends that do not support them).

        Returns
        -------
        result : :class:`SolverResult` or scipy OdeSolution
            Object with attributes ``.t`` (shape ``(n,)``) and ``.y``
            (shape ``(nvar, n)``).
        """


# ---------------------------------------------------------------------------
# Scipy backend
# ---------------------------------------------------------------------------

class ScipySolver(ODESolver):
    """ODE solver backend using :func:`scipy.integrate.solve_ivp`.

    Parameters
    ----------
    method : str
        Integration method forwarded to ``solve_ivp`` (default ``'DOP853'``).
    """

    def __init__(self, method='DOP853'):
        self.method = method

    def solve(self, rhs, t_span, y0, first_step=None, rtol=1e-9, atol=1e-9, **kwargs):
        from scipy.integrate import solve_ivp
        return solve_ivp(
            rhs, t_span, y0,
            first_step=first_step,
            method=self.method,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Numba backend
# ---------------------------------------------------------------------------

class NumbaSolver(ODESolver):
    """ODE solver backend using an explicit adaptive RK45 integrator.

    The integration loop is implemented in plain NumPy so that it can be
    JIT-compiled with numba once the EOS layer is made numba-compatible.
    Even without JIT compilation this backend is independent of scipy and
    serves as the architectural foundation for the numba acceleration path.

    Parameters
    ----------
    max_steps : int
        Maximum number of integration steps (default ``100_000``).
    """

    def __init__(self, max_steps=100_000):
        self.max_steps = max_steps

    def solve(self, rhs, t_span, y0, first_step=None, rtol=1e-9, atol=1e-9, **kwargs):
        y0 = np.asarray(y0, dtype=float)
        t0, t1 = float(t_span[0]), float(t_span[1])
        h0 = float(first_step) if first_step is not None else abs(t1 - t0) * 1e-3
        t_vals, y_vals = _rk45_adaptive(rhs, t0, t1, y0, h0, self.max_steps, rtol, atol)
        return SolverResult(np.array(t_vals), np.array(y_vals).T)


def _rk45_adaptive(rhs, t0, t1, y0, h0, max_steps, rtol, atol):
    """Dormand-Prince RK45 adaptive-step integrator.

    This pure-NumPy implementation mirrors the logic of
    :func:`scipy.integrate.solve_ivp` with ``method='RK45'`` and is
    structured to be JIT-compiled with numba in the future.

    Returns lists ``(t_vals, y_vals)`` where ``y_vals[i]`` is the state at
    ``t_vals[i]``.
    """
    # Dormand-Prince RK45 Butcher tableau (pre-computed float constants)
    C2 = 0.2
    C3 = 0.3
    C4 = 0.8
    C5 = 0.8888888888888888    # 8/9
    A21 = 0.2
    A31 = 0.075;               A32 = 0.225
    A41 = 0.9777777777777778;  A42 = -3.7333333333333334; A43 = 3.5555555555555554
    A51 = 2.9525986892242035;  A52 = -11.595793324188385; A53 = 9.822892851699436;  A54 = -0.2908093278463649
    A61 = 2.8462752525252526;  A62 = -10.757575757575758; A63 = 8.906422717743473;  A64 = 0.2784090909090909; A65 = -0.2735313036020583
    # Error coefficients (difference between 5th and 4th-order weights)
    E1 = 0.0012326388888888888;   E3 = -0.0042527702905061394
    E4 = 0.036979166666666667;    E5 = -0.050863797169811321
    E6 = 0.041904761904761905;    E7 = -0.025

    sign = 1.0 if t1 > t0 else -1.0
    h = sign * abs(h0)
    t = t0
    y = y0.copy()

    t_vals = [t]
    y_vals = [y.copy()]

    for _ in range(max_steps):
        if sign * (t - t1) >= 0.0:
            break

        # Clamp step to not overshoot
        if sign * (t + h - t1) > 0.0:
            h = t1 - t

        k1 = np.asarray(rhs(t,              y))
        k2 = np.asarray(rhs(t + C2*h,       y + h*A21*k1))
        k3 = np.asarray(rhs(t + C3*h,       y + h*(A31*k1 + A32*k2)))
        k4 = np.asarray(rhs(t + C4*h,       y + h*(A41*k1 + A42*k2 + A43*k3)))
        k5 = np.asarray(rhs(t + C5*h,       y + h*(A51*k1 + A52*k2 + A53*k3 + A54*k4)))
        k6 = np.asarray(rhs(t + h,          y + h*(A61*k1 + A62*k2 + A63*k3 + A64*k4 + A65*k5)))

        # 5th-order solution
        y_new = y + h*(35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)

        k7 = np.asarray(rhs(t + h, y_new))

        # Error estimate (difference between 4th and 5th order)
        err_vec = h*(E1*k1 + E3*k3 + E4*k4 + E5*k5 + E6*k6 + E7*k7)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err = np.linalg.norm(err_vec / scale) / np.sqrt(len(scale))

        if err <= 1.0:
            # Accept step
            t = t + h
            y = y_new
            t_vals.append(t)
            y_vals.append(y.copy())

        # Adaptive step-size update (standard formula)
        if err == 0.0:
            factor = 5.0
        else:
            factor = min(5.0, max(0.2, 0.9 * err**(-0.2)))
        h = h * factor

    return t_vals, y_vals


# ---------------------------------------------------------------------------
# JAX backend (stub)
# ---------------------------------------------------------------------------

class JaxSolver(ODESolver):
    """ODE solver backend using JAX (future perspective).

    .. note::
        This backend is not yet implemented.  It is provided as an
        architectural stub to guide future development.

        **Requirements for a working JAX implementation:**

        1. **Pure-function RHS** — JAX tracing cannot capture Python objects
           (``self``).  The TOV RHS must be restructured as a standalone
           function whose closed-over data consists only of JAX arrays
           (e.g. the pre-built EOS log-tables already stored as plain NumPy
           arrays in ``_eos_eval``).

        2. **JAX-compatible EOS evaluation** — replace ``np.interp`` /
           ``_interp_positive`` with ``jnp.interp`` or an equivalent
           pure-JAX interpolation.  The ``_eos_eval`` closure in ``TOV`` is
           already structured for this: once its body is ported to ``jnp``
           operations the same closure pattern works under JAX.

        3. **ODE integrator** — use a JAX-native integrator such as
           ``diffrax`` (``diffrax.diffeqsolve``) or a custom ``jax.lax.while_loop``
           based implementation.

        4. **No Python control flow on traced values** — all branching in
           the RHS must be static (compile-time constants), which is already
           the case in the current design (even/odd perturbation updates are
           pre-bound at construction, not evaluated inside the hot loop).

        Contributions are welcome!
    """

    def solve(self, rhs, t_span, y0, first_step=None, rtol=1e-9, atol=1e-9, **kwargs):
        raise NotImplementedError(
            "The JAX ODE backend is not yet implemented. "
            "See the JaxSolver docstring for the requirements. "
            "Contributions are welcome!"
        )


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

_SOLVER_REGISTRY = {
    'scipy': ScipySolver,
    'numba': NumbaSolver,
    'jax':   JaxSolver,
}


def make_solver(backend, **kwargs):
    """Return an :class:`ODESolver` instance for the requested *backend*.

    Parameters
    ----------
    backend : str or :class:`ODESolver`
        One of ``'scipy'``, ``'numba'``, ``'jax'``, or an already-instantiated
        :class:`ODESolver`.
    **kwargs
        Additional keyword arguments forwarded to the solver constructor
        (e.g. ``method='RK45'`` for the scipy backend).

    Returns
    -------
    ODESolver
    """
    if isinstance(backend, ODESolver):
        return backend
    if backend not in _SOLVER_REGISTRY:
        raise ValueError(
            f"Unknown ODE backend {backend!r}. "
            f"Valid options are: {list(_SOLVER_REGISTRY)}"
        )
    return _SOLVER_REGISTRY[backend](**kwargs)
