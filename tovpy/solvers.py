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

Every backend integrates ``dy/dt = rhs(t, y, args, xp)``, with ``xp`` the
array module it runs on, ``numpy`` or ``jax.numpy``. Equations written against
``xp`` therefore run on either backend.

  - ``'scipy'``, :func:`scipy.integrate.solve_ivp` (default).
  - ``'jax'``, diffrax, compiled by JAX. Needs ``pip install "jax[cpu]" diffrax``.

Both report whether each integration reached its end point, and neither returns
a failed integration as a success.

Usage example::

    from tovpy.tov import TOV
    from tovpy.solvers import make_solver

    tov = TOV(eos=my_eos, ode_backend='scipy')          # default
    tov = TOV(eos=my_eos, ode_backend='jax')            # JAX/diffrax backend
    tov = TOV(eos=my_eos, ode_backend=make_solver('scipy', method='RK45'))
"""

from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np


class SolverResult:
    """Trajectory of one integration, with the ``.t`` / ``.y`` interface of
    :class:`scipy.integrate.OdeSolution`.

    Parameters
    ----------
    t : array_like, shape (n,)
    y : array_like, shape (nvar, n)
        ``y[:, i]`` is the state at ``t[i]``.
    """

    def __init__(self, t, y):
        self.t = np.asarray(t)
        self.y = np.asarray(y)


class ODESolver(ABC):
    """Common interface of the ODE backends.

    ``rhs(t, y, args, xp)`` must be a hashable callable, e.g. a frozen
    dataclass, so that the JAX backend reuses one compiled kernel for every
    call with the same equations. ``args`` holds the data the equations read
    (e.g. EOS tables) and may change from call to call.
    """

    @abstractmethod
    def integrate(
        self, rhs, args, t0, t1, y0, first_step, rtol, atol, trajectory=False
    ):
        """Integrate one state from ``t0`` to ``t1``.

        Returns
        -------
        tuple
            ``(t_end, y_end, success, trajectory)``. ``success`` is True only if
            the integration reached ``t1``; ``trajectory`` is a
            :class:`SolverResult` if requested, else None.
        """

    def integrate_many(self, rhs, args, t0, t1, y0, first_step, rtol, atol):
        """Integrate n states with the same ``rhs`` and ``args``. ``t0`` and
        ``t1`` have shape (n,), ``y0`` has shape (n, nvar).

        Returns
        -------
        tuple of ndarray
            ``(t_end, y_end, success)``, of shapes (n,), (n, nvar), (n,).
        """
        out = [
            self.integrate(rhs, args, a, b, y, first_step, rtol, atol)[:3]
            for a, b, y in zip(t0, t1, y0)
        ]
        t_end, y_end, success = zip(*out)
        return np.array(t_end), np.array(y_end), np.array(success)


class ScipySolver(ODESolver):
    """Backend using :func:`scipy.integrate.solve_ivp`.

    Parameters
    ----------
    method : str
        Any ``solve_ivp`` method (default ``'DOP853'``).
    """

    def __init__(self, method="DOP853"):
        self.method = method

    def integrate(
        self, rhs, args, t0, t1, y0, first_step, rtol, atol, trajectory=False
    ):
        from scipy.integrate import solve_ivp

        sol = solve_ivp(
            lambda t, y: rhs(t, y, args, np),
            (t0, t1),
            np.asarray(y0, dtype=float),
            method=self.method,
            first_step=first_step,
            rtol=rtol,
            atol=atol,
        )
        success = bool(sol.success) and sol.t[-1] == t1
        traj = SolverResult(sol.t, sol.y) if trajectory else None
        return float(sol.t[-1]), sol.y[:, -1].copy(), success, traj


# The diffrax solver behind each method name. A scipy name selects the diffrax
# method of the same order.
_JAX_METHODS = {
    "Dopri5": "Dopri5",
    "Dopri8": "Dopri8",
    "Tsit5": "Tsit5",
    "Kvaerno5": "Kvaerno5",
    "RK45": "Dopri5",
    "DOP853": "Dopri8",
    "Radau": "Kvaerno5",
}


class JaxSolver(ODESolver):
    """Backend using diffrax, compiled by JAX.

    Compiles the equations once per ``(rhs, method, rtol, atol, max_steps)``.
    The data in ``args`` are kernel inputs, so another EOS with tables of the
    same length reuses the kernel. Many states run in one kernel call.
    Switches JAX to 64-bit floats for the whole process, since the TOV
    equations are not accurate in 32 bits.

    Parameters
    ----------
    method : str
        A diffrax solver name, or the scipy name of the same order (default
        ``'Dopri5'``).
    max_steps : int
        Largest number of steps per integration (default ``100_000``). An
        integration that needs more fails.
    """

    def __init__(self, method="Dopri5", max_steps=100_000):
        try:
            import jax
            import diffrax  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The 'jax' backend needs jax and diffrax: "
                "pip install 'jax[cpu]' diffrax"
            ) from exc
        if method not in _JAX_METHODS:
            raise ValueError(
                f"Unknown method {method!r} for the jax backend. "
                f"Valid options are: {sorted(_JAX_METHODS)}"
            )
        jax.config.update("jax_enable_x64", True)
        self.method = method
        self.max_steps = max_steps

    def integrate(
        self, rhs, args, t0, t1, y0, first_step, rtol, atol, trajectory=False
    ):
        solve = _compiled_solve(
            rhs,
            _JAX_METHODS[self.method],
            rtol,
            atol,
            self.max_steps,
            trajectory,
            False,
        )
        dt0 = np.copysign(first_step, t1 - t0)
        ts, ys, success = solve(t0, t1, dt0, np.asarray(y0, dtype=float), args)
        if not trajectory:
            return float(ts), np.asarray(ys), bool(success), None
        # diffrax fills the entries after the last step with inf
        ts, ys = np.asarray(ts), np.asarray(ys)
        keep = np.isfinite(ts)
        ts, ys = ts[keep], ys[keep]
        return float(ts[-1]), ys[-1].copy(), bool(success), SolverResult(ts, ys.T)

    def integrate_many(self, rhs, args, t0, t1, y0, first_step, rtol, atol):
        solve = _compiled_solve(
            rhs, _JAX_METHODS[self.method], rtol, atol, self.max_steps, False, True
        )
        t0, t1 = np.asarray(t0, dtype=float), np.asarray(t1, dtype=float)
        dt0 = np.copysign(first_step, t1 - t0)
        ts, ys, success = solve(t0, t1, dt0, np.asarray(y0, dtype=float), args)
        return np.asarray(ts), np.asarray(ys), np.asarray(success)


@lru_cache(maxsize=32)
def _compiled_solve(rhs, method, rtol, atol, max_steps, trajectory, batched):
    """Compiled kernel ``solve(t0, t1, dt0, y0, args) -> (t, y, success)`` for
    one set of equations. With ``batched``, it takes arrays of ``t0, t1, dt0,
    y0`` and solves them all in one call."""
    import jax
    import jax.numpy as jnp
    import diffrax as dx

    term = dx.ODETerm(lambda t, y, args: rhs(t, y, args, jnp))
    solver = getattr(dx, method)()
    if isinstance(solver, dx.AbstractImplicitSolver):
        solver = getattr(dx, method)(root_finder=dx.VeryChord(rtol=rtol, atol=atol))
    saveat = dx.SaveAt(steps=True) if trajectory else dx.SaveAt(t1=True)

    def solve(t0, t1, dt0, y0, args):
        sol = dx.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            args=args,
            stepsize_controller=dx.PIDController(rtol=rtol, atol=atol),
            saveat=saveat,
            max_steps=max_steps,
            throw=False,
        )
        success = sol.result == dx.RESULTS.successful
        if trajectory:
            return sol.ts, sol.ys, success
        return sol.ts[-1], sol.ys[-1], success

    if batched:
        return jax.jit(jax.vmap(solve, in_axes=(0, 0, 0, 0, None)))
    return jax.jit(solve)


_SOLVER_REGISTRY = {
    "scipy": ScipySolver,
    "jax": JaxSolver,
}


def make_solver(backend, **kwargs):
    """Return an :class:`ODESolver` for *backend*.

    Parameters
    ----------
    backend : str or :class:`ODESolver`
        ``'scipy'``, ``'jax'``, or an already-built :class:`ODESolver`
        (returned as is).
    **kwargs
        Forwarded to the solver, e.g. ``method='RK45'``.

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
