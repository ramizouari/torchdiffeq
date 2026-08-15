"""Tests for jump ODE support.

The reference problem throughout is a linear ODE with multiplicative jumps,

    dz/dt = a z ,    z(t_k+) = z(t_k-) (1 + b)    for each event time t_k

whose solution is  z(T) = z0 exp(a T) (1 + b)^n  for n events in [0, T]. Being
analytic, it pins down not just "a jump happened" but its size, its timing and
the order of convergence of the solver that produced it.

Several tests below carry a `# regression:` note naming the defect they cover;
each of those fails on the implementation that preceded this suite.
"""

import math

import pytest
import torch

import torchdiffeq
from torchdiffeq._impl.fixed_grid import Euler
from torchdiffeq._impl.odeint import SOLVERS
from torchdiffeq.jump import (
    CoupledStochasticJumpMechanism,
    FixedJumpMechanism,
    odeint_jump_adjoint,
)

torch.manual_seed(0)

DTYPE = torch.float64
BATCH, LATENT, N_TYPES = 2, 3, 1

FIXED_GRID_METHODS = ["euler", "midpoint", "heun2", "heun3", "rk4"]
ADAPTIVE_METHODS = ["dopri5", "tsit5", "bosh3", "adaptive_heun", "fehlberg2", "dopri8"]
IMPLICIT_METHODS = [
    "implicit_euler",
    "implicit_midpoint",
    "trapezoid",
    "gl4",
    "gl6",
    "radauIIA3",
    "radauIIA5",
    "sdirk2",
    "trbdf2",
]
STEPPED_METHODS = FIXED_GRID_METHODS + IMPLICIT_METHODS + [
    "explicit_adams",
    "implicit_adams",
    "fixed_adams",
]


# --------------------------------------------------------------------------- #
#                             Reference problem                               #
# --------------------------------------------------------------------------- #


class LinearDynamics(torch.nn.Module):
    """dz/dt = a z on the latent block; augmented coordinates stay frozen."""

    def __init__(self, a=0.5, n_augmented=0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a, dtype=DTYPE))
        self.n_augmented = n_augmented

    def forward(self, t, z):
        if self.n_augmented == 0:
            return self.a * z
        dz = torch.zeros_like(z)
        dz[..., : -self.n_augmented] = self.a * z[..., : -self.n_augmented]
        return dz


class ScaleJump(torch.nn.Module):
    """h(t, z) = b z, so an event multiplies the state by (1 + b)."""

    def __init__(self, b=0.3, n_types=N_TYPES):
        super().__init__()
        self.b = torch.nn.Parameter(
            torch.tensor([b * (k + 1) for k in range(n_types)], dtype=DTYPE)
        )
        self.n_types = n_types

    def forward(self, t, z):
        if self.n_types == 0:  # simple mode: h has the shape of z
            return self.b[0] * z
        return z.unsqueeze(-1) * self.b


class SimpleScaleJump(torch.nn.Module):
    """h(t, z) = b z for an uncoupled mechanism."""

    def __init__(self, b=0.3):
        super().__init__()
        self.b = torch.nn.Parameter(torch.tensor(b, dtype=DTYPE))

    def forward(self, t, z):
        return self.b * z


class LatentProj(torch.nn.Module):
    """Keep everything but the trailing `n_augmented` coordinates."""

    def __init__(self, n_augmented):
        super().__init__()
        self.n_augmented = n_augmented

    def forward(self, x):
        if self.n_augmented == 0:
            return x[...]
        return x[..., : -self.n_augmented]


def fixed_mechanism(event_times, n_types=N_TYPES, batch=BATCH, n_augmented=0,
                    mask=None, coupled=True):
    event_times = torch.as_tensor(event_times, dtype=DTYPE)
    if mask is None:
        mask = (
            torch.ones((batch, len(event_times), n_types), dtype=DTYPE)
            if coupled
            else torch.ones((batch, len(event_times)), dtype=DTYPE)
        )
    return FixedJumpMechanism(
        event_times,
        mask,
        coupled=coupled,
        latent_proj=LatentProj(n_augmented) if n_augmented else None,
    )


def exact(a=0.5, b=0.3, n_events=1, T=1.0, z0=1.0):
    return z0 * math.exp(a * T) * (1 + b) ** n_events


def solve(event_times, method="euler", t=None, a=0.5, b=0.3, z0=None,
          n_augmented=0, mechanism=None, jump=None, **options):
    dyn = LinearDynamics(a, n_augmented)
    jump = ScaleJump(b) if jump is None else jump
    if z0 is None:
        z0 = torch.ones(BATCH, LATENT + n_augmented, dtype=DTYPE)
    if t is None:
        t = torch.tensor([0.0, 1.0], dtype=DTYPE)
    if mechanism is None:
        mechanism = fixed_mechanism(event_times, n_augmented=n_augmented)
    options = {"jump": jump, "jump_mechanism": mechanism, **options}
    return torchdiffeq.odeint(
        dyn, z0, t, method=method, rtol=1e-10, atol=1e-12, options=options
    )


# --------------------------------------------------------------------------- #
#                          Backward compatibility                             #
# --------------------------------------------------------------------------- #

_A = torch.tensor([[-0.5, 1.0], [-1.0, -0.5]], dtype=DTYPE)


def _linear_system(t, y):
    return y @ _A.T


def _linear_system_exact(t):
    return torch.matrix_exp(_A * t) @ torch.tensor([1.0, 0.0], dtype=DTYPE)


@pytest.mark.parametrize("method", sorted(set(SOLVERS) - {"scipy_solver"}))
def test_solver_still_works_without_jump_arguments(method):
    """Every registered solver integrates a plain ODE.

    # regression: implicit fixed-grid solvers raised AttributeError on
    # `jump_mechanism` even when no jump argument was passed.
    """
    options = {"step_size": 0.05} if method in STEPPED_METHODS else {}
    t = torch.linspace(0, 1, 5, dtype=DTYPE)
    out = torchdiffeq.odeint(_linear_system, torch.tensor([1.0, 0.0], dtype=DTYPE),
                             t, method=method, options=options)
    assert out.shape == (5, 2)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[-1], _linear_system_exact(t[-1]), rtol=5e-2,
                               atol=5e-2)


@pytest.mark.parametrize("method", ["euler", "midpoint", "rk4", "dopri5"])
def test_complex_valued_state(method):
    """Complex states integrate on fixed-grid solvers.

    # regression: an empty jump_t typed from y0.dtype promoted the time grid to
    # complex, and sorting it raised.
    """
    out = torchdiffeq.odeint(
        lambda t, y: 1j * y,
        torch.tensor([1.0 + 0j], dtype=torch.complex128),
        torch.linspace(0, 1, 3, dtype=DTYPE),
        method=method,
        options={"step_size": 0.01},
    )
    torch.testing.assert_close(
        out[-1],
        torch.tensor([complex(math.cos(1.0), math.sin(1.0))], dtype=torch.complex128),
        rtol=1e-2, atol=1e-2,
    )


@pytest.mark.parametrize("method", ["euler", "rk4", "dopri5"])
def test_time_stays_differentiable_without_jumps(method):
    """`t` stays differentiable when no jump times are supplied.

    # regression: the grid was de-duplicated with `unique`, which has no
    # derivative, so gradients w.r.t. `t` raised NotImplementedError.
    """
    y0 = torch.tensor([1.0, 0.5], dtype=DTYPE, requires_grad=True)
    t = torch.tensor([0.0, 0.4, 1.0], dtype=DTYPE, requires_grad=True)
    options = {"step_size": 0.1} if method != "dopri5" else {}
    out = torchdiffeq.odeint(_linear_system, y0, t, method=method, options=options)
    grad_t, grad_y0 = torch.autograd.grad(out.sum(), (t, y0))
    assert torch.isfinite(grad_t).all()
    assert torch.isfinite(grad_y0).all()
    assert (grad_t != 0).any()
    # The value of d/dt is checked separately by gradcheck on an adaptive
    # solver; for a fixed grid it also picks up the grid's dependence on t[-1].


def test_gradcheck_with_respect_to_time_without_jumps():
    y0 = torch.tensor([1.0, 0.5], dtype=DTYPE, requires_grad=True)
    t = torch.tensor([0.0, 0.4, 1.0], dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda y, tt: torchdiffeq.odeint(_linear_system, y, tt, method="dopri5"),
        (y0, t),
    )


def test_method_may_be_a_solver_class():
    """`method` accepts a solver class, not just a registered name.

    # regression: the callback check did SOLVERS[method] unconditionally.
    """
    out = torchdiffeq.odeint(
        _linear_system, torch.tensor([1.0, 0.0], dtype=DTYPE),
        torch.tensor([0.0, 1.0], dtype=DTYPE), method=Euler,
        options={"step_size": 0.01},
    )
    torch.testing.assert_close(out[-1], _linear_system_exact(torch.tensor(1.0)),
                               rtol=5e-2, atol=5e-2)


# --------------------------------------------------------------------------- #
#                        Accuracy and convergence                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("method", "order"), [("euler", 1), ("midpoint", 2),
                                               ("rk4", 4)])
def test_convergence_order_is_preserved_across_a_jump(method, order):
    """A jump must not cost the solver its convergence order.

    # regression: the jump was added after the step rather than splitting it, so
    # every fixed-grid solver collapsed to first order.
    """
    errors = []
    for step in (0.02, 0.01, 0.005):
        out = solve([0.31337], method=method, step_size=step)
        errors.append(abs(out[-1, 0, 0].item() - exact()))
    observed = [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert min(observed) > order - 0.4, (
        f"{method}: observed orders {observed}, expected ~{order}"
    )


@pytest.mark.parametrize("method", FIXED_GRID_METHODS)
def test_fixed_grid_hits_the_analytic_solution(method):
    out = solve([0.31337], method=method, step_size=1e-4)
    assert out[-1, 0, 0].item() == pytest.approx(exact(), abs=1e-3)


@pytest.mark.parametrize("method", ADAPTIVE_METHODS)
def test_adaptive_hits_the_analytic_solution(method):
    """Adaptive solvers honour a jump mechanism.

    # regression: `jump_mechanism` was not a recognised argument for adaptive
    # solvers; it was dropped with a warning.
    """
    out = solve([0.31337], method=method)
    assert out[-1, 0, 0].item() == pytest.approx(exact(), abs=1e-5)


@pytest.mark.parametrize("method", IMPLICIT_METHODS)
def test_implicit_fixed_grid_honours_a_jump(method):
    out = solve([0.31337], method=method, step_size=1e-3)
    assert out[-1, 0, 0].item() == pytest.approx(exact(), abs=1e-2)


def test_fixed_grid_and_adaptive_agree():
    fixed = solve([0.2, 0.55, 0.9], method="rk4", step_size=1e-4)
    adaptive = solve([0.2, 0.55, 0.9], method="dopri5")
    torch.testing.assert_close(fixed, adaptive, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
#                            Event bookkeeping                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["euler", "rk4", "dopri5"])
def test_event_exactly_on_a_grid_point_fires(method):
    """# regression: an event landing on a step boundary was silently dropped.

    Frozen dynamics (a = 0) isolate the jump algebra: the answer is exactly
    (1 + b) whatever the solver's own accuracy.
    """
    out = solve([0.5], method=method, a=0.0, step_size=0.1)
    assert out[-1, 0, 0].item() == pytest.approx(1.3, abs=1e-12)


@pytest.mark.parametrize("method", ["euler", "rk4", "dopri5"])
def test_event_fires_without_an_output_time_in_its_step(method):
    """# regression: events were only discovered in steps holding an output time.

    With `t = [0, 1]` and a small step size, every interior step is free of
    output times, so this used to return the jump-free trajectory.
    """
    out = solve([0.31337], method=method, t=torch.tensor([0.0, 1.0], dtype=DTYPE),
                step_size=1e-4)
    assert out[-1, 0, 0].item() == pytest.approx(exact(), abs=1e-3)
    assert out[-1, 0, 0].item() != pytest.approx(math.exp(0.5), abs=1e-3)


def test_event_is_not_applied_twice_with_several_outputs_per_step():
    """# regression: a later event leaked into earlier outputs and was applied
    twice when a step contained more than one output time."""
    t = torch.tensor([0.0, 0.25, 0.75, 1.0], dtype=DTYPE)
    out = solve([0.9], method="euler", a=0.0, t=t, step_size=1.0)
    # Before the event the state is untouched; after it, scaled exactly once.
    assert out[1, 0, 0].item() == pytest.approx(1.0, abs=1e-12)
    assert out[2, 0, 0].item() == pytest.approx(1.0, abs=1e-12)
    assert out[3, 0, 0].item() == pytest.approx(1.3, abs=1e-12)


@pytest.mark.parametrize("method", ["euler", "dopri5"])
def test_event_at_the_initial_time_is_cadlag(method):
    out = solve([0.0], method=method, a=0.0, step_size=0.01)
    assert out[0, 0, 0].item() == pytest.approx(1.3, abs=1e-12)
    assert out[-1, 0, 0].item() == pytest.approx(1.3, abs=1e-12)


@pytest.mark.parametrize("method", ["euler", "rk4", "dopri5"])
def test_output_at_an_event_time_is_the_post_jump_value(method):
    t = torch.tensor([0.0, 0.4, 1.0], dtype=DTYPE)
    out = solve([0.4], method=method, t=t, step_size=1e-4)
    assert out[1, 0, 0].item() == pytest.approx(math.exp(0.5 * 0.4) * 1.3, abs=1e-4)


@pytest.mark.parametrize("method", ["euler", "rk4", "dopri5"])
def test_several_events_inside_one_step(method):
    out = solve([0.21, 0.24, 0.27], method=method, a=0.0, step_size=0.5)
    assert out[-1, 0, 0].item() == pytest.approx(1.3 ** 3, abs=1e-12)


def test_cubic_interpolation_applies_jumps():
    """# regression: the cubic branch built its own increment and threw it away."""
    t = torch.tensor([0.0, 0.5, 1.0], dtype=DTYPE)
    linear = solve([0.31337], method="rk4", t=t, step_size=1e-3, interp="linear")
    cubic = solve([0.31337], method="rk4", t=t, step_size=1e-3, interp="cubic")
    assert cubic[-1, 0, 0].item() == pytest.approx(exact(), abs=1e-4)
    torch.testing.assert_close(cubic, linear, rtol=1e-5, atol=1e-5)


def test_events_past_the_final_time_are_not_realised():
    """An adaptive step may overshoot `t[-1]`; events beyond it stay unconsumed."""
    mechanism = fixed_mechanism([0.5, 1.5])
    out = solve(None, method="dopri5", mechanism=mechanism,
                t=torch.tensor([0.0, 1.0], dtype=DTYPE))
    assert out[-1, 0, 0].item() == pytest.approx(exact(n_events=1), abs=1e-6)
    assert mechanism.idx == 1


# --------------------------------------------------------------------------- #
#                        Projections and shapes                               #
# --------------------------------------------------------------------------- #


def test_identity_latent_projection_is_allowed():
    """# regression: the documented default projection always raised."""
    mechanism = FixedJumpMechanism(
        torch.tensor([0.31337], dtype=DTYPE),
        torch.ones((BATCH, 1, N_TYPES), dtype=DTYPE),
        coupled=True,
    )
    out = solve(None, method="rk4", mechanism=mechanism, step_size=1e-3)
    assert out[-1, 0, 0].item() == pytest.approx(exact(), abs=1e-4)


def test_augmented_coordinates_are_never_jumped():
    """A cumulative hazard tracked alongside the latent state must not jump."""
    n_aug = 2
    out = solve([0.31337], method="rk4", step_size=1e-3, n_augmented=n_aug)
    # The augmented block has zero dynamics and must not receive the jump.
    torch.testing.assert_close(
        out[-1, :, -n_aug:], torch.ones(BATCH, n_aug, dtype=DTYPE)
    )
    assert out[-1, 0, 0].item() == pytest.approx(exact(), abs=1e-4)


def test_non_view_latent_projection_is_rejected():
    class Copying(torch.nn.Module):
        def forward(self, x):
            return x.clone()[..., :LATENT]

    mechanism = FixedJumpMechanism(
        torch.tensor([0.3], dtype=DTYPE),
        torch.ones((BATCH, 1, N_TYPES), dtype=DTYPE),
        coupled=True,
        latent_proj=Copying(),
    )
    with pytest.raises(RuntimeError, match="view"):
        solve(None, method="euler", mechanism=mechanism, step_size=0.1)


def test_coupled_event_types_use_their_own_jump():
    """Each event stream contributes its own column of h."""
    n_types = 3
    mask = torch.zeros((BATCH, 1, n_types), dtype=DTYPE)
    mask[:, 0, 1] = 1.0  # only the middle stream fires
    mechanism = FixedJumpMechanism(torch.tensor([0.31337], dtype=DTYPE), mask,
                                   coupled=True)
    out = solve(None, method="rk4", mechanism=mechanism,
                jump=ScaleJump(0.3, n_types=n_types), step_size=1e-3)
    # ScaleJump gives stream k the coefficient 0.3*(k+1); stream 1 -> 0.6.
    assert out[-1, 0, 0].item() == pytest.approx(exact(b=0.6), abs=1e-4)


def test_simple_uncoupled_mechanism():
    mechanism = fixed_mechanism([0.31337], coupled=False)
    out = solve(None, method="rk4", mechanism=mechanism, jump=SimpleScaleJump(0.3),
                step_size=1e-3)
    assert out[-1, 0, 0].item() == pytest.approx(exact(), abs=1e-4)


def test_fixed_mechanism_validates_its_inputs():
    with pytest.raises(ValueError, match="one entry per event time"):
        FixedJumpMechanism(torch.tensor([0.1, 0.2], dtype=DTYPE),
                           torch.ones((BATCH, 1, N_TYPES), dtype=DTYPE), coupled=True)
    with pytest.raises(ValueError, match="non-decreasing"):
        FixedJumpMechanism(torch.tensor([0.2, 0.1], dtype=DTYPE),
                           torch.ones((BATCH, 2, N_TYPES), dtype=DTYPE), coupled=True)


# --------------------------------------------------------------------------- #
#                              Argument handling                              #
# --------------------------------------------------------------------------- #


def test_legacy_jump_t_events_api_matches_the_mechanism_api():
    """The original jump_t/events API still works and agrees with a mechanism.

    # regression: with the default linear interpolation it raised on
    # `next_after(None)`; with cubic it silently applied nothing.
    """
    dyn = LinearDynamics(0.5)
    z0 = torch.ones(BATCH, LATENT, dtype=DTYPE)
    t = torch.linspace(0, 1, 11, dtype=DTYPE)
    jump_t = torch.tensor([0.25, 0.65], dtype=DTYPE)
    events = torch.ones((BATCH, 2), dtype=DTYPE)

    legacy = torchdiffeq.odeint(
        dyn, z0, t, method="rk4", rtol=1e-10, atol=1e-12,
        options={"step_size": 1e-3, "jump": SimpleScaleJump(0.3),
                 "jump_t": jump_t, "events": events},
    )
    mechanism = torchdiffeq.odeint(
        dyn, z0, t, method="rk4", rtol=1e-10, atol=1e-12,
        options={"step_size": 1e-3, "jump": SimpleScaleJump(0.3),
                 "jump_mechanism": FixedJumpMechanism(jump_t, events, coupled=False)},
    )
    torch.testing.assert_close(legacy, mechanism)
    assert legacy[-1, 0, 0].item() == pytest.approx(exact(n_events=2), abs=1e-4)


def test_jump_t_alone_keeps_its_upstream_meaning():
    """`jump_t` without `jump`/`events` is a discontinuity in f, not a state jump."""
    t = torch.tensor([0.0, 1.0], dtype=DTYPE)
    plain = torchdiffeq.odeint(_linear_system, torch.tensor([1.0, 0.0], dtype=DTYPE),
                               t, method="rk4", options={"step_size": 0.01})
    with_jump_t = torchdiffeq.odeint(
        _linear_system, torch.tensor([1.0, 0.0], dtype=DTYPE), t, method="rk4",
        options={"step_size": 0.01, "jump_t": torch.tensor([0.5], dtype=DTYPE)},
    )
    torch.testing.assert_close(plain, with_jump_t, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"jump": SimpleScaleJump()}, "nothing to tell the solver"),
        ({"events": torch.ones((BATCH, 1), dtype=DTYPE)}, "requires both"),
        (
            {"jump": SimpleScaleJump(), "events": torch.ones((BATCH, 1), dtype=DTYPE)},
            "requires both",
        ),
    ],
)
def test_incomplete_jump_arguments_raise(options, match):
    """Partial jump specifications fail immediately, with an explanation."""
    with pytest.raises(ValueError, match=match):
        torchdiffeq.odeint(
            _linear_system, torch.tensor([1.0, 0.0], dtype=DTYPE),
            torch.tensor([0.0, 1.0], dtype=DTYPE), method="euler",
            options={"step_size": 0.1, **options},
        )


def test_reverse_time_with_jumps_raises():
    with pytest.raises(NotImplementedError, match="decreasing"):
        solve([0.3], method="euler", t=torch.tensor([1.0, 0.0], dtype=DTYPE),
              step_size=0.1)


# --------------------------------------------------------------------------- #
#                                 Gradients                                   #
# --------------------------------------------------------------------------- #


def _grads(fn, method, step=None, event_times=(0.31337, 0.62)):
    dyn, jump = LinearDynamics(0.5), ScaleJump(0.3)
    z0 = torch.ones(1, 1, dtype=DTYPE, requires_grad=True)
    options = {"jump": jump,
               "jump_mechanism": fixed_mechanism(list(event_times), batch=1)}
    if step is not None:
        options["step_size"] = step
    kwargs = {}
    if fn is odeint_jump_adjoint:
        kwargs["adjoint_params"] = tuple(dyn.parameters()) + tuple(jump.parameters())
    out = fn(dyn, z0, torch.tensor([0.0, 1.0], dtype=DTYPE), method=method,
             rtol=1e-11, atol=1e-13, options=options, **kwargs)
    out[-1].sum().backward()
    return out[-1].item(), z0.grad.item(), dyn.a.grad.item(), jump.b.grad.item()


@pytest.mark.parametrize(("method", "step"), [("rk4", 1e-3), ("dopri5", None)])
def test_adjoint_gradients_match_direct_mode(method, step):
    """# regression: adjoint mode integrated straight through the discontinuity
    and returned gradients for the jump-free problem, with exactly zero
    gradient reaching the jump network."""
    direct = _grads(torchdiffeq.odeint, method, step)
    adjoint = _grads(odeint_jump_adjoint, method, step)
    for got, want, name in zip(adjoint, direct, ("z", "dz0", "da", "db")):
        assert got == pytest.approx(want, rel=1e-5), name
    assert adjoint[3] != 0.0


def test_adjoint_gradients_match_the_analytic_solution():
    z, dz0, da, db = _grads(odeint_jump_adjoint, "dopri5")
    eps = 1e-6
    assert z == pytest.approx(exact(n_events=2), rel=1e-7)
    assert dz0 == pytest.approx(exact(n_events=2), rel=1e-6)
    assert da == pytest.approx(
        (exact(a=0.5 + eps, n_events=2) - exact(a=0.5 - eps, n_events=2)) / (2 * eps),
        rel=1e-5,
    )
    assert db == pytest.approx(
        (exact(b=0.3 + eps, n_events=2) - exact(b=0.3 - eps, n_events=2)) / (2 * eps),
        rel=1e-5,
    )


def test_jump_adjoint_without_events_matches_odeint_adjoint():
    def run(fn):
        dyn = LinearDynamics(0.5)
        z0 = torch.ones(1, 1, dtype=DTYPE, requires_grad=True)
        out = fn(dyn, z0, torch.tensor([0.0, 0.3, 1.0], dtype=DTYPE), method="dopri5",
                 rtol=1e-10, atol=1e-12, adjoint_params=tuple(dyn.parameters()))
        out[-1].sum().backward()
        return out, z0.grad.clone(), dyn.a.grad.clone()

    a_out, a_z0, a_a = run(torchdiffeq.odeint_adjoint)
    b_out, b_z0, b_a = run(odeint_jump_adjoint)
    torch.testing.assert_close(a_out, b_out)
    torch.testing.assert_close(a_z0, b_z0)
    torch.testing.assert_close(a_a, b_a)


def test_gradients_reach_the_jump_network_in_direct_mode():
    _, _, _, db = _grads(torchdiffeq.odeint, "rk4", 1e-3)
    assert db != 0.0


def test_adjoint_reports_intermediate_times_like_direct_mode():
    t = torch.tensor([0.0, 0.2, 0.31337, 0.5, 0.62, 0.8, 1.0], dtype=DTYPE)
    dyn, jump = LinearDynamics(0.5), ScaleJump(0.3)
    common = dict(method="dopri5", rtol=1e-11, atol=1e-13)
    z0 = torch.ones(1, 1, dtype=DTYPE)
    direct = torchdiffeq.odeint(
        dyn, z0, t,
        options={"jump": jump, "jump_mechanism": fixed_mechanism([0.31337, 0.62],
                                                                 batch=1)},
        **common,
    )
    adjoint = odeint_jump_adjoint(
        dyn, z0, t,
        options={"jump": jump, "jump_mechanism": fixed_mechanism([0.31337, 0.62],
                                                                 batch=1)},
        adjoint_params=tuple(dyn.parameters()) + tuple(jump.parameters()),
        **common,
    )
    torch.testing.assert_close(direct, adjoint, rtol=1e-7, atol=1e-9)


# --------------------------------------------------------------------------- #
#                          Stochastic mechanism                               #
# --------------------------------------------------------------------------- #


class ConstantHazard(torch.nn.Module):
    """dz/dt = 0 on the latent block, dLambda/dt = rate."""

    def __init__(self, rate, latent):
        super().__init__()
        self.rate = rate
        self.latent = latent

    def forward(self, t, z):
        d = torch.zeros_like(z)
        d[..., self.latent:] = self.rate
        return d


class UnitJump(torch.nn.Module):
    def __init__(self, n_types=1):
        super().__init__()
        self.n_types = n_types

    def forward(self, t, z):
        return torch.ones(*z.shape, self.n_types, dtype=z.dtype, device=z.device)


def stochastic_run(method, batch=64, rate=1.5, horizon=6.0, seed=0, max_events=torch.inf,
                   **options):
    latent, n_types = 1, 1
    generator = torch.Generator().manual_seed(seed)
    mechanism = CoupledStochasticJumpMechanism(
        batch_shape=(batch,),
        coupling_dim=n_types,
        cumulative_hazards_proj=lambda t, z: z[..., latent:],
        rng=generator,
        latent_proj=LatentProj(n_types),
        max_events=max_events,
    )
    out = torchdiffeq.odeint(
        ConstantHazard(rate, latent),
        torch.zeros(batch, latent + n_types, dtype=DTYPE),
        torch.tensor([0.0, horizon], dtype=DTYPE),
        method=method, rtol=1e-9, atol=1e-11,
        options={"jump": UnitJump(n_types), "jump_mechanism": mechanism, **options},
    )
    return mechanism, out


def test_constant_hazard_gives_a_poisson_number_of_events():
    """N(T) ~ Poisson(rate*T): mean and variance both equal rate*T."""
    rate, horizon, batch = 1.5, 6.0, 400
    mechanism, out = stochastic_run("dopri5", batch=batch, rate=rate, horizon=horizon)
    counts = torch.tensor([float(len(e)) for e in mechanism._realised_events])
    expected = rate * horizon
    # Standard error of the mean of a Poisson sample.
    tol = 4 * math.sqrt(expected / batch)
    assert counts.mean().item() == pytest.approx(expected, abs=tol)
    assert counts.var().item() == pytest.approx(expected, rel=0.3)
    # Every event adds exactly one unit to the latent coordinate.
    torch.testing.assert_close(out[-1, :, 0], counts.to(DTYPE))


def test_event_times_are_conditionally_uniform():
    """Event times are uniform on [0, T] given how many there are.

    Conditional on N(T) = n, the arrival times of a homogeneous Poisson process
    are distributed as n iid Uniform(0, T) order statistics. Testing that -
    rather than the inter-arrival gaps - avoids the right-censoring bias that
    truncating the last, incomplete gap would introduce.
    """
    rate, horizon = 1.5, 8.0
    mechanism, _ = stochastic_run("dopri5", batch=200, rate=rate, horizon=horizon,
                                  seed=3)
    times = torch.tensor(
        sorted(float(e[0]) for events in mechanism._realised_events for e in events),
        dtype=DTYPE,
    )
    n = times.numel()
    assert n > 1000, "not enough events for a meaningful test"
    cdf = times / horizon
    empirical = torch.arange(1, n + 1, dtype=DTYPE) / n
    ks = torch.maximum((empirical - cdf).abs().max(),
                       (cdf - (empirical - 1 / n)).abs().max()).item()
    assert ks < 1.63 / math.sqrt(n), f"KS={ks:.4f}, n={n}"  # 1% critical value


def test_solvers_agree_on_stochastic_event_times():
    """Event location is a property of the problem, not of the solver."""
    fine, _ = stochastic_run("euler", batch=8, seed=11, step_size=1e-4)
    adaptive, _ = stochastic_run("dopri5", batch=8, seed=11)
    for a, b in zip(fine._realised_events, adaptive._realised_events):
        assert len(a) == len(b)
        for (ta, ka), (tb, kb) in zip(a, b):
            assert ka == kb
            assert float(ta) == pytest.approx(float(tb), abs=1e-3)


def test_max_events_caps_the_number_of_events():
    mechanism, out = stochastic_run("dopri5", batch=32, rate=3.0, horizon=6.0,
                                    seed=5, max_events=2)
    counts = [len(e) for e in mechanism._realised_events]
    assert max(counts) <= 2
    assert max(counts) == 2  # the cap is reachable, so the test is meaningful


def test_stochastic_jump_leaves_the_cumulative_hazard_untouched():
    """The hazard coordinate integrates rate*T regardless of how many jumps fire."""
    rate, horizon = 1.5, 4.0
    _, out = stochastic_run("dopri5", batch=16, rate=rate, horizon=horizon, seed=2)
    torch.testing.assert_close(
        out[-1, :, -1], torch.full((16,), rate * horizon, dtype=DTYPE),
        rtol=1e-6, atol=1e-6,
    )
