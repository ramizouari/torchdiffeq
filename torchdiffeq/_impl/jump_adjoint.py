"""Adjoint sensitivities for jump ODEs.

The adjoint method integrates a backward ODE through the solution, which is
exactly what a jump breaks: the state is discontinuous at an event, so a single
backward sweep silently skips the discontinuity and returns gradients for the
jump-free problem.

The fix here is to split the solve at the event times and treat the two kinds of
transition separately. Between events the dynamics are smooth, so each segment
goes through the ordinary `odeint_adjoint` and keeps its O(1) memory profile.
The jump map itself is a cheap pointwise function, so it is differentiated by
plain autograd, which links consecutive segments together into one graph.

Event times are discovered by a `torch.no_grad()` forward pass that records which
event streams fired and when; the differentiable replay then applies exactly
those recorded events. Recording rather than re-deciding matters for two
reasons: a stochastic mechanism would otherwise draw fresh thresholds and
simulate a different trajectory, and re-locating a state-dependent event from a
slightly different state could place it on the other side of a threshold.

Not differentiated: the event *times* of a state-dependent mechanism. Gradients
flow through the state and through the jump network, but the sensitivity of the
event time itself to the parameters is dropped -- the same treatment
`odeint_event` gives event times. For a `FixedJumpMechanism` the times are data,
so there is nothing to drop.
"""

import torch

from .adjoint import odeint_adjoint
from .jump import JumpMechanism
from .odeint import odeint

__all__ = ["odeint_jump_adjoint"]

_JUMP_KEYS = ("jump", "jump_mechanism", "events")


class _RecordingJumpMechanism(JumpMechanism):
    """Wraps a mechanism and records every event it realises."""

    def __init__(self, inner: JumpMechanism):
        self.inner = inner
        self.records = []

    @property
    def event_times(self):
        return self.inner.event_times

    @property
    def kind(self):
        return self.inner.kind

    def has_event_at(self, t):
        return self.inner.has_event_at(t)

    def next_event_time(self, z_interp, t0, t1, tol=1e-6):
        return self.inner.next_event_time(z_interp, t0, t1, tol)

    def latent_proj(self, z, enforce=False):
        return self.inner.latent_proj(z, enforce=enforce)

    def realise_event(self, t, z):
        mask = self.inner.realise_event(t, z)
        self.records.append(
            (
                torch.as_tensor(t).detach().clone(),
                mask.detach().clone(),
            )
        )
        return mask


def _strip_jump_options(options):
    """A copy of `options` with the jump arguments removed."""
    if options is None:
        return {}
    return {k: v for k, v in options.items() if k not in _JUMP_KEYS}


def odeint_jump_adjoint(
    func,
    y0,
    t,
    *,
    rtol=1e-7,
    atol=1e-9,
    method=None,
    options=None,
    adjoint_rtol=None,
    adjoint_atol=None,
    adjoint_method=None,
    adjoint_options=None,
    adjoint_params=None,
):
    """Solve a jump ODE with adjoint-mode gradients.

    Takes the same arguments as :func:`odeint_adjoint`, with the jump network
    and the jump mechanism supplied through ``options`` exactly as
    :func:`odeint` takes them:

        odeint_jump_adjoint(func, y0, t, method="dopri5",
                            options={"jump": h, "jump_mechanism": mechanism})

    With no jump arguments this is :func:`odeint_adjoint`.

    Returns:
        The solution evaluated at ``t``, differentiable with respect to ``y0``,
        the parameters of ``func`` and the parameters of the jump network.
    """
    options = {} if options is None else dict(options)
    jump = options.get("jump")
    mechanism = options.get("jump_mechanism")

    segment_options = _strip_jump_options(options)
    segment_kwargs = dict(
        rtol=rtol,
        atol=atol,
        method=method,
        adjoint_rtol=adjoint_rtol,
        adjoint_atol=adjoint_atol,
        adjoint_method=adjoint_method,
        adjoint_options=_strip_jump_options(adjoint_options)
        if adjoint_options is not None
        else None,
        adjoint_params=adjoint_params,
    )

    if jump is None and mechanism is None:
        return odeint_adjoint(func, y0, t, options=segment_options, **segment_kwargs)
    if jump is None or mechanism is None:
        raise ValueError(
            "`jump` and `jump_mechanism` must be supplied together in `options`."
        )
    if len(t) > 1 and t[0] > t[1]:
        raise NotImplementedError(
            "odeint_jump_adjoint requires increasing `t`; the jump map is not "
            "invertible in general."
        )

    # ---------------------------------------------------------------- #
    # Pass 1: discover the events, without building a graph.
    # ---------------------------------------------------------------- #
    recorder = _RecordingJumpMechanism(mechanism)
    with torch.no_grad():
        odeint(
            func,
            y0,
            t,
            rtol=rtol,
            atol=atol,
            method=method,
            options={**options, "jump_mechanism": recorder},
        )
    # The solver works in its own time dtype (float64 by default), which need
    # not be the dtype of `t`; segment boundaries have to be built in `t`'s.
    events = [
        (torch.as_tensor(t_e).to(dtype=t.dtype, device=t.device), mask)
        for t_e, mask in recorder.records
        if mask.any()
    ]

    # ---------------------------------------------------------------- #
    # Pass 2: replay differentiably, one adjoint solve per smooth segment.
    # ---------------------------------------------------------------- #
    solution = [None] * len(t)
    y = y0
    j = 0

    # Cadlag: an event on the first time point is already reflected there.
    while events and events[0][0] <= t[0]:
        t_e, mask = events.pop(0)
        y = y + mechanism.increment_from_mask(t[0], y, jump, mask)
    solution[0] = y
    j = 1

    t_segment_start = t[0]
    for t_e, mask in events:
        if t_e <= t_segment_start:
            # Two events at the same instant, or one rounded onto the segment
            # start by the cast above: no interval to integrate, just jump.
            y = y + mechanism.increment_from_mask(t_e, y, jump, mask)
            if j < len(t) and t[j] == t_e:
                solution[j] = y
                j += 1
            continue

        # Integrate up to the event, reporting any output times on the way.
        inner = [t[i] for i in range(j, len(t)) if t[i] < t_e]
        seg_t = torch.stack([t_segment_start, *inner, t_e])
        seg_sol = odeint_adjoint(
            func, y, seg_t, options=dict(segment_options), **segment_kwargs
        )
        for offset in range(len(inner)):
            solution[j + offset] = seg_sol[1 + offset]
        j += len(inner)
        y = seg_sol[-1]

        # The jump: ordinary autograd, linking this segment to the next.
        y = y + mechanism.increment_from_mask(t_e, y, jump, mask)
        if j < len(t) and t[j] == t_e:
            solution[j] = y
            j += 1
        t_segment_start = t_e

    if j < len(t):
        seg_t = torch.stack([t_segment_start, *[t[i] for i in range(j, len(t))]])
        seg_sol = odeint_adjoint(
            func, y, seg_t, options=dict(segment_options), **segment_kwargs
        )
        for offset in range(len(t) - j):
            solution[j + offset] = seg_sol[1 + offset]

    return torch.stack(solution, dim=0)
