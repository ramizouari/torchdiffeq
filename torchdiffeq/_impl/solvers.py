import abc

import torch
from .event_handling import find_event
from .misc import _handle_unused_kwargs
from .jump import MAX_EVENTS_PER_STEP, FixedJumpMechanism, JumpMechanism


def next_after(x: torch.Tensor) -> torch.Tensor:
    """
    Returns the next representable floating-point value after x in the direction of infinity.
    """
    return torch.nextafter(x, torch.tensor(torch.inf, device=x.device, dtype=x.dtype))


def _resolve_jump_mechanism(jump, jump_t, events, jump_mechanism):
    """Normalise the two jump APIs onto a single :class:`JumpMechanism`.

    The original API takes a `jump` network plus prescribed `jump_t` times and a
    per-time `events` indicator; that is exactly a `FixedJumpMechanism` in
    uncoupled mode whose jump lands on the whole state, so it is expressed as
    one rather than kept as a second code path.

    Passing `jump_t` on its own keeps its upstream meaning - a discontinuity in
    `f` that the grid should land on - and produces no state jump.
    """
    if jump_mechanism is not None:
        if not isinstance(jump_mechanism, JumpMechanism):
            raise TypeError(
                "`jump_mechanism` must be a JumpMechanism, got "
                f"{type(jump_mechanism).__name__}."
            )
        if jump is None:
            raise ValueError(
                "`jump_mechanism` was given without `jump`: there is no jump "
                "network to evaluate when an event fires."
            )
        if events is not None:
            raise ValueError(
                "`events` and `jump_mechanism` are mutually exclusive; the "
                "mechanism already decides which events fire."
            )
        return jump_mechanism

    if events is not None:
        if jump is None or jump_t is None:
            raise ValueError(
                "`events` requires both `jump` and `jump_t`: they specify the "
                "jump network and the times its events occur at."
            )
        return FixedJumpMechanism(jump_t, events, coupled=False)

    if jump is not None:
        raise ValueError(
            "`jump` was given without `jump_mechanism` or `jump_t`+`events`: "
            "there is nothing to tell the solver when a jump occurs."
        )
    return None


class AdaptiveStepsizeODESolver(metaclass=abc.ABCMeta):
    def __init__(self, dtype, y0, norm, **unused_kwargs):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.y0 = y0
        self.dtype = dtype

        self.norm = norm

    def _before_integrate(self, t):
        pass

    @abc.abstractmethod
    def _advance(self, next_t):
        raise NotImplementedError

    @classmethod
    def valid_callbacks(cls):
        return set()

    def integrate(self, t):
        solution = torch.empty(
            len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device
        )
        t = t.to(self.dtype)
        # Before, not after, seeding solution[0]: `_before_integrate` is where a
        # jump landing on the very first time point is folded into the state.
        self._before_integrate(t)
        solution[0] = self.y0
        for i in range(1, len(t)):
            solution[i] = self._advance(t[i])
        return solution


class AdaptiveStepsizeEventODESolver(AdaptiveStepsizeODESolver, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _advance_until_event(self, event_fn):
        raise NotImplementedError

    def integrate_until_event(self, t0, event_fn):
        t0 = t0.to(self.y0.device, self.dtype)
        self._before_integrate(t0.reshape(-1))
        event_time, y1 = self._advance_until_event(event_fn)
        solution = torch.stack([self.y0, y1], dim=0)
        return event_time, solution


class FixedGridODESolver(metaclass=abc.ABCMeta):
    order: int

    def __init__(
            self,
            func,
            y0,
            step_size=None,
            grid_constructor=None,
            interp="linear",
            perturb=False,
            jump_t=None,
            events=None,
            jump=None,
            jump_mechanism=None,
            **unused_kwargs,
    ):
        self.atol = unused_kwargs.pop("atol")
        unused_kwargs.pop("rtol", None)
        unused_kwargs.pop("norm", None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        self.device = y0.device
        self.step_size = step_size
        self.interp = interp
        self.perturb = perturb
        # NB: jump_t is a *time*, so it must not be typed from the state dtype -
        # doing so promotes the time grid to complex for a complex-valued state.
        # It is cast to the grid dtype in `_merge_grid_points`.
        self.jump_t = (
            torch.as_tensor(jump_t, device=self.device) if jump_t is not None else None
        )
        self.jump = jump
        self.events = (
            torch.as_tensor(events, dtype=self.y0.dtype, device=self.device)
            if events is not None
            else None
        )
        self.jump_mechanism = _resolve_jump_mechanism(
            jump=jump,
            jump_t=self.jump_t,
            events=self.events,
            jump_mechanism=jump_mechanism,
        )

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, y0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError(
                    "step_size and grid_constructor are mutually exclusive arguments."
                )

    @classmethod
    def valid_callbacks(cls):
        return {"callback_step"}

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = (
                    torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size
                    + start_time
            )
            t_infer[-1] = t[-1]

            return t_infer

        return _grid_constructor

    @abc.abstractmethod
    def _step_func(self, func, t0, dt, t1, y0):
        pass

    @property
    def _has_jumps(self):
        return self.jump is not None and self.jump_mechanism is not None

    def _merge_grid_points(self, time_grid):
        """Fold prescribed discontinuity/event times into the integration grid.

        Landing a step boundary exactly on every known event time is what keeps
        the jump exact and the solver at its nominal order. When there is
        nothing to merge the grid is returned untouched, so a solve without
        jumps behaves exactly as it did before jump support existed - in
        particular it stays differentiable with respect to `t`.
        """
        extra = []
        for candidate in (
            self.jump_t,
            None if self.jump_mechanism is None else self.jump_mechanism.event_times,
        ):
            if candidate is not None and candidate.numel() > 0:
                extra.append(
                    torch.as_tensor(candidate).to(
                        dtype=time_grid.dtype, device=time_grid.device
                    )
                )
        if not extra:
            return time_grid

        extra = torch.cat(extra)
        # Points outside the integration interval are not ours to integrate
        # through; the endpoints are already in the grid.
        extra = extra[(extra > time_grid[0]) & (extra < time_grid[-1])]
        if extra.numel() == 0:
            return time_grid

        merged, _ = torch.sort(torch.cat([time_grid, extra]))
        # De-duplicate without `unique`, which has no derivative.
        keep = torch.ones(merged.shape, dtype=torch.bool, device=merged.device)
        keep[1:] = merged[1:] != merged[:-1]
        return merged[keep]

    def _interpolant(self, t0, t1, y0, y1, f0):
        """Dense output over ``[t0, t1]`` for the step that produced ``y1``."""
        if self.interp == "linear":
            return lambda t: self._linear_interp(t0, t1, y0, y1, t)
        if self.interp == "cubic":
            f1 = self.func(t1, y1)
            return lambda t: self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t)
        raise ValueError(f"Unknown interpolation method {self.interp}")

    def integrate(self, t):
        time_grid = self._merge_grid_points(
            self.grid_constructor(self.func, self.y0, t)
        )
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = torch.empty(
            len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device
        )

        mechanism = self.jump_mechanism if self._has_jumps else None
        y0 = self.y0
        # Cadlag: an event sitting on the very first time point is already
        # reflected in the value reported there.
        if mechanism is not None and mechanism.has_event_at(time_grid[0]):
            y0 = mechanism.apply_jump(time_grid[0], y0, self.jump)
        solution[0] = y0

        j = 1
        for t_start, t_end in zip(time_grid[:-1], time_grid[1:]):
            t0 = t_start
            n_events = 0
            while t0 < t_end:
                t1 = t_end
                dt = t1 - t0
                self.func.callback_step(t0, y0, dt)
                dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
                y1 = y0 + dy
                interp = self._interpolant(t0, t1, y0, y1, f0)

                t_event = None
                if mechanism is not None:
                    t_event = mechanism.next_event_time(interp, t0, t1)
                    if t_event is not None and t_event < t1:
                        # A state-dependent event fell strictly inside the step.
                        # Retake the step so that it ends exactly on the event:
                        # the solver then integrates up to the discontinuity at
                        # full order and restarts from the post-jump state.
                        t1 = t_event
                        dt = t1 - t0
                        dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
                        y1 = y0 + dy
                        interp = self._interpolant(t0, t1, y0, y1, f0)

                # Output times strictly inside the step come from the dense
                # output; the endpoint is handled below so that it can carry the
                # post-jump value.
                while j < len(t) and t[j] < t1:
                    solution[j] = interp(t[j])
                    j += 1

                if t_event is not None:
                    y1 = mechanism.apply_jump(t1, y1, self.jump)
                    n_events += 1
                    if n_events > MAX_EVENTS_PER_STEP:
                        raise RuntimeError(
                            f"More than {MAX_EVENTS_PER_STEP} events realised in "
                            f"the single step [{float(t_start)}, {float(t_end)}]; "
                            "the jump mechanism is not making progress."
                        )

                if j < len(t) and t[j] == t1:
                    solution[j] = y1
                    j += 1

                t0, y0 = t1, y1

        return solution

    def integrate_until_event(self, t0, event_fn):
        assert (
                self.step_size is not None
        ), "Event handling for fixed step solvers currently requires `step_size` to be provided in options."

        t0 = t0.type_as(self.y0.abs())
        y0 = self.y0
        dt = self.step_size

        sign0 = torch.sign(event_fn(t0, y0))
        max_itrs = 20000
        itr = 0
        while True:
            itr += 1
            t1 = t0 + dt
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            sign1 = torch.sign(event_fn(t1, y1))

            if sign0 != sign1:
                if self.interp == "linear":
                    interp_fn = lambda t: self._linear_interp(t0, t1, y0, y1, t)
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    interp_fn = lambda t: self._cubic_hermite_interp(
                        t0, y0, f0, t1, y1, f1, t
                    )
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                event_time, y1 = find_event(
                    interp_fn, sign0, t0, t1, event_fn, float(self.atol)
                )
                break
            else:
                t0, y0 = t1, y1

            if itr >= max_itrs:
                raise RuntimeError(f"Reached maximum number of iterations {max_itrs}.")
        solution = torch.stack([self.y0, y1], dim=0)
        return event_time, solution

    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = t1 - t0
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)
