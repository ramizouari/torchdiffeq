import abc
from functools import partial

import torch
from .event_handling import find_event
from .misc import _handle_unused_kwargs
from .jump import JumpMechanism


def next_after(x: torch.Tensor) -> torch.Tensor:
    """
    Returns the next representable floating-point value after x in the direction of infinity.
    """
    return torch.nextafter(x, torch.tensor(torch.inf, device=x.device, dtype=x.dtype))


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
        solution[0] = self.y0
        t = t.to(self.dtype)
        self._before_integrate(t)
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
        self.jump_t = (
            torch.as_tensor(jump_t, dtype=self.dtype, device=self.device)
            if jump_t is not None
            else torch.tensor([], dtype=self.dtype, device=self.device)
        )
        self.jump = jump
        self.jump_mechanism = jump_mechanism
        self.events = (
            torch.as_tensor(events, dtype=self.y0.dtype, device=self.device)
            if events is not None
            else None
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
        return self.jump is not None or self.jump_mechanism is not None

    def integrate(self, t):
        event_index = 0
        time_grid = self.grid_constructor(self.func, self.y0, t)
        time_grid = torch.concat([time_grid, self.jump_t], dim=0)
        time_grid, _ = time_grid.sort()
        time_grid = time_grid.unique()
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = torch.empty(
            len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device
        )
        solution[0] = self.y0

        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dJ = torch.zeros_like(y0)
            dt = t1 - t0
            self.func.callback_step(t0, y0, dt)
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy
            t_event = t0
            while j < len(t) and t1 >= t[j]:
                if self.interp == "linear":
                    if self._has_jumps:
                        _, t_event, event_index = self._linear_interp_events(
                            t0, t1, y0, y1, next_after(t_event), t[j], event_index, dJ
                        )
                    solution[j] = self._linear_interp(t0, t1, y0, y1, t[j]) + dJ
                    if self._has_jumps:
                        _, t_event, event_index = self._linear_interp_events(
                            t0,
                            t1,
                            y0,
                            y1,
                            next_after(t_event),
                            next_after(t1),
                            event_index,
                            dJ,
                        )
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    if self._has_jumps:
                        _, t_event, event_index = self._cubic_hermite_events(
                            t0, y0, f0, t1, y1, f1, t_event, t[j], event_index
                        )
                    solution[j] = (
                        self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t[j]) + dJ
                    )
                    if self._has_jumps:
                        _, t_event, event_index = self._cubic_hermite_events(
                            t0, y0, f0, t1, y1, f1, t_event, next_after(t1), event_index
                        )

                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                j += 1
            y0 = y1 + dJ

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

    def _linear_interp_events(
        self, t0, t1, y0, y1, t_start, t_end, e_index, dJ: torch.Tensor
    ):
        def _linear_interp_with_jump(*args, dJ):
            return self._linear_interp(*args) + dJ

        # interp = partial(self._linear_interp, t0, t1, y0, y1)
        # dJ = torch.zeros_like(y0)
        interp = partial(_linear_interp_with_jump, t0, t1, y0, y1, dJ=dJ)
        if isinstance(self.jump_mechanism, JumpMechanism):
            t_event = self.jump_mechanism.find_event_time(interp, t_start, t_end)
            while t_event < t_end:
                dN = self.jump_mechanism.realise_event(t_event, interp(t_event))
                dN = (
                    dN[..., None, :]
                    if self.jump_mechanism.kind == "coupled"
                    else dN[..., None]
                )
                # If marked, shape of dN is (B...,n_events), else (B...)
                y_event = interp(t_event)
                # z.shape = (B...,L) Where L is the latent dimension
                # If marked, shape of h is (B...,L,n_events), else (B...,L)
                h = self.jump(
                    t_event, self.jump_mechanism.latent_proj(y_event, enforce=False)
                )
                # If marked, the einsum will be a dot-like product, else a pointwise multiplication
                jmp = h * dN
                if self.jump_mechanism.kind == "coupled":
                    jmp = jmp.sum(dim=-1)
                # Apply the projection to the latent space. Projection MUST BE A VIEW OPERATION.
                dJ_proj = self.jump_mechanism.latent_proj(dJ, enforce=True)
                dJ_proj += jmp
                t_event = self.jump_mechanism.find_event_time(
                    interp, next_after(t_event), t_end
                )
            return dJ, t_event, None
        else:
            while e_index < len(self.jump_t) and t_end >= self.jump_t[e_index]:
                y_event = interp(self.jump_t[e_index])
                h = self.jump(self.jump_t[e_index], y_event)
                dN = self.events[:, e_index]
                shape = list(dN.shape) + [
                    1 for _ in range(len(h.shape) - len(dN.shape))
                ]
                dJ += h * dN.reshape(shape)
                e_index += 1
            return dJ, None, e_index

    def _cubic_hermite_events(self, t0, y0, f0, t1, y1, f1, t_start, t_end, e_index):
        interp = partial(self._cubic_hermite_interp, t0, y0, f0, t1, y1, f1)
        dJ = torch.zeros_like(y0)

        if isinstance(self.jump_mechanism, JumpMechanism):
            t_event = self.jump_mechanism.find_event_time(interp, t_start, t_end)
            while t_event < t_end:
                dN = self.jump_mechanism.realise_event(t_event, interp(t_event))
                dN = (
                    dN[..., None, :]
                    if self.jump_mechanism.kind == "coupled"
                    else dN[..., None]
                )
                y_event = interp(t_event)
                h = self.jump(
                    t_event, self.jump_mechanism.latent_proj(y_event, enforce=False)
                )
                jmp = h * dN
                if self.jump_mechanism.kind == "coupled":
                    jmp = jmp.sum(dim=-1)
                dJ_proj = self.jump_mechanism.latent_proj(dJ, enforce=True)
                dJ_proj += jmp[..., None]
                t_event = self.jump_mechanism.find_event_time(
                    interp, next_after(t_event), t_end
                )
            return dJ, t_event, e_index
        else:
            while e_index < len(self.jump_t) and t_end >= self.jump_t[e_index]:
                y_event = interp(self.jump_t[e_index])
                h = self.jump(self.jump_t[e_index], y_event)
                dN = self.events[:, e_index]
                shape = list(dN.shape) + [
                    1 for _ in range(len(h.shape) - len(dN.shape))
                ]
                dJ += h * dN.reshape(shape)
                e_index += 1
            return dJ, None, e_index
