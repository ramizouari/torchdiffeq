import abc
import bisect
from abc import abstractmethod
from typing import Callable, Literal, Optional, Sequence

import torch

### OURS


def _scalar(t) -> float:
    """Time value as a plain float, detached from any autograd graph."""
    if isinstance(t, torch.Tensor):
        return t.detach().item()
    return float(t)

JumpMechanismKind = Literal["coupled", "simple", "marked"]

#: Hard cap on the number of events realised inside a single solver step. Guards
#: against a mechanism that keeps reporting an event at the same instant.
MAX_EVENTS_PER_STEP = 1024


def standard_exponential_random(
    shape: Sequence[int], generator: torch.Generator = None
):
    """
    Generates random numbers from a standard exponential distribution.

    This function creates random samples from a standard exponential
    distribution (the exponential distribution with λ = 1). It uses
    PyTorch functions to compute the negative logarithm of uniformly
    distributed random numbers.

    Args:
        shape (Sequence[int]): The shape of the output tensor
            defining how many samples are generated and their dimensions.
        generator (torch.Generator): The random number generator to use.
            If None, a new generator is created.

    Returns:
        Tensor: A tensor of random samples from the standard exponential
        distribution with the specified shape.
    """
    return -torch.log(torch.rand(shape, generator=generator))


class JumpMechanism(abc.ABC):
    """
    Handles operations related to jump mechanisms in a given context.

    A jump mechanism answers three questions for the solver:

    1. *When* does the next event happen inside a step? -- :meth:`next_event_time`.
    2. *Which* event streams fire at that instant? -- :meth:`realise_event`.
    3. *Where* in the state does the jump land? -- :meth:`latent_proj` /
       :meth:`scatter_jump`.

    The solver drives the mechanism as follows. Event times are half-open on the
    left: a scan of ``(t0, t1]`` never re-reports an event already realised at
    ``t0``. The state is treated as càdlàg (right-continuous), so the value
    reported at an event time is the *post*-jump state.

    Attributes:
        batch_dims (int): The number of batch dimensions in the system.
        events_t (list): A list to store event times.

    Notes:
        The jump is applied to the system after the latent projection.
    """

    def __init__(
        self,
        batch_dims: int,
        latent_proj: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        """
        Initializes the instance with the provided batch dimensions, marking flag,
        and latent projection function.

        Args:
            batch_dims (int): The number of batch dimensions to be used.
            latent_proj (Callable[[torch.Tensor], torch.Tensor], optional): A function
                that applies a projection to a latent tensor. Defaults to the identity
                function if not provided.
        """
        self.batch_dims = batch_dims
        self.events_t = []
        self._latent_proj = latent_proj if latent_proj is not None else lambda x: x

    # ------------------------------------------------------------------ #
    #                        Event discovery                             #
    # ------------------------------------------------------------------ #

    @property
    def event_times(self) -> Optional[torch.Tensor]:
        """Event times known ahead of the solve, if any.

        The solver merges these into its integration grid so that a step
        boundary falls exactly on every event, which makes the jump exact and
        keeps the solver at its nominal order. Return ``None`` when the event
        times are state-dependent and can only be found by root-finding.
        """
        return None

    def has_event_at(self, t: torch.Tensor) -> bool:
        """Whether an unrealised event sits exactly at time ``t``.

        Only used for the very first time point of a solve; every later event
        is discovered through :meth:`next_event_time` on a half-open interval.
        """
        return False

    def next_event_time(
        self,
        z_interp: Callable[[torch.Tensor], torch.Tensor],
        t0: torch.Tensor,
        t1: torch.Tensor,
        tol: float = 1e-6,
    ) -> Optional[torch.Tensor]:
        """Earliest event time in the half-open interval ``(t0, t1]``.

        Args:
            z_interp: Interpolant of the state over ``[t0, t1]``.
            t0: Exclusive lower bound of the scan.
            t1: Inclusive upper bound of the scan.
            tol: Absolute tolerance for state-dependent event location.

        Returns:
            The event time, or ``None`` when no event occurs in the interval.
        """
        raise NotImplementedError

    def find_event_time(
        self,
        z_interp: Callable[[torch.Tensor], torch.Tensor],
        t0: torch.Tensor,
        t1: torch.Tensor,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """Deprecated. Use :meth:`next_event_time`.

        Kept for callers written against the original API. Returns ``t1`` when
        no event occurs, which cannot be distinguished from an event landing
        exactly on ``t1`` -- the reason it was replaced.
        """
        t1 = torch.as_tensor(t1)
        t_event = self.next_event_time(z_interp, t0, t1, tol)
        if t_event is None:
            return t1
        return torch.minimum(torch.as_tensor(t_event), t1)

    @abstractmethod
    def realise_event(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Consume the event(s) at time ``t`` and return their indicator mask.

        The mask has shape ``(*batch,)`` in ``simple`` mode and
        ``(*batch, n_event_types)`` in ``coupled`` mode.
        """

    # ------------------------------------------------------------------ #
    #                        Applying the jump                           #
    # ------------------------------------------------------------------ #

    def latent_proj(self, z: torch.Tensor, enforce: bool = False) -> torch.Tensor:
        """
        Projects the input latent tensor using a projection method.

        Note:
            The Jump mechanism alters the latent projection part only, so the
            projection must be a *view* of ``z`` (or the identity) for
            :meth:`scatter_jump` to be able to write through it.

        Args:
            z (torch.Tensor): The input latent tensor to be projected.
            enforce (bool, optional): Whether to require that the projection
                aliases ``z`` rather than copying it.

        Returns:
            torch.Tensor: The projected latent tensor.

        Raises:
            RuntimeError: If ``enforce`` is set and the projection does not
                alias the input tensor.
        """
        z_ = self._latent_proj(z)
        if enforce and z_ is not z and z_._base is not z:
            raise RuntimeError(
                "The latent projection must be a view of its input (or the "
                "identity), otherwise the jump cannot be written back into "
                "the state."
            )
        return z_

    def scatter_jump(self, y: torch.Tensor, jmp: torch.Tensor) -> torch.Tensor:
        """Return ``y`` with ``jmp`` added to its latent projection.

        Coordinates outside the latent projection -- the cumulative hazards of a
        survival model, for instance -- are left untouched.
        """
        dJ = torch.zeros_like(y)
        dJ_proj = self.latent_proj(dJ, enforce=True)
        dJ_proj += jmp
        return y + dJ

    def increment_from_mask(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        jump_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        dN: torch.Tensor,
    ) -> torch.Tensor:
        """The state increment produced by an already-decided event mask.

        Split out from :meth:`jump_increment` so that a recorded mask can be
        replayed without re-deciding which events fire -- which is what makes
        the adjoint pass reproduce the forward pass exactly.
        """
        kind = self.kind
        if kind == "marked":
            raise NotImplementedError(
                "Marked jump mechanisms are not implemented yet."
            )
        h = jump_fn(t, self.latent_proj(y))
        dN = dN.to(dtype=h.dtype, device=h.device)
        if kind == "coupled":
            jmp = (h * dN[..., None, :]).sum(dim=-1)
        else:
            jmp = h * dN[..., None]
        dJ = torch.zeros_like(y)
        dJ_proj = self.latent_proj(dJ, enforce=True)
        dJ_proj += jmp
        return dJ

    def jump_increment(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        jump_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Realise the events at ``t`` and return the increment ``y(t+) - y(t-)``.

        Args:
            t: The event time.
            y: The pre-jump state ``y(t-)``.
            jump_fn: The jump network ``h(t, z)``. It is evaluated on the latent
                projection of ``y`` and must return ``(*batch, latent)`` in
                ``simple`` mode or ``(*batch, latent, n_event_types)`` in
                ``coupled`` mode.

        Returns:
            The state increment, zero outside the latent projection.
        """
        return self.increment_from_mask(t, y, jump_fn, self.realise_event(t, y))

    def apply_jump(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        jump_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Realise the events at ``t`` and return the post-jump state ``y(t+)``."""
        return y + self.jump_increment(t, y, jump_fn)

    @property
    @abstractmethod
    def kind(self) -> JumpMechanismKind:
        pass


class FixedJumpMechanism(JumpMechanism):
    """FixedJumpMechanism implements a jump mechanism with fixed event times.

    This class is designed to handle events occurring at fixed points in time.
    It provides methods to identify the next event time within a given interval
    and to realize the impact of the events at specific times. Additionally, it
    maintains the state of all realized events up to a certain index.

    Because the event times are known ahead of the solve they are reported
    through :attr:`event_times` and merged into the solver's integration grid,
    so each jump lands exactly on a step boundary.

    Attributes:
        events_mask (torch.Tensor): Mask representing the occurrence of events.
        events_t (torch.Tensor): Tensor containing the times at which events occur.
        idx (int): Index of the next event to be processed in the sequence.
    """

    def __init__(
        self,
        events_t: torch.Tensor,
        events_mask: torch.Tensor,
        coupled: bool = False,
        latent_proj: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        if coupled:
            batch_dims = events_mask.dim() - 2
        else:
            batch_dims = events_mask.dim() - 1
        super().__init__(batch_dims, latent_proj=latent_proj)
        events_t = torch.as_tensor(events_t)
        if events_t.dim() != 1:
            raise ValueError("events_t must be a one-dimensional tensor.")
        if events_t.numel() > 1 and (events_t[1:] < events_t[:-1]).any():
            raise ValueError("events_t must be non-decreasing.")
        n_times = events_mask.shape[-2] if coupled else events_mask.shape[-1]
        if n_times != events_t.numel():
            raise ValueError(
                "events_mask must carry one entry per event time: expected "
                f"{events_t.numel()} along the time axis, got {n_times}."
            )
        self.events_mask = events_mask
        self.events_t = events_t
        # Cached python floats: bisect on a tensor would rebuild this list on
        # every call, which is O(n) inside the per-step event loop.
        self._times = events_t.tolist()
        self.idx = 0
        self.coupled = coupled

    @property
    def event_times(self) -> Optional[torch.Tensor]:
        """The prescribed event times."""
        return self.events_t

    def has_event_at(self, t: torch.Tensor) -> bool:
        """Whether an unrealised event sits exactly at ``t``."""
        tv = _scalar(t)
        idx = bisect.bisect_left(self._times, tv, lo=self.idx)
        return idx < len(self._times) and self._times[idx] == tv

    def next_event_time(
        self,
        z_interp: Callable[[torch.Tensor], torch.Tensor],
        t0: torch.Tensor,
        t1: torch.Tensor,
        tol: float = 1e-6,
    ) -> Optional[torch.Tensor]:
        """
        Returns the earliest prescribed event time in ``(t0, t1]``.

        The interpolant is unused: the event times do not depend on the state.

        Args:
            z_interp: Ignored.
            t0: Exclusive lower bound of the scan.
            t1: Inclusive upper bound of the scan.
            tol: Ignored.

        Returns:
            The event time, or ``None`` when no prescribed event falls in the
            interval.
        """
        t0 = torch.as_tensor(t0)
        t1 = torch.as_tensor(t1)
        idx = bisect.bisect_right(self._times, _scalar(t0), lo=self.idx)
        if idx >= len(self._times):
            return None
        t_event = self.events_t[idx].to(device=t0.device)
        if t_event > t1:
            return None
        return t_event

    def realise_event(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Realises the event(s) prescribed at time ``t`` and advances the internal
        pointer past them, so the same event can never be applied twice.

        Args:
            t: A tensor representing the time at which to evaluate the event.
            z: A tensor representing the state associated with the event.

        Returns:
            A tensor representing the realised event mask at the given time. If
            no event is prescribed at ``t``, a zero mask is returned.
        """
        tv = _scalar(t)
        idx = bisect.bisect_left(self._times, tv, lo=self.idx)
        if idx >= len(self._times) or self._times[idx] != tv:
            return torch.zeros_like(self._mask_row(0))
        mask = self._mask_row(idx)
        stop = idx + 1
        # Several rows may share the same timestamp; consume all of them.
        while stop < len(self._times) and self._times[stop] == tv:
            mask = mask + self._mask_row(stop)
            stop += 1
        self.idx = stop
        return mask

    def _mask_row(self, i: int) -> torch.Tensor:
        """The dense event mask prescribed at time index ``i``."""
        row = (
            self.events_mask[..., i, :]
            if self.coupled
            else self.events_mask[..., i]
        )
        return row.to_dense() if row.is_sparse else row

    @property
    def all_realised_events(self):
        """The mask rows consumed so far, sliced along the time axis."""
        if self.coupled:
            return self.events_mask[..., : self.idx, :]
        return self.events_mask[..., : self.idx]

    @property
    def kind(self) -> JumpMechanismKind:
        return "simple" if not self.coupled else "coupled"


class CoupledStochasticJumpMechanism(JumpMechanism):
    """
    Class representing a coupled stochastic jump mechanism.

    This class handles the simulation of stochastic events based on cumulative hazard
    projections. It supports functionalities like detecting events, finding their times,
    and realizing events with support for batch processing and optional coupling. The
    mechanism operates under the assumption of one-dimensional batch shapes and limited
    event constraints.

    Notes:
        - In the case of coupled events, the mechanism assumes that each event has its own
        hazard rate.
        - The coupled mechanism only supports discrete events with finite types.


    Attributes:
        cumulative_hazard_proj (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
            A function to project cumulative hazards, mapping input tensors to an
            adjusted tensor. Defaults to an identity mapping.
        rng (torch.Generator): The random number generator used for event simulations,
            enabling reproducibility.
    """

    def __init__(
        self,
        batch_shape: Sequence[int],
        coupling_dim: int = 0,
        cumulative_hazards_proj: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = None,
        rng: torch.Generator = None,
        latent_proj: Callable[[torch.Tensor], torch.Tensor] = None,
        max_events: torch.Tensor | int | float = torch.inf,
    ):
        """
        Initializes an instance of the class with specified configurations for stochastic jump mechanisms. Handles
        initial setup of attributes and validation of input parameters.

        Args:
            batch_shape (Sequence[int]): The shape of the batch as a sequence of integers. Only supports up to
                one-dimensional batch shapes.
            cumulative_hazards_proj (Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional): A projection
                function to adjust cumulative hazards, taking in two tensors and returning a tensor. Defaults to
                an identity function that maps the input z to itself.
            rng (torch.Generator, optional): A PyTorch random number generator for reproducibility. Defaults to a
                newly initialized `torch.Generator` instance if not specified.
            coupling_dim (int, optional): The dimension of the coupled intensity. Defaults to 0.

        Raises:
            ValueError: If the `batch_shape` argument specifies more than one dimension in its batch, which is not
                supported by this mechanism.
        """
        super().__init__(len(batch_shape), latent_proj=latent_proj)
        if len(batch_shape) > 1:
            raise ValueError(
                "CoupledStochasticJumpMechanism does not support batched events with more than one dimension."
            )
        self.cumulative_hazard_proj = (
            cumulative_hazards_proj
            if cumulative_hazards_proj is not None
            else lambda _, z: z
        )
        if len(batch_shape) == 0:
            self._realised_events = []
        else:
            self._realised_events = [[] for _ in range(batch_shape[0])]
        self.rng = rng if rng is not None else torch.Generator()
        self.coupling_dim = coupling_dim
        coupling_shape = [coupling_dim] if self.coupling_dim > 0 else []
        event_realisation = standard_exponential_random(
            list(batch_shape) + coupling_shape, generator=self.rng
        )
        max_events: torch.Tensor = torch.as_tensor(
            max_events, device=event_realisation.device
        )
        rnk_diff = event_realisation.dim() - max_events.dim()
        max_events = max_events.view((1,) * rnk_diff + max_events.shape).expand(
            event_realisation.shape
        )
        self.max_events = max_events
        self._event_counter = torch.zeros_like(self.max_events)
        self._event_realisation = torch.where(
            self._event_counter < max_events, event_realisation, torch.inf
        )
        # (time, crossing mask) memoised by the last successful event location.
        self._located = None

    def detect_event(self, t: torch.Tensor, z: torch.Tensor) -> bool:
        """Whether any cumulative hazard has crossed its threshold at ``t``."""
        return (
            (
                self.cumulative_hazard_proj(t, z)
                >= self._event_realisation.to(device=z.device)
            )
            .any()
            .item()
        )

    def next_event_time(
        self,
        z_interp: Callable[[torch.Tensor], torch.Tensor],
        t0: torch.Tensor,
        t1: torch.Tensor,
        tol: float = 1e-6,
    ) -> Optional[torch.Tensor]:
        """
        Locates the first threshold crossing in ``(t0, t1]`` by bisection.

        The cumulative hazard is non-decreasing, so a crossing has happened
        somewhere in the interval if and only if it has happened by ``t1``. That
        makes a single check at ``t1`` a sound test for "is there an event at
        all", and the bisection below only runs when there is one.

        Args:
            z_interp: Interpolant of the state over ``[t0, t1]``.
            t0: Exclusive lower bound of the scan.
            t1: Inclusive upper bound of the scan.
            tol: Absolute tolerance on the located event time.

        Returns:
            The event time, or ``None`` when no threshold is crossed.
        """
        t0 = torch.as_tensor(t0)
        t1 = torch.as_tensor(t1)
        self._located = None
        if not self.detect_event(t1, z_interp(t1)):
            return None
        lo, hi = t0, t1
        # Bisection on a monotone predicate: hi always satisfies it, lo never
        # does, so the loop converges to the crossing from the right.
        while (hi - lo) > tol:
            mid = (lo + hi) / 2
            if mid <= lo or mid >= hi:  # exhausted floating point resolution
                break
            if self.detect_event(mid, z_interp(mid)):
                hi = mid
            else:
                lo = mid
        # Memoise which streams crossed. The solver may retake the step so that
        # it ends exactly here, and the recomputed state can then sit a hair on
        # the wrong side of the threshold; the located crossing is authoritative.
        z_hi = z_interp(hi)
        crossed = self.cumulative_hazard_proj(hi, z_hi) >= self._event_realisation.to(
            device=z_hi.device
        )
        self._located = (hi, crossed)
        return hi

    def realise_event(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Consume the crossings at ``t`` and resample their thresholds."""
        if self._located is not None and self._located[0] == t:
            realised = self._located[1]
            self._located = None
        else:
            realised = self.cumulative_hazard_proj(t, z) >= self._event_realisation.to(
                device=z.device
            )
        indexes = torch.argwhere(realised)
        if len(indexes) == 0:
            raise ValueError("No event detected during realisation.")
        if self.batch_dims == 0:
            if self.coupled:  # Record which event type fired
                for idx in indexes:
                    self._realised_events.append([t, idx.item()])
                    self._event_counter[idx.item()] += 1
            else:  # Otherwise just record the event time
                self._realised_events.append(t)
                self._event_counter += 1
        else:
            for idx in indexes:
                if self.coupled:  # Record which event type fired
                    self._realised_events[idx[0].item()].append([t, idx[1].item()])
                else:  # Otherwise just record the event time
                    self._realised_events[idx.item()].append(t)
                self._event_counter[
                    tuple(idx.to(device=self._event_counter.device))
                ] += 1

        delta_event = realised.float().to(
            device=self._event_realisation.device
        ) * standard_exponential_random(realised.shape, generator=self.rng)
        # Sample a threshold for the next event, or +inf if events are exhausted.
        self._event_realisation += torch.where(
            self._event_counter < self.max_events, delta_event, torch.inf
        )
        return realised

    @property
    def all_realised_events(self):
        if self.batch_dims == 0:
            return torch.tensor(self._realised_events)
        else:
            return torch.nested.nested_tensor(
                [torch.as_tensor(e) for e in self._realised_events], layout=torch.jagged
            )

    @property
    def coupled(self):
        return self.coupling_dim > 0

    @property
    def kind(self) -> JumpMechanismKind:
        return "simple" if not self.coupled else "coupled"


class SimpleStochasticJumpMechanism(CoupledStochasticJumpMechanism):
    """
    Implements a simple stochastic jump mechanism with customization options for
    projections and random number generation.

    This class is a specialized version of CoupledStochasticJumpMechanism,
    providing stochastic modeling capabilities for jumps in a process. It allows
    users to define custom projections for cumulative hazards and latent states,
    and to specify a random number generator for stochastic behavior. The
    mechanism is initialized with a specific batch shape, facilitating handling
    of tensor dimensions in stochastic computations.
    """

    def __init__(
        self,
        batch_shape: Sequence[int],
        cumulative_hazards_proj: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = None,
        rng: torch.Generator = None,
        latent_proj: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__(
            batch_shape,
            coupling_dim=0,
            cumulative_hazards_proj=cumulative_hazards_proj,
            rng=rng,
            latent_proj=latent_proj,
        )
