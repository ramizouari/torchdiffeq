import abc
from abc import abstractmethod

import torch
import bisect
from typing import Sequence, Callable, Literal

### OURS

JumpMechanismKind = Literal["coupled", "simple", "marked"]


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

    Provides functionalities to find event times and realize events,
    allowing dynamic computations involving time and states.

    Attributes:
        batch_dims (int): The number of batch dimensions in the system.
        events_t (list): A list to store event times.
        marked (bool): Indicates if the mechanism is marked, affecting event handling.

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
            marked (bool): Indicates whether the instance is marked. Defaults to False.
            latent_proj (Callable[[torch.Tensor], torch.Tensor], optional): A function
                that applies a projection to a latent tensor. Defaults to the identity
                function if not provided.
        """
        self.batch_dims = batch_dims
        self.events_t = []
        self._latent_proj = latent_proj if latent_proj is not None else lambda x: x

    def find_event_time(
        self,
        z_interp: Callable[[torch.Tensor], torch.Tensor],
        t0: torch.Tensor,
        t1: torch.Tensor,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def realise_event(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        pass

    def latent_proj(self, z: torch.Tensor, enforce: bool = False) -> torch.Tensor:
        """
        Projects the input latent tensor using a projection method and verifies
        that the projection does not alter the base tensor.

        Note:
            The Jump mechanism alters the latent projection part.

        Args:
            z (torch.Tensor): The input latent tensor to be projected.
            enforce (bool, optional): Whether to enforce that the projection does not alter the base tensor.

        Returns:
            torch.Tensor: The projected latent tensor.

        Raises:
            RuntimeError: If the latent projection changes the base tensor.
        """
        z_ = self._latent_proj(z)
        if enforce and (z_ is z or z_._base is not z):
            raise RuntimeError(
                "The latent projection should not change the base Tensor"
            )
        return z_

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

    Attributes:
        events_mask (torch.Tensor): Mask representing the occurrence of events.
        events_t (torch.Tensor): Tensor containing the times at which events occur.
        idx (int): Index of the next event to be processed in the sequence.
        marked (bool): Flag indicating if the mechanism is marked.
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
        self.events_mask = events_mask
        self.events_t = events_t
        self.idx = 0
        self.coupled = coupled

    def find_event_time(
        self,
        z_interp: Callable[[torch.Tensor], torch.Tensor],
        t0: torch.Tensor,
        t1: torch.Tensor,
        tol: float = 1e-6,
    ):
        """
        Finds the event time within a specified time range, interpolating using the
        provided function and ensuring the time lies within the given tolerance.

        This method searches for an event time within the range [t0, t1] using
        binary search. The function `z_interp` is used to calculate interpolations
        and enforce the tolerance constraint.

        If no suitable event is found within the specified range, the method returns
        the upper bound `t1`.

        Args:
            z_interp: A callable interpolation function that accepts a tensor as input
                and returns a tensor, typically used for interpolating event values.
            t0: A tensor representing the start of the time range within which to
                search for the event.
            t1: A tensor representing the end of the time range within which to
                search for the event.
            tol: A float specifying the allowed tolerance for the event time. Defaults
                to 1e-6.

        Returns:
            torch.Tensor: The event time found within the specified range, either the
            closest match from `events_t` for the given tolerance or the upper bound
            `t1` if no suitable event is found.
        """
        t0 = torch.as_tensor(t0)
        t1 = torch.as_tensor(t1)
        idx = bisect.bisect_left(self.events_t.tolist(), t0, lo=self.idx)
        if idx == len(self.events_t):
            t = t1
        else:
            t = self.events_t[idx].to(device=t0.device)
        return torch.minimum(t, t1)

    def realise_event(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Realises an event based on the given time `t` and state `z`. This function determines
        the occurrence of an event by finding the corresponding index in the pre-defined
        event times and updates the internal pointer. If no events occur, it returns a
        tensor of zeros with the same shape as the first column of the event mask.

        Args:
            t: A tensor representing the time at which to evaluate the event.
            z: A tensor representing the state associated with the event.

        Returns:
            A tensor representing the realised event mask at the given time. If no event
            occurs, a tensor of zeros is returned with the appropriate shape.
        """
        idx = bisect.bisect_left(self.events_t.tolist(), t, lo=self.idx)
        self.idx = idx
        if self.coupled:
            if self.idx == len(self.events_t):
                return torch.zeros_like(self.events_mask[..., 0, :])
            return self.events_mask[..., idx, :].to_dense()

        else:
            if self.idx == len(self.events_t):
                return torch.zeros_like(self.events_mask[..., 0])
            return self.events_mask[..., idx].to_dense()

    @property
    def all_realised_events(self):
        return self.events_mask[..., : self.idx + 1]

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

    def detect_event(self, t: torch.Tensor, z: torch.Tensor) -> bool:
        return (
            (
                self.cumulative_hazard_proj(t, z)
                >= self._event_realisation.to(device=z.device)
            )
            .any()
            .item()
        )

    def _proj_fn(self, z_interp: Callable[[torch.Tensor], torch.Tensor]):
        def proj_fn(t: torch.Tensor):
            cumulative_hazards = self.cumulative_hazard_proj(t, z_interp(t))
            if cumulative_hazards.dim() == self.batch_dims:
                return cumulative_hazards
            elif cumulative_hazards.dim() == self.batch_dims + 1:
                raise ValueError("Currently, only one event is supported.")
            return cumulative_hazards

        return proj_fn

    def find_event_time(
        self,
        z_interp: Callable[[torch.Tensor], torch.Tensor],
        t0: torch.Tensor,
        t1: torch.Tensor,
        tol: float = 1e-6,
    ):
        t0 = torch.as_tensor(t0)
        t1 = torch.as_tensor(t1)
        while not torch.allclose(t0, t1, atol=tol):
            t_mid = (t0 + t1) / 2
            if self.detect_event(t_mid, z_interp(t_mid)):
                t1 = t_mid
            else:
                t0 = t_mid
        return t1

    def realise_event(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        realised = self.cumulative_hazard_proj(t, z) >= self._event_realisation.to(
            device=z.device
        )
        indexes = torch.argwhere(realised)
        if len(indexes) == 0:
            raise ValueError("No event detected during realisation.")
        if self.batch_dims == 0:
            if self.coupled:  # If marked, we need to store the event index
                for idx in indexes:
                    self._realised_events.append([t, idx.item()])
                    self._event_counter[idx.item()] += 1
            else:  # If not marked, we just append the event time
                self._realised_events.append(t)
                self._event_counter += 1
        else:  # Batch size = 1
            for idx in indexes:
                if self.coupled:  # If marked, we need to store the event index
                    self._realised_events[idx[0].item()].append([t, idx[1].item()])
                else:  # If not marked, we just append the event time
                    self._realised_events[idx.item()].append(t)
                self._event_counter[
                    tuple(idx.to(device=self._event_counter.device))
                ] += 1

        torch.where(self._event_counter >= self.max_events, 0, 0)

        delta_event = realised.float().to(
            device=self._event_realisation.device
        ) * standard_exponential_random(realised.shape, generator=self.rng)
        # Sample an event time for next event, or +inf if events are exhausted
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
