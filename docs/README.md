# Jump ODE documentation

Support for solving — and differentiating through — ODEs whose solution is
piecewise continuous:

```
dy/dt = f(t, y)                between events
y(t⁺) = y(t⁻) + h(t, y(t⁻))    at each event time t
```

This is the object underlying neural jump and marked-point-process models:
smooth dynamics punctuated by discrete events that displace the state. It also
covers ordinary hybrid systems — a bouncing ball, a dosing schedule, a queue
that empties on arrival.

## Start here

```python
from torchdiffeq import odeint
from torchdiffeq.jump import FixedJumpMechanism

# One dose at t = 1.5 and t = 3.5, added to a decaying state.
mechanism = FixedJumpMechanism(
    torch.tensor([1.5, 3.5]),          # event times
    torch.ones(batch, 2),              # which of them fire, per batch element
)
y = odeint(
    decay, y0, t, method="dopri5",
    options={"jump": dose, "jump_mechanism": mechanism},
)
```

`jump` is the map `h`; `jump_mechanism` decides *when* events fire. Everything
else about `odeint` is unchanged.

## The documents

| | |
| --- | --- |
| [**mathematics.md**](mathematics.md) | The model, the two event mechanisms as maths, why a jump must land on a step boundary, how state-dependent events are located, and how the adjoint crosses a jump it cannot invert. |
| [**api.md**](api.md) | Reference for the options, the three built-in mechanisms, the `JumpMechanism` protocol for writing your own, and `odeint_jump_adjoint`. |
| [**examples.md**](examples.md) | Six worked examples, each one runnable as-is: prescribed events, latent projections, stochastic sampling, competing risks, adjoint training, and a custom contact mechanism. |
| [**solvers.md**](solvers.md) | Which solver to pick. Support matrix, measured convergence orders, the multistep caveat, and cost. |
| [**compatibility.md**](compatibility.md) | What changed for code that does *not* use jumps, and how to migrate from the older `jump_t`+`events` spelling. |
| [**troubleshooting.md**](troubleshooting.md) | Every error this feature raises, what causes it, and what to do. |

Shorter reference material lives with the rest of the library:
the [README](../README.md#jump-odes) for the one-paragraph version and
[FURTHER_DOCUMENTATION](../FURTHER_DOCUMENTATION.md#jump-options) for the option
list alongside the other solver options.

## Conventions

The solution is **càdlàg** — right-continuous with left limits. The value
reported at an event time is the *post*-jump state, so `y[i]` for `t[i] == t*`
is `y(t*⁺)`. This is applied consistently by every solver and by the adjoint.

Jump ODEs require **increasing `t`**. The jump map is not invertible in general,
so integrating one backwards is not a matter of flipping signs; a decreasing `t`
raises rather than returning a plausible wrong answer. See
[compatibility.md](compatibility.md#decreasing-t-is-refused).
