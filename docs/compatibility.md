# Compatibility and migration

## For code that does not use jumps

Jump support is inert unless `jump` and `jump_mechanism` (or `jump_t` +
`events`) are passed. A solve without them takes the original code path: the
integration grid is returned unrebuilt, no predicate is evaluated, and no
mechanism is consulted.

The pre-existing test suite is unchanged by this feature. `api_tests.py`,
`event_tests.py`, `norm_tests.py`, `gradient_tests.py` and `odeint_tests.py`
give 20 passed / 2 failed, and those two — `TestMinMaxStep::test_min_max_step`
and `TestEventHandling::test_odeint` — fail identically on `master` at the same
lines with the same values.

Three behaviours changed for everyone, all of them fixes:

**`method` may be a solver class.** Previously only a registered name was
accepted, despite the dispatch supporting a class.

```python
from torchdiffeq._impl.fixed_grid import Euler
odeint(func, y0, t, method=Euler, options={"step_size": 0.1})
```

**`AdaptiveStepsizeODESolver.integrate` calls `_before_integrate` before seeding
`solution[0]`.** That is where a jump landing on the very first time point is
folded in. No base-class solver modifies `y0` in `_before_integrate`, so nothing
without jumps observes the difference.

**`FixedGridFIRKODESolver.__init__` delegates to its base class** instead of
restating its constructor. The duplication was how the implicit solvers came to
be missing attributes the base class had gained.

Two internal removals: `_PerturbFunc` lost an unreachable `dN` argument, and
`next_after` is gone — no caller reached it. Neither was public API.

> Much of the diff in `misc.py`, `odeint.py`, `rk_common.py` and `solvers.py` is
> `black` reformatting. `git diff -w` is the smaller read.

## Decreasing `t` is refused

Jump ODEs can only be integrated with increasing `t`:

```
NotImplementedError: Jump ODEs cannot be integrated with decreasing `t`: the
jump map y(t+) = y(t-) + h(t, y(t-)) is not invertible in general.
```

Backwards, equation (2) is *implicit* in the unknown $y(t^-)$: you would have to
solve $y^- + h(t, y^-) = y^+$, which has no solution at all when the jump map is
not injective — take $h(t,z) = -z$, which maps every state to zero. So a
decreasing `t` raises rather than returning the plausible wrong answer it used
to.

Three things this is *not*:

- **It does not affect the adjoint.** `odeint_jump_adjoint` never integrates
  backwards *through* a jump; it runs the adjoint ODE backwards within each
  smooth segment and crosses each jump with a vector-Jacobian product, which
  needs the Jacobian of the jump map rather than its inverse. See
  [mathematics §7](mathematics.md#7-adjoint-sensitivities).
- **It does not affect jump-free solves.** Decreasing `t` works exactly as
  before when no jump options are present.
- **It does not affect `jump_t` on its own**, which means a discontinuity in
  $f$, not a state jump.

If you want the trajectory presented in reverse, integrate forwards and flip the
output.

## Migrating from `jump_t` + `events`

The older spelling still works and is exactly a `FixedJumpMechanism` in
uncoupled mode acting on the whole state:

```python
# before
odeint(func, y0, t, options={"jump": h, "jump_t": times, "events": mask})

# equivalent
mechanism = FixedJumpMechanism(times, mask, coupled=False)
odeint(func, y0, t, options={"jump": h, "jump_mechanism": mechanism})
```

There is a test asserting the two produce identical output. Move to the
mechanism form when you want any of:

| | |
| --- | --- |
| **Competing event types** | `coupled=True` and a jump map returning `(*batch, latent, n_types)`. |
| **A latent projection** | Keep the jump off augmented coordinates such as a cumulative hazard. |
| **Generated events** | The stochastic mechanisms; the legacy form is prescribed-only. |
| **Custom event logic** | Subclass `JumpMechanism`. |

Note that `jump_t` must be sorted. The legacy path walks it with a monotone
index, so an unsorted `jump_t` raises `events_t must be non-decreasing` at
construction rather than mis-timing the jumps.

## Known limitations

- **Marked events** — carrying a continuous mark — are not implemented;
  `kind == "marked"` raises `NotImplementedError`.
- **`scipy_solver`** ignores jump options and returns the jump-free trajectory
  with only a generic warning. See [solvers.md](solvers.md).
- **`∂t*/∂θ` is not propagated** for state-dependent mechanisms, matching the
  treatment `odeint_event` gives event times.
- **Batched event location is sequential**: one trajectory's event stops the
  step for the whole batch.
- **`CoupledStochasticJumpMechanism` supports at most one batch dimension.**

## Mechanisms are stateful

A mechanism is consumed by the solve: `FixedJumpMechanism.idx` advances past
realised events, and the stochastic mechanisms carry thresholds and event
counters that mutate as events fire. **Build a fresh mechanism per solve.**
Reusing one across calls silently continues from where the last solve left it —
usually exhausted, so the second solve has no events at all.

This is also how you read the results out afterwards:
`mechanism.all_realised_events` and, for the stochastic mechanisms,
`_realised_events` hold what actually fired.
