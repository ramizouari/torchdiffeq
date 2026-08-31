# Troubleshooting

Every error the jump feature raises, what causes it, and what to do. Failures
that are *silent* are listed last — they are the ones worth knowing about
before you hit them.

---

## Errors

### `jump` and `jump_mechanism` must be supplied together in `options`

You passed one without the other. The mechanism says *when* events fire, the
jump map says *what they do*; neither is useful alone.

Related messages, all from the same check:

| Message | Cause |
| --- | --- |
| `` `jump` was given without `jump_mechanism` or `jump_t`+`events` `` | Nothing tells the solver when a jump occurs. |
| `` `jump_mechanism` was given without `jump` `` | No jump network to evaluate when an event fires. |
| `` `events` requires both `jump` and `jump_t` `` | The legacy spelling needs all three. |
| `` `events` and `jump_mechanism` are mutually exclusive `` | The mechanism already decides which events fire; pick one API. |
| `` `jump_mechanism` must be a JumpMechanism, got X `` | Pass an instance, not a class or a callable. |

### `events_t must be non-decreasing` / `events_t must be a one-dimensional tensor`

`FixedJumpMechanism` walks its event list with a monotone index, so the times
must be sorted. Sort `events_t` **and** the matching axis of `events_mask`
together:

```python
order = torch.argsort(events_t)
mechanism = FixedJumpMechanism(events_t[order], events_mask[..., order])
```

### `events_mask must carry one entry per event time: expected N, got M`

The time axis of `events_mask` disagrees with `len(events_t)`. The mask is
`(*batch, n_events)` when `coupled=False` and `(*batch, n_events, n_types)` when
`coupled=True` — a mismatch here usually means `coupled` is set the wrong way.

### `The latent projection must be a view of its input (or the identity)`

The increment is written back *through* the projection, so it must alias its
input:

```python
latent_proj=lambda z: z[..., :k]          # view — fine
latent_proj=lambda z: z.clone()[..., :k]  # copy — raises
latent_proj=lambda z: z[..., idx]         # advanced indexing copies — raises
```

Basic slicing gives a view; fancy/boolean indexing does not. If your latent
coordinates are not contiguous, reorder the state so they are.

### `The size of tensor a (L) must match the size of tensor b (K)`

The jump map returned the wrong shape for the mechanism's mode. In `coupled`
mode `h` must return `(*batch, latent, n_event_types)`; in `simple` mode it
returns the shape of its input. `mechanism.kind` tells you which mode you are
in, and it is `"coupled"` exactly when `coupled=True` (fixed) or
`coupling_dim > 0` (stochastic).

```python
# coupled: one column per event stream
def forward(self, t, z):
    return self.layers(z).reshape(*z.shape, n_event_types)
```

This raises in every case, including `latent == n_event_types` — there is no
shape coincidence that lets a mis-shaped jump map through quietly.

### `Jump ODEs cannot be integrated with decreasing t`

Expected: the jump map is not invertible in general. Integrate forwards and
reverse the output. If it came from `odeint_adjoint`, that is the next entry.

### `NotImplementedError` from `odeint_adjoint` with jump options

`odeint_adjoint`'s backward pass integrates with a decreasing `t` and inherits
`options` as `adjoint_options`, so the jump arguments reach the guard above.
Use `odeint_jump_adjoint`, which takes the same arguments:

```python
- from torchdiffeq import odeint_adjoint
+ from torchdiffeq import odeint_jump_adjoint
```

This is a *good* failure. Before the guard, that call returned gradients for the
jump-free problem with none reaching the jump network, and raised nothing.

### `Marked jump mechanisms are not implemented yet`

`kind == "marked"` is reserved for events carrying a continuous mark. Use
`"coupled"` with one stream per discrete event type.

### `More than 1024 events realised in the single step [a, b]`

A mechanism is not making progress: it keeps reporting an event at effectively
the same instant. Usual causes:

- `next_event_time` returns a time `<= t0`. It must return an event in the
  **half-open** `(t0, t1]`; returning `t0` re-fires the event just applied.
- `realise_event` does not consume the event, so the same one is found again.
  A prescribed mechanism must advance its index; a hazard-driven one must
  advance its threshold.
- A genuine Zeno point — contacts crowding to an accumulation time. That is a
  modelling issue: add a minimum inter-event time or a resting condition.

### `No event detected during realisation`

`realise_event` was called at a time where the mechanism's own predicate does
not hold. In custom code this usually means `next_event_time` and
`realise_event` disagree about the condition. The built-in stochastic mechanisms
avoid it by memoising the crossing mask at location time —
[mathematics §5](mathematics.md#5-locating-a-state-dependent-event).

---

## Silent failures

### `scipy_solver` ignores jumps entirely

It delegates to SciPy, which knows nothing about the mechanism. You get a
generic `ScipyWrapperODESolver: Unexpected arguments {...}` warning and the
**jump-free trajectory** — numbers that look reasonable and are wrong. Use any
other solver.

### `odeint_adjoint` instead of `odeint_jump_adjoint`

Guarded now (above), but worth stating: the plain adjoint is not jump-aware.
If you ever see a jump network whose parameters sit at exactly zero gradient,
this is the first thing to check.

### A reused mechanism has no events left

Mechanisms are stateful and consumed by a solve. Reusing one silently continues
from where the last solve ended — usually exhausted:

```python
mechanism = FixedJumpMechanism(times, mask)
a = odeint(f, y0, t, options={"jump": h, "jump_mechanism": mechanism})
b = odeint(f, y0, t, options={"jump": h, "jump_mechanism": mechanism})  # no jumps
```

Build a fresh one per call. In a training loop, per batch.

### Hazards drift when they should not

If a stochastic mechanism's cumulative hazard is not monotone — because `f`
returns a negative rate in those coordinates — the event search silently
mislocates events. The monotonicity of $\Lambda$ is what makes the single test
at the step's right endpoint a sound existence check. Enforce $\lambda \ge 0$,
e.g. with a Softplus head.

---

## Diagnostics

**Did the jump fire at all?** Run the same solve with the event mask zeroed and
compare. Identical output means it did not.

```python
on  = odeint(f, y0, t, options={"jump": h, "jump_mechanism": make_mechanism(mask)})
off = odeint(f, y0, t, options={"jump": h, "jump_mechanism": make_mechanism(mask * 0)})
print((on - off).abs().max())     # 0.0 means no jump happened
```

**What fired, and when?** `FixedJumpMechanism.idx` counts consumed events;
`all_realised_events` returns the consumed mask rows. The stochastic mechanisms
keep `_realised_events` as `(time, type)` pairs per batch element, and
`_event_counter` as counts per stream.

**Is the accuracy the solver's or the jump's?** Halve the step and look at how
the error scales. It should follow the method's order — if it falls to first
order, an event is landing somewhere other than a step boundary.
