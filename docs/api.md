# API reference

Everything public lives in `torchdiffeq.jump`:

```python
from torchdiffeq.jump import (
    JumpMechanism,                    # base class for custom mechanisms
    FixedJumpMechanism,               # events at prescribed times
    SimpleStochasticJumpMechanism,    # one hazard-driven stream
    CoupledStochasticJumpMechanism,   # competing event types
    odeint_jump_adjoint,              # adjoint-mode gradients
)
```

---

## Solver options

The jump map and the mechanism are passed through `options`, so every solver
keeps its existing signature.

```python
odeint(func, y0, t, method="dopri5",
       options={"jump": h, "jump_mechanism": mechanism})
```

### `jump`

The jump map $h(t, z)$. Called with the event time and the **latent projection**
of the pre-jump state, and must return

| mechanism `kind` | return shape |
| --- | --- |
| `"simple"` | same shape as its input, `(*batch, latent)` |
| `"coupled"` | `(*batch, latent, n_event_types)` — one column per stream |

It is an ordinary callable; an `nn.Module` if it has parameters you want
trained.

### `jump_mechanism`

A `JumpMechanism` deciding when events fire and which streams they belong to.

`jump` and `jump_mechanism` must be passed **together**. Passing one without the
other raises rather than silently doing nothing —
see [troubleshooting](troubleshooting.md#jump-and-jump_mechanism-must-be-supplied-together-in-options).

### `jump_t` and `events` (legacy)

`options={"jump": h, "jump_t": times, "events": mask}` is still accepted and is
exactly `FixedJumpMechanism(times, mask, coupled=False)` acting on the whole
state. `events` has shape `(batch, len(jump_t))`.

Passing **`jump_t` alone** keeps its upstream meaning — a discontinuity in $f$
that the grid should land on — and produces no state jump. That behaviour is
unchanged.

---

## `FixedJumpMechanism`

```python
FixedJumpMechanism(
    events_t,          # (n_events,) non-decreasing event times
    events_mask,       # (*batch, n_events) or (*batch, n_events, n_types)
    coupled=False,
    latent_proj=None,
)
```

Events at times known before the solve. They are merged into the integration
grid, so every jump lands exactly on a step boundary and the solver keeps its
nominal order.

`events_mask` says which streams fire at each time; it may be bool, float, or
sparse. Its time axis must match `events_t` in length, and `events_t` must be
non-decreasing — both are validated at construction.

**Attributes**

| | |
| --- | --- |
| `event_times` | The prescribed times. Present so the solver can put them on its grid. |
| `all_realised_events` | The mask rows consumed so far, sliced along the time axis. |
| `idx` | How many event rows have been consumed. |

A mechanism is **stateful**: `idx` advances as events are realised, so a
mechanism is good for one solve. Build a fresh one per call (or per batch).

---

## `SimpleStochasticJumpMechanism`

```python
SimpleStochasticJumpMechanism(
    batch_shape,                 # () or (batch,)
    cumulative_hazards_proj=None,  # (t, y) -> Lambda
    rng=None,
    latent_proj=None,
)
```

A single event stream per batch element, fired when a cumulative hazard carried
in the state crosses an $\mathrm{Exp}(1)$ threshold. `cumulative_hazards_proj`
says which coordinates of the state hold $\Lambda$; it takes `(t, y)` and
returns the hazard block.

Your `func` is responsible for integrating $\Lambda$ — that is, for returning
$\lambda \ge 0$ in those coordinates. A hazard that can go negative breaks the
monotonicity the event search relies on
(see [mathematics §5](mathematics.md#5-locating-a-state-dependent-event)).

---

## `CoupledStochasticJumpMechanism`

```python
CoupledStochasticJumpMechanism(
    batch_shape,
    coupling_dim=0,              # number of competing event types, K
    cumulative_hazards_proj=None,
    rng=None,
    latent_proj=None,
    max_events=torch.inf,
)
```

As above, with `coupling_dim` competing event types, each with its own hazard
and its own column of `h`.

`max_events` caps how many events **each type** may fire — a scalar for all
types, or one value per type. With three types and `max_events=2`, a trajectory
may fire up to six events in total.

**Attributes**

| | |
| --- | --- |
| `all_realised_events` | Nested tensor of realised `(time, type)` pairs per batch element. |
| `coupled` | `coupling_dim > 0`. |

Only zero- or one-dimensional `batch_shape` is supported.

---

## `JumpMechanism` — writing your own

Subclass it when your events are neither prescribed nor hazard-driven: a
contact, a threshold crossing, a control switch.

**You must implement**

```python
@property
def kind(self) -> str:            # "simple" or "coupled"

def realise_event(self, t, z):    # -> indicator mask, consumed at time t
```

**You will almost always implement**

```python
def next_event_time(self, z_interp, t0, t1, tol=1e-6):
    """Earliest event in the half-open (t0, t1], or None."""
```

`z_interp` is the current step's dense output — call it at any time in
`[t0, t1]` to get the state there. The interval is **half-open on the left**: an
event at `t0` was already consumed by the previous step, so returning it again
applies it twice.

**You may override**

| | |
| --- | --- |
| `event_times` | Return known times to have them merged into the grid; default `None` (state-dependent). |
| `has_event_at(t)` | Whether an unrealised event sits exactly at `t`. Only consulted for the first time point, for càdlàg at `t[0]`. Default `False`. |
| `latent_proj(z, enforce=False)` | Restrict the jump to part of the state. Default identity. |

**Provided for you**

`jump_increment`, `increment_from_mask`, `apply_jump` and `scatter_jump` turn a
mask into a state increment using the shape rules of `kind` and the latent
projection. The solver calls them; you rarely need to.

A minimal example is in [examples.md](examples.md#6-a-custom-mechanism-bouncing-ball).

### `latent_proj`

Restricts the jump to part of the state — coordinates outside it, such as the
cumulative hazards of a survival model, are left untouched.

It must return a **view** of its input, e.g. `lambda z: z[..., :k]`, because the
increment is written back through it. A projection that copies (`z.clone()[...]`)
raises. The identity is allowed and means "jump the whole state".

---

## `odeint_jump_adjoint`

```python
odeint_jump_adjoint(
    func, y0, t, *,
    rtol=1e-7, atol=1e-9, method=None, options=None,
    adjoint_rtol=None, adjoint_atol=None, adjoint_method=None,
    adjoint_options=None, adjoint_params=None,
)
```

Adjoint-mode gradients for a jump ODE. Same arguments as `odeint_adjoint`, with
the jump map and mechanism arriving through `options` exactly as they do for
`odeint`.

**Use this rather than `odeint_adjoint`.** The plain adjoint is not jump-aware:
its backward sweep integrates straight through the discontinuity and returns
gradients for the jump-free problem, with none reaching the jump network. It
does not raise — it returns plausible, wrong numbers.

Gradients flow to `y0`, to the parameters of `func`, and to the parameters of
the jump network. With no jump arguments this is exactly `odeint_adjoint`.

Not differentiated: the event *times* of a state-dependent mechanism. See
[mathematics §7.4](mathematics.md#74-what-is-not-differentiated).

`adjoint_params` need only cover `func`; the jump network's parameters are
reached by ordinary autograd and are picked up automatically.

---

## Constants

| | |
| --- | --- |
| `MAX_EVENTS_PER_STEP` | 1024. A mechanism reporting more events than this within one step raises rather than looping — it means the mechanism is not making progress. |
| `JumpMechanismKind` | `Literal["coupled", "simple", "marked"]`. `"marked"` is reserved and raises. |
