# Mathematical framework

How the solver integrates $dy = f\,dt + h\,dN$ without losing the order of the
underlying method, how event times are found when they depend on the state, and
why adjoint gradients exist through a jump map that cannot be inverted.

Every algorithm described here is the one in the code, not an idealisation of
it. Section references point at the implementation.

---

## 1. The model

The solver integrates a state $y(t) \in \mathbb{R}^n$ that follows an ODE
between isolated instants and is displaced discontinuously at those instants:

$$
dy(t) = f(t, y(t))\,dt + \sum_{k=1}^{K} h_k\!\left(t, y(t^-)\right) dN_k(t)
\tag{1}
$$

where $N_k$ counts the events of stream $k$. The solution is taken **càdlàg** —
right-continuous with left limits — so at an event time $t^*$,

$$
y(t^*) = y(t^{*+}) = y(t^{*-}) + \sum_k h_k\!\left(t^*, y(t^{*-})\right)\Delta N_k(t^*)
\tag{2}
$$

The value reported at an event time is the state *after* the jump. That
convention is carried consistently by every solver and by the adjoint; it is
what makes "the output at $t^*$" unambiguous.

### 1.1 The augmented state and the latent projection

A survival model does not want the jump applied to all of $y$. Its state is
augmented, $y = (z, \Lambda) \in \mathbb{R}^L \times \mathbb{R}^K$, where $z$ is
the latent trajectory and $\Lambda$ the running cumulative hazard, and only $z$
may jump — a cumulative hazard that jumped would no longer be the integral of
anything.

Write $P : \mathbb{R}^n \to \mathbb{R}^L$ for the coordinate projection
(`latent_proj`) and $P^\top$ for its zero-padded inclusion. With
$h(t,\cdot) : \mathbb{R}^L \to \mathbb{R}^{L \times K}$ and an indicator vector
$dN \in \{0,1\}^K$, the jump map is

$$
G(t, y) = y + P^\top h(t, Py)\,dN
\tag{3}
$$

In the **coupled** mode above, each stream contributes its own column of $h$ and
the contributions add. The **simple** mode is $K = 1$ with the trailing axis
dropped. A third, **marked** mode — events carrying a continuous mark — is not
implemented and raises.

> **Implementation.** $P^\top$ is never materialised. `scatter_jump` allocates
> zeros shaped like $y$, writes the jump through the *view* that `latent_proj`
> returns, and adds. That is why the projection must alias its input rather than
> copy it, and why the identity projection is legal: writing through it touches
> the whole state, which is what a non-augmented model wants.

---

## 2. Where the events come from

Everything above is agnostic to *when* events fire. That is the job of a *jump
mechanism*, and there are two kinds, corresponding to the two things one does
with such a model.

### 2.1 Prescribed events — training

Supervised fitting has the event times in the data. The mechanism is a sorted
list $\mathcal{E} = \{(t_i, m_i)\}$ of times and indicator masks. Because the
times are known before the solve begins, the solver puts them on its integration
grid, which makes the jumps *exact* rather than merely well-located — §4.

### 2.2 State-dependent events — sampling

Simulation must generate events from the model itself. Stream $k$ has intensity
$\lambda_k(t, z) \ge 0$ and compensator
$\Lambda_k(t) = \int_0^t \lambda_k\,ds$, carried as coordinates of $y$ so the
ODE solver integrates it alongside everything else.

By the time-rescaling theorem, a point process with compensator $\Lambda$
becomes a unit-rate Poisson process in the $\Lambda$-clock. Inverting that gives
the sampler: with $E^{(i)} \sim \mathrm{Exp}(1)$ i.i.d.,

$$
T_k^{(n)} = \inf\{\, t : \Lambda_k(t) \ge \varepsilon_k^{(n)} \,\},
\qquad
\varepsilon_k^{(n)} = \sum_{i \le n} E_k^{(i)}
\tag{4}
$$

So the mechanism holds one running threshold $\varepsilon \in \mathbb{R}^K$,
fires stream $k$ the moment $\Lambda_k$ reaches $\varepsilon_k$, and then
advances that threshold by a fresh $\mathrm{Exp}(1)$ draw. The additive update
is the memorylessness of the exponential distribution expressed as bookkeeping:
no history need be retained beyond the current threshold. A per-stream cap
$M_k$ is imposed by setting $\varepsilon_k \leftarrow \infty$ once $M_k$ events
have fired.

> **Why this matters for the solver.** $\Lambda$ is non-decreasing, because
> $\lambda \ge 0$ is enforced by construction (a Softplus head). That single
> fact is what makes event location cheap and sound — §5.

---

## 3. Why a jump must land on a step boundary

A Runge–Kutta method of order $p$ attains that order by matching a Taylor
expansion of the exact solution across the step. The expansion requires
$y \in C^{p+1}$ on the step. Across an event $y$ is not even continuous, so the
order conditions say nothing and the local error of a step straddling $t^*$ is
$O(h)$ — the size of the jump itself, undiminished by taking a smaller step.

Only the steps containing an event are polluted, and there are $|\mathcal{E}|$
of them, so the global error picks up a term $|\mathcal{E}| \cdot O(h)$ that no
refinement of the tableau touches. Whatever $p$ was, the method now converges
like a first-order one.

Adding the jump at the step boundary misplaces it in time by up to one step, and
that misplacement dominates everything the tableau does:

| Method | Order | error at $h = 0.005$, jump at the boundary | error at $h = 0.005$, step split at the event | observed order, split |
| --- | --- | --- | --- | --- |
| `euler` | 1 | $1.74 \times 10^{-3}$ | $1.33 \times 10^{-3}$ | 0.99 |
| `midpoint` | 2 | $4.04 \times 10^{-4}$ | $1.11 \times 10^{-6}$ | 1.99 |
| `rk4` | 4 | $4.03 \times 10^{-4}$ | $3.46 \times 10^{-13}$ | 3.99 |

Measured on $dz/dt = az$ with one multiplicative jump at an off-grid time,
solution $z(T) = z_0 e^{aT}(1+b)$ exact; orders are ratios of successive errors
over $h = 0.02, 0.01, 0.005$.

Note the second and third rows: under the boundary treatment, `midpoint` and
`rk4` produce *the same error*. They are no longer solving with their tableaus —
they are reporting where the event was mistakenly placed. Their convergence
ratios in that regime are erratic (0.01, 2.02 for midpoint; −0.00, 2.02 for
RK4), because the misplacement is a sawtooth in $h$ depending on where $t^*$
falls between grid points: never better than first order, and not cleanly first
order either.

**The conclusion drives the whole design: never step across a discontinuity.**

---

## 4. Fixed-grid solvers

Let $\Pi$ be the grid the solver would use anyway. Prescribed event times are
merged into it:

$$
\Pi' = \mathrm{dedup} \circ \mathrm{sort}\;\bigl(\Pi \cup \{\,t_i \in \mathcal{E} : t_0 < t_i < t_N\,\}\bigr)
\tag{5}
$$

When there is nothing to merge, $\Pi'$ is $\Pi$ unchanged — the same object, not
a rebuilt copy. This is what keeps a jump-free solve bit-identical to the
original code path, and differentiable with respect to `t`: de-duplication uses
a comparison mask rather than `torch.unique`, which has no derivative.

Within each grid interval the solver sub-steps, splitting at whatever events it
finds:

```
for each grid interval [τ_j, τ_j+1]:
    a, y := τ_j, y(τ_j)
    while a < τ_j+1:
        b := τ_j+1
        y_b, φ := step(a → b, y)          # φ is the dense output on [a,b]
        t* := next_event_time(φ, a, b)    # in the half-open (a, b]

        if t* exists and t* < b:
            b := t*
            y_b, φ := step(a → b, y)      # retake, ending on the event

        emit φ(t) for each output time t ∈ (a, b)

        if t* exists:
            y_b := G(b, y_b)              # the jump, eq. (3)
            restart()                     # multistep history is now invalid

        emit y_b if b is an output time   # post-jump: càdlàg
        a, y := b, y_b
```

Two details carry most of the correctness.

**The scan interval is half-open.** A sub-step consumes the events in
$(a, b]$ — open on the left because an event at $a$ was already consumed by the
previous sub-step, closed on the right so an event landing exactly on $b$ is
consumed by this one. Every event therefore belongs to exactly one sub-step. Get
this wrong at either end and events on grid points fall through the crack, or
are applied twice.

**Output times are emitted per sub-step, not per grid interval.** Times strictly
inside come from that sub-step's dense output; the endpoint is emitted
separately, after the jump, so it carries the post-jump value. A later event
cannot leak into an earlier output because it belongs to a later sub-step that
has not run yet.

### 4.1 Multistep solvers

`explicit_adams` and `implicit_adams` extrapolate from derivatives evaluated at
earlier steps. After a jump those derivatives lie on the other side of the
discontinuity, and continuing to use them costs most of the solver's accuracy.
`FixedGridODESolver` therefore calls a `_restart()` hook whenever it applies a
jump: a no-op for single-step methods, and for Adams it clears the stored
`prev_f`/`prev_t` so the method rebuilds its order from the RK4 startup steps.

On the reference problem at step $10^{-3}$, that is the difference between an
error of $1.9\times10^{-4}$ and $1.3\times10^{-7}$ for `explicit_adams`, and
$9.0\times10^{-6}$ versus $3.4\times10^{-8}$ for `implicit_adams`.

---

## 5. Locating a state-dependent event

For prescribed times this is a binary search on a sorted array. The interesting
case is §2.2, where the event time is defined implicitly by the state.

Define the crossing predicate on the step's dense output $\varphi$:

$$
D(t) = \bigl[\;\exists k : \Lambda_k(\varphi(t)) \ge \varepsilon_k \;\bigr]
\tag{6}
$$

Because each $\Lambda_k$ is non-decreasing and each $\varepsilon_k$ is fixed
within the step, $D$ is **monotone**: once true, it stays true. Two consequences,
and the algorithm is nothing more than them:

- **Existence is one evaluation.** There is an event in $(a, b]$ if and only if
  $D(b)$ holds. A step with no event costs a single test — not a search.
- **Location is bisection on a monotone predicate.** Maintain
  $\neg D(\mathrm{lo}) \wedge D(\mathrm{hi})$; the invariant is preserved by each
  halving and $\mathrm{hi}$ converges to the crossing from the right in
  $\lceil \log_2((b-a)/\delta) \rceil$ evaluations for tolerance $\delta$.

Converging *from the right* is deliberate: the returned time always satisfies the
predicate, so the event it reports is real rather than imminent.

> **The subtlety worth knowing.** Once $t^*$ is located, the step is retaken to
> end there, and the state recomputed by that shorter step is *not*
> bit-identical to $\varphi(t^*)$. It can land a hair on the wrong side of the
> threshold, and a naive re-test would then report no event at the very time the
> solver stopped for one.
>
> So the crossing mask is **memoised at location time** and is authoritative
> when the jump is applied. Deciding once, and only once, which streams fire
> removes the inconsistency instead of papering over it with a tolerance.

A custom mechanism whose predicate is *not* monotone must override
`next_event_time` with a search that suits it; the base contract only requires
that it return the earliest event in $(t_0, t_1]$, or `None`.

---

## 6. Adaptive solvers

An embedded Runge–Kutta method chooses its own steps, so it cannot be handed a
grid. It is steered instead.

**Prescribed times** are appended to `jump_t`, the existing mechanism by which
the solver truncates a step at a known discontinuity of $f$. A step boundary
then lands on each event time exactly, and the jump is applied there.

**State-dependent events** cannot be known before the step is taken, so the order
is reversed: take the step, fit its dense output, locate the crossing in it (§5),
and if $t^*$ falls strictly inside, *reject the step* and force the next attempt
to have $dt = t^* - t_0$. The retry ends exactly on the event, at which point it
is handled like any boundary event. Rejection reuses the machinery the error
controller already has, rather than introducing a second notion of a partial
step.

### 6.1 Carrying the jump without corrupting the interpolant

An adaptive solver answers queries by interpolation, and its interpolant is
fitted to $(y_0, y_1)$ at the step's endpoints. If the jump were folded into
$y_1$ immediately, the interpolant would be fitted to a value the smooth
dynamics never produced, and every query inside the step would be wrong.

So the increment is carried *beside* the state, in a `dJ` field of
`_RungeKuttaState`. It keeps $y_1$ pre-jump and is consumed in two places:

- a query at exactly $t_1$ returns $\varphi(t_1) + \Delta J$ — the càdlàg value —
  while queries inside the step use $\varphi$ alone;
- the next step begins from $y_1 + \Delta J$ and **re-evaluates $f$ there**,
  because $f(t, \cdot)$ at the post-jump state is a different value: the
  derivative is discontinuous across the event even though $f$ itself is smooth.

Reading the query does not clear $\Delta J$ — a query is a read, and the same
instant may be asked for by the caller and then stepped away from. It is cleared
by the step that consumes it.

### 6.2 The horizon

An adaptive step may overshoot the last requested time and interpolate back.
Event location is therefore capped at $t_N$: without the cap the solver would
realise an event past the horizon, consume a threshold, and mutate the sampler's
state for an interval the caller never asked about. With it, a solve over
$[0, T]$ and a solve over $[0, T+1]$ agree on $[0, T]$.

---

## 7. Adjoint sensitivities

The adjoint method computes gradients by integrating a second ODE backwards in
time. A jump ODE appears hostile to exactly that. The resolution is that the
backward pass never needs to *invert* the jump — only to *differentiate* it.

### 7.1 The two transitions

Let $L$ be a scalar loss and let the adjoint
$a(t) = \partial L / \partial y(t)$ be a row covector. On a smooth segment it
obeys the familiar equation, integrated backwards:

$$
\frac{da}{dt} = -\,a(t)\,\frac{\partial f}{\partial y}(t, y(t))
\tag{7}
$$

At an event, $y^+ = G(t^*, y^-)$, so the chain rule gives the transition

$$
a^- = a^+ \frac{\partial G}{\partial y}
    = a^+ \left( I + P^\top \frac{\partial (h\,dN)}{\partial z} P \right)
\tag{8}
$$

with the jump network's parameters picking up
$\partial L / \partial \theta_h \mathrel{+}= a^+ P^\top \partial (h\,dN)/\partial \theta$.

> **The crux.** Equation (8) contains $\partial G/\partial y$. It does **not**
> contain $G^{-1}$. A Jacobian-transpose-vector product always exists and costs
> one backward evaluation; an inverse may not exist at all. *Differentiating a
> non-invertible map is routine; inverting it is impossible.*
>
> Take $h(t, z) = c - z$, so $G \equiv c$ collapses every state to a constant.
> No reverse-time integration through it can exist. Yet
> $\partial G/\partial y = 0$, so $a^- = 0$ — and that is the right answer: the
> output does not depend on the pre-jump state, because the jump discarded it.
> Measured against the analytic solution, $\partial L/\partial y_0 = 0$,
> $\partial L/\partial a$ and $\partial L/\partial c$ all agree to ten digits.

### 7.2 Obtaining it by composition

Neither (7) nor (8) is coded by hand. Let $\Phi_k$ be the flow over the $k$-th
inter-event segment and $G_k$ the jump at its right endpoint. The whole solve is
a composition,

$$
y(t_N) = \Phi_{M+1} \circ G_M \circ \Phi_M \circ \cdots \circ G_1 \circ \Phi_1 (y_0)
\tag{9}
$$

in which each $\Phi_k$ is an `odeint_adjoint` call — itself a differentiable
primitive whose backward *is* equation (7) — and each $G_k$ is an ordinary
tensor expression. Reverse-mode automatic differentiation of (9) therefore
produces exactly the alternation of (7) and (8), without either being written
down.

The memory profile survives: each segment retains $O(1)$ state under the
adjoint, and the only thing held across a jump is one state per event.

Note also that the backward pass runs in reversed time *only within* a smooth
segment, where the flow is a diffeomorphism and reverse integration is the
ordinary, well-posed adjoint. Each jump is crossed by a vector-Jacobian product
instead.

### 7.3 Why the events are recorded first

The composition (9) presupposes that the segment boundaries are known. They are
obtained by a preliminary `no_grad` forward solve that records
$(t^*_i, dN_i)$ for every event; the differentiable replay then applies precisely
those. Recording, rather than re-deriving, is load-bearing for two reasons:

- a stochastic mechanism would draw fresh $\mathrm{Exp}(1)$ thresholds on a
  second pass and simulate a *different* trajectory, so forward and backward
  would disagree about which function was being differentiated;
- re-locating a state-dependent event from a marginally different state could
  place it on the other side of a threshold — the same inconsistency as in §5,
  now between passes rather than within one.

The replay is deterministic given the record, so the function being
differentiated is the function that was evaluated.

### 7.4 What is not differentiated

For a state-dependent mechanism, $t^*$ is itself a function of the parameters —
it is where $\Lambda(t) - \varepsilon$ crosses zero — so a complete gradient
carries a term through $\partial t^*/\partial\theta$. That term is dropped,
matching the treatment `odeint_event` gives event times. For prescribed events
the times are data and there is nothing to differentiate, so gradients on the
training path are complete.

---

## 8. Invariants

These hold by construction and each is covered by a test in
`tests/jump_tests.py`.

| Invariant | Enforced by |
| --- | --- |
| Every event is applied exactly once | Half-open $(a, b]$ scan; the mechanism's pointer advances past a realised event |
| Reported values are càdlàg | Endpoint emitted after the jump; adaptive queries add $\Delta J$ at $t_1$ only |
| Jumps stay inside $\mathrm{range}(P^\top)$ | The increment is written through the projection view; augmented coordinates untouched |
| Order $p$ is preserved | No step ever spans a discontinuity |
| A jump-free solve is unchanged | The grid is only rebuilt when there is something to merge |
| Forward and adjoint differentiate the same function | Recorded event schedule replayed verbatim |

---

## 9. Numerical evidence

The reference problem is $dz/dt = az$ with multiplicative jumps
$z \to z(1+b)$, whose solution $z(T) = z_0 e^{aT}(1+b)^m$ is available in closed
form together with its derivatives — so the tests check the size and the timing
of the jump, not merely that one occurred.

| | $z(1)$ | $\partial/\partial z_0$ | $\partial/\partial\theta_f$ | $\partial/\partial\theta_h$ |
| --- | --- | --- | --- | --- |
| analytic | 2.7863389475 | 2.7863389475 | 2.7863389476 | 4.2866753036 |
| direct | 2.7863389476 | 2.7863389476 | 2.7863389485 | 4.2866753040 |
| adjoint | 2.7863389476 | 2.7863389475 | 2.7863389476 | 4.2866753040 |

`dopri5`, two events, float64.

The sampler of §2.2 is checked distributionally rather than by smoke test. Under
a constant intensity the realised events must form a Poisson process, which pins
down both how many events occur and where they fall: with $\lambda = 1.5$ over
$[0, 6]$ and 400 trajectories the counts give mean 9.105 and variance 8.475
against 9 for both; and conditional on $N(T) = n$ the event times are
distributed as $n$ i.i.d. $\mathrm{Uniform}(0, T)$ order statistics, which they
satisfy with a Kolmogorov–Smirnov statistic of 0.0195 against a 1% critical
value of 0.0334.

Conditional uniformity is used deliberately in place of the more obvious test on
inter-arrival gaps: the final gap of every trajectory is right-censored by the
horizon, so discarding it biases the sample short and would fail a correct
sampler.

---

## 10. Limits

- **Marked events** — events carrying a continuous mark — are not implemented
  and raise `NotImplementedError`. The solver-side contract they would need is
  in place; what is missing is the mark's distribution and its entry into $h$.
- **Decreasing `t`** is refused when jumps are present. Equation (2) is explicit
  forwards and implicit backwards, and $G$ need not be injective, so there is no
  general inverse to integrate through. Reversing a *recorded* solve is
  well-posed and would reuse the machinery of §7.3.
- **$\partial t^*/\partial\theta$** is not propagated for state-dependent
  mechanisms (§7.4).
- **Event location reduces over the batch.** The crossing predicate ends in
  `.any()`, so one trajectory's event stops the step for all of them. Correct,
  and wasteful when batch trajectories have widely separated events.
