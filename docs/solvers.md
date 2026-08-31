# Choosing a solver

Every registered solver **except `scipy_solver`** integrates a jump ODE, and
each keeps its own convergence order across the jump. This page is about which
one to reach for.

## Accuracy

Measured on the reference problem $dz/dt = az$ with one multiplicative jump at
an off-grid time, whose solution $z(1) = z_0 e^{a}(1+b)$ is exact. Fixed-grid
and implicit solvers at `step_size=1e-3`, adaptive at `rtol=1e-10, atol=1e-12`.

| Solver | Family | Error |
| --- | --- | --- |
| `rk4` | fixed grid | 2.2 × 10⁻¹⁵ |
| `gl4`, `gl6` | implicit | 5.3 × 10⁻¹³, 9.2 × 10⁻¹³ |
| `radauIIA3`, `radauIIA5` | implicit | 4.8 × 10⁻¹², 4.3 × 10⁻¹² |
| `heun3` | fixed grid | 5.6 × 10⁻¹² |
| `adaptive_heun` | adaptive | 8.5 × 10⁻¹¹ |
| `dopri5` | adaptive | 1.0 × 10⁻¹⁰ |
| `tsit5` | adaptive | 1.3 × 10⁻⁹ |
| `fehlberg2` | adaptive | 1.5 × 10⁻⁹ |
| `fixed_adams`, `implicit_adams` | multistep | 3.4 × 10⁻⁸ |
| `explicit_adams` | multistep | 1.3 × 10⁻⁷ |
| `midpoint`, `heun2` | fixed grid | 4.5 × 10⁻⁸ |
| `implicit_midpoint`, `trapezoid` | implicit | 2.2 × 10⁻⁸ |
| `sdirk2`, `trbdf2` | implicit | 1.1 × 10⁻⁸ |
| `bosh3`, `dopri8` | adaptive | 6.2 × 10⁻⁷, 6.4 × 10⁻⁷ |
| `euler`, `implicit_euler` | first order | 2.7 × 10⁻⁴ |

The observed order across a jump matches the nominal order — `euler` 0.99,
`midpoint` 1.99, `rk4` 3.99 — because the step is split at the event rather than
the jump being applied at the boundary. See
[mathematics §3](mathematics.md#3-why-a-jump-must-land-on-a-step-boundary) for
what happens without that, and `tests/jump_tests.py` for the regression test
that pins it.

## Recommendations

**Default: `dopri5`.** Adaptive stepping suits jump problems: the solver takes
long steps between events and tightens around them, and prescribed event times
are put on `jump_t` so a step boundary lands on each exactly.

**Fixed grid when the event times are known and dense** — a clinical visit
schedule, a dosing regimen. `rk4` at a step matching the data is hard to beat,
and the grid merge makes each jump exact.

**Implicit when the smooth dynamics are stiff.** All nine implicit methods carry
jumps; the jump machinery is in the shared base class, so `radauIIA5` gets the
same treatment as `euler`.

**Avoid multistep (`explicit_adams`, `implicit_adams`, `fixed_adams`) on
event-heavy problems.** They extrapolate from derivatives evaluated at earlier
steps, and a jump invalidates that history. The solver calls a `_restart()` hook
at every event, which is correct but drops the method back to its startup order
locally — so a problem with many events spends most of its time restarting. On
the reference problem the restart is the difference between an error of
1.9 × 10⁻⁴ and 1.3 × 10⁻⁷ for `explicit_adams` — that is, without it a
fourth-order multistep method lands in the same range as `euler`
(2.7 × 10⁻⁴ at the same step).

**`scipy_solver` does not support jumps.** It delegates to SciPy, which knows
nothing about the mechanism. It emits a generic `Unexpected arguments` warning
and returns the **jump-free trajectory** — plausible numbers that are silently
wrong. Do not use it with jump options.

## Cost

Relative to the same solve without jumps:

- **A step containing no event** costs one extra evaluation of the crossing
  predicate at the step's right endpoint. For a prescribed mechanism it is a
  binary search on a sorted array; for a stochastic one it is one call to
  `cumulative_hazards_proj`. Negligible either way.
- **A step containing a state-dependent event** costs a bisection —
  $\lceil \log_2((b-a)/\delta) \rceil$ interpolant evaluations, roughly 20 at
  the default tolerance — plus one retaken step. The interpolant is already
  built, so these are cheap relative to a step.
- **A prescribed event costs nothing extra**: its time is on the grid, so no
  search and no retake.
- **Adaptive solvers** additionally re-evaluate `f` on the post-jump side of
  each discontinuity: one function evaluation per event.

The practical consequence is that a **prescribed** mechanism is markedly cheaper
than a stochastic one, which is convenient, because prescribed events are the
training path and stochastic ones are the sampling path.

### Batch behaviour

Event location reduces over the batch with an `any`, so one trajectory's event
stops the step for every trajectory in the batch. Results are correct — each
trajectory realises only its own events — but a batch whose events are widely
scattered in time takes many short steps. If that dominates, either reduce the
batch size or group trajectories with similar event timing.

## Interpolation

Fixed-grid solvers accept `options={"interp": "linear"}` (default) or
`"cubic"`. Both apply jumps correctly. Cubic costs one extra `f` evaluation per
step that emits an output, and gives a more accurate value at output times that
fall strictly inside a step; it does not change the accuracy at step boundaries,
which is where jumps land.
