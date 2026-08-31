# Further documentation

## Solver options

Adaptive and fixed solvers all support several options. Also shown are their default values.

**Adaptive solvers (dopri8, dopri5, bosh3, adaptive_heun):**<br>
For these solvers, `rtol` and `atol` correspond to the tolerances for accepting/rejecting an adaptive step.

- `first_step=None`: What size the first step of the solver should be; by default this is selected empirically.

- `safety=0.9, ifactor=10.0, dfactor=0.2`: How the next optimal step size is calculated, see E. Hairer, S. P. Norsett G. Wanner, *Solving Ordinary Differential Equations I: Nonstiff Problems*, Sec. II.4. Roughly speaking, `safety` will try to shrink the step size slightly by this amount, `ifactor` is the most that the step size can grow by, and `dfactor` is the most that it can shrink by.

- `max_num_steps=2 ** 31 - 1`: The maximum number of steps the solver is allowed to take.

- `dtype=torch.float64`: what dtype to use for timelike quantities. Setting this to `torch.float32` will improve speed but may produce underflow errors more easily.

- `step_t=None`: Times that a step must me made to. In particular this is useful when `func` has kinks (derivative discontinuities) at these times, as the solver then does not need to (slowly) discover these for itself. If passed this should be a `torch.Tensor`.

- `jump_t=None`: Times that a step must be made to, and `func` re-evaluated at. In particular this is useful when `func` has discontinuites at these times, as then the solver knows that the final function evaluation of the previous step is not equal to the first function evaluation of this step. (i.e. the FSAL property does not hold at this point.) If passed this should be a `torch.Tensor`. Note that this may not be efficient when using PyTorch 1.6.0 or earlier.

- `norm`: What norm to compute the accept/reject criterion with respect to. Given tensor input, this defaults to an RMS norm. Given tupled input, this defaults to computing an RMS norm over each tensor, and then taking a max over the tuple, producing a mixed L-infinity/RMS norm. If passed this should be a function consuming a tensor/tuple with the same shape as `y0`, and return a scalar corresponding to its norm. When passed as part of `adjoint_options`, then the special value `"seminorm"` may be used to zero out the contribution from the parameters, as per the ["Hey, that's not an ODE"](https://arxiv.org/abs/2009.09457) paper.

- `jump=None`,<br>`jump_mechanism=None`: Jump support; see [Jump options](#jump-options) below.

**Fixed solvers (euler, midpoint, rk4, explicit_adams, implicit_adams):**<br>

- `step_size=None`: How large each discrete step should be. If not passed then this defaults to stepping between the values of `t`. Note that if using `t` just to specify the start and end of the regions of integration, then it is very important to specify this argument! It is mutually exclusive with the `grid_constructor` argument, below.

- `grid_constructor=None`: A more fine-grained way of setting the steps, by setting these particular locations as the locations of the steps. Should be a callable `func, y0, t -> grid`, transforming the arguments `func, y0, t` of `odeint` into the desired grid (which should be a one dimensional tensor).

- `perturb`: Defaults to False. If True, then automatically add small perturbations to the start and end of each step, so that stepping to discontinuities works. Note that this this may not be efficient when using PyTorch 1.6.0 or earlier.

Individual solvers also offer certain options.

**explicit_adams:**<br>
For this solver, `rtol` and `atol` are ignored. This solver also supports:

- `max_order`: The maximum order of the Adams-Bashforth predictor.

**implicit_adams:**<br>
For this solver, `rtol` and `atol` correspond to the tolerance for convergence of the Adams-Moulton corrector. This solver also supports:

- `max_order`: The maximum order of the Adams-Bashforth-Moulton predictor-corrector.

- `max_iters`: The maximum number of iterations to run the Adams-Moulton corrector for.

**scipy_solver:**<br>
- `solver`: which SciPy solver to use; corresponds to the `'method'` argument of `scipy.integrate.solve_ivp`.

## Jump options

> Longer-form documentation — the mathematical framework, worked examples,
> solver selection and troubleshooting — is in [docs/](docs/).

Every solver except `scipy_solver` can integrate a *jump ODE*, whose solution is
piecewise continuous:

```
dy/dt = f(t, y)          between events
y(t+) = y(t-) + h(t, y(t-))    at each event time t
```

The solution is treated as càdlàg (right-continuous): the value reported at an
event time is the *post*-jump state. Two options control this, and they must be
passed together:

- `jump=None`: The jump map `h(t, z)`, called with the event time and the latent
  projection of the pre-jump state. In uncoupled mode it returns a tensor shaped
  like its input; in coupled mode it returns `(*batch, latent, n_event_types)`,
  one jump per event type.

- `jump_mechanism=None`: A `JumpMechanism`, which decides *when* events happen
  and *which* event streams fire. Three are provided:

  - `FixedJumpMechanism(events_t, events_mask, coupled=False, latent_proj=None)`:
    events at prescribed times. Because the times are known ahead of the solve
    they are merged into the integration grid, so every jump lands exactly on a
    step boundary and the solver keeps its nominal order.

  - `SimpleStochasticJumpMechanism(batch_shape, cumulative_hazards_proj=None, rng=None, latent_proj=None)`:
    a single event stream per batch element, fired when a cumulative hazard
    carried in the state crosses an exponential threshold. Use
    `cumulative_hazards_proj(t, z)` to say which coordinates of the state hold
    the cumulative hazard.

  - `CoupledStochasticJumpMechanism(batch_shape, coupling_dim, cumulative_hazards_proj=None, rng=None, latent_proj=None, max_events=inf)`:
    as above with `coupling_dim` competing event types, each with its own hazard
    and its own column of `h`. `max_events` caps the number of events per stream.

  Custom mechanisms subclass `JumpMechanism` and implement `next_event_time` and
  `realise_event`. State-dependent event times are located from the solver's
  dense output and the step is retaken to end exactly on the event, so the
  interpolant never straddles a discontinuity.

`latent_proj` restricts the jump to part of the state: coordinates outside it -
the cumulative hazards of a survival model, for instance - are left untouched.
It must be a *view* of its input (e.g. `lambda z: z[..., :k]`), so that the jump
can be written back into the state.

The older `jump_t` + `events` spelling is still accepted and is equivalent to a
`FixedJumpMechanism` in uncoupled mode acting on the whole state. Passing
`jump_t` on its own keeps its upstream meaning - a discontinuity in `f` that the
grid should land on - and produces no state jump.

The multistep solvers (`explicit_adams`, `implicit_adams`) restart at every
event: their stored derivatives lie on the other side of the discontinuity, so
the predictor cannot extrapolate through them. They therefore fall back to their
startup order locally, and a single-step solver is the better choice on a problem
with many events.

Jump ODEs can only be integrated with increasing `t`: the jump map is not
invertible in general, so a decreasing `t` raises rather than returning a
plausible wrong answer.

### Gradients through jumps

`odeint` is differentiable through jumps by backpropagation as usual.
`odeint_adjoint` is **not** jump-aware: its backward sweep integrates straight
through the discontinuity and returns gradients for the jump-free problem. Use
`odeint_jump_adjoint` instead, which takes the same arguments:

```python
from torchdiffeq import odeint_jump_adjoint

y = odeint_jump_adjoint(func, y0, t, method="dopri5",
                        options={"jump": h, "jump_mechanism": mechanism})
```

It discovers the event times in a `torch.no_grad()` forward pass, then replays
them differentiably: each smooth segment between events goes through
`odeint_adjoint` and keeps its O(1) memory profile, and the jump map itself is
differentiated by plain autograd, linking consecutive segments into one graph.
Recording the events rather than re-deciding them matters -- a stochastic
mechanism would otherwise draw fresh thresholds and simulate a different
trajectory. With no jump arguments this is exactly `odeint_adjoint`.

Gradients flow to `y0`, to the parameters of `func` and to the parameters of the
jump network. Not differentiated: the event *times* of a state-dependent
mechanism, which get the same treatment `odeint_event` gives event times. For a
`FixedJumpMechanism` the times are data, so there is nothing to drop.

 ## Adjoint options

 The function `odeint_adjoint` offers some adjoint-specific options.

 - `adjoint_rtol`,<br>`adjoint_atol`,<br>`adjoint_method`,<br>`adjoint_options`:<br>The `rtol, atol, method, options` to use for the backward pass. Defaults to the values used for the forward pass.

 - `adjoint_options` has the special key-value pair `{"norm": "seminorm"}` that provides a potentially more efficient adjoint solve when using adaptive step solvers, as described in the ["Hey, that's not an ODE"](https://arxiv.org/abs/2009.09457) paper.

 - `adjoint_params`: The parameters to compute gradients with respect to in the backward pass. Should be a tuple of tensors. Defaults to `tuple(func.parameters())`.
   - If passed then `func` does not have to be a `torch.nn.Module`.
   - If `func` has no parameters, `adjoint_params=()` must be specified.


 ## Callbacks

 Callbacks can be triggered during the solve. Callbacks should be specified as methods of the `func` argument to `odeint` and `odeint_adjoint`.

 At the moment support for this is minimal: let us know if you'd find additional callbacks useful.

 **callback_step(self, t0, y0, dt):**<br>
 This is called immediately before taking a step of size `dt`, at time `t0`, with current solution value `y0`. This is supported by every solver except `scipy_solver`.

 **callback_accept_step(self, t0, y0, dt):**<br>
 This is called when accepting a step of size `dt` at time `t0`, with current solution value `y0`. This is supported by the adaptive solvers (dopri8, dopri5, bosh3, adaptive_heun).

 **callback_reject_step(self, t0, y0, dt):**<br>
 As `callback_accept_step`, except called when rejecting steps.

 In addition, callbacks can be triggered during the adjoint pass by adding `_adjoint` to the name of any one of the supported callbacks, e.g. `callback_step_adjoint`.