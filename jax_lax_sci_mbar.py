"""Small SCI solver for MBAR using jax.lax.while_loop.

Notes:
* Nearly identical functionality is also available in PyMBAR:
    https://github.com/choderalab/pymbar/blob/9ddeab46ce173c9c9edd1d01b2f55a6f73a7f3db/pymbar/mbar_solvers.py#L193-L244
    where it is used to initialize more advanced solvers.
* See also FastMBAR https://github.com/BrooksResearchGroup-UM/FastMBAR ,
    which implements other advanced solvers with PyTorch acceleration.
"""

import numpy as np
import jax
from jax import jit, numpy as jnp
from jax.scipy.special import logsumexp
from tqdm import tqdm  # TODO: remove progress bars


def reweight_from_mixture(u_kn, trial_f_k, N_k):
    """eq. C3 of [Shirts, Chodera, 2008] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2671659/
    + intuition from [Shirts, 2017] https://arxiv.org/abs/1704.00891"""

    log_weights_n = logsumexp(trial_f_k - u_kn.T, b=N_k, axis=1)
    implied_f_k = -logsumexp(-u_kn - log_weights_n, axis=1) - jnp.log(jnp.sum(N_k))
    return implied_f_k


@jit
def solve_mbar(u_kn, N_k, initial_f_k=None, max_iterations=10_000, atol=0.0):
    """self-consistent iteration, C.1 [Shirts, Chodera, 2008] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2671659/"""

    t = 0  # iteration counter
    f_k = jnp.zeros(len(u_kn)) if initial_f_k is None else jnp.array(initial_f_k, dtype=float)
    diff = jnp.inf  # max(abs(f_k - f_{k-1}))
    state = (t, f_k, diff)

    def update(state):
        t, f_k, diff = state

        t_next = t + 1

        f_k_next = reweight_from_mixture(u_kn, f_k, N_k)
        f_k_next = f_k_next - f_k_next[0]

        diff = jnp.max(jnp.abs((f_k_next - f_k)))

        return (t_next, f_k_next, diff)

    def cond_fun(state):
        t, _, diff = state

        not_timed_out = t < max_iterations
        not_converged = diff > atol
        return not_timed_out * not_converged

    num_iters_completed, f_k, _ = jax.lax.while_loop(cond_fun, update, state)
    aux = dict(num_iters_completed=num_iters_completed)

    return f_k, aux


def bootstrap_mbar(u_kn, N_k, num_bootstrap_samples=100, bootstrap_max_iters=1_000, bootstrap_atol=1e-6):
    """call SCI solver num_bootstrap_samples times with possibly looser settings"""
    initial_f_k = jnp.zeros(len(u_kn))
    f_k, aux = solve_mbar(u_kn, N_k, initial_f_k)
    bootstrap_samples = []
    rng = np.random.default_rng(2023)

    for _ in tqdm(range(num_bootstrap_samples)):
        _u_kn = rng.choice(u_kn, (u_kn.shape[1],), replace=True, axis=1)
        f_k_bootstrap, b_aux = solve_mbar(_u_kn, N_k, f_k, bootstrap_max_iters, bootstrap_atol)
        bootstrap_samples.append(f_k_bootstrap)

    return f_k, jnp.array(bootstrap_samples)
