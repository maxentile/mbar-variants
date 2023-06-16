"""WARNING: slow/untested brainstorming-phase implementations -- please use pymbar or another implementation!"""

import numpy as np
from scipy.special import logsumexp
from functools import partial


def reweight_from_mixture(u_kn, trial_f_k, N_k):
    """eq. C3 of [Shirts, Chodera, 2008] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2671659/
    + intuition from [Shirts, 2017] https://arxiv.org/abs/1704.00891"""
    log_weights_n = logsumexp(trial_f_k - u_kn.T, b=N_k, axis=1)
    implied_f_k = -logsumexp(-u_kn - log_weights_n, axis=1) - np.log(np.sum(N_k))
    return implied_f_k


def solve_mbar(u_kn, N_k):
    """self-consistent iteration, C.1 [Shirts, Chodera, 2008] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2671659/"""
    f_k = np.zeros(len(u_kn))

    for _ in range(1000):  # skip convergence checks, for simplicity
        f_k = reweight_from_mixture(u_kn, f_k, N_k)

    return f_k - f_k[0]


def reweight_from_local_mixtures(trial_f_k, states, samples, neighborhoods):
    """variant: only reweight from user-specified neighborhoods of each state, rather than from all states.
    (reweight_from_mixture, where neighborhoods[k] = set(range(K)) for all k.)

    WARNING: Experimental, untested, likely incorrect.

    TODO[correctness]: think harder about whether this is reasonable
        (Study [Elvira+, 2015] Generalized multiple importance sampling https://arxiv.org/abs/1511.03095
        ... esp. ~ section 7.2 on using subsets of proposals... I may be missing a term in the denominator.)

    TODO[efficiency]: refactor to accept sparse u_kln computed once rather than a list of thermo states
        (extremely slow as written, esp. for large neighborhoods!)
    """

    K = len(states)

    num_samples = np.array([len(s) for s in samples])
    mixture_weights = num_samples / np.sum(num_samples)

    f_k_prime = np.zeros(K)

    for k in range(K):
        idxs = np.array(list(neighborhoods[k]))

        # compute u_j(samples[i]) for i,j in neighborhood of k
        x_n = np.concatenate([samples[i] for i in idxs])
        u_kn = np.array([states[i].reduced_potential(x_n) for i in idxs])
        w_k = mixture_weights[k]

        # interpret these neighboring states as a weighted mixture
        log_weights_n = logsumexp(trial_f_k[idxs] - u_kn.T, b=w_k, axis=1)

        # compute Z_k / Z_mix
        # Z_k / Z_mix = (Z_k / Z_neighborhood_k) * (Z_neighborhood_k / Z_mix)
        assert k in idxs, "neighborhoods[k] should also contain k"
        log_target_n = -u_kn[idxs == k][0]
        delta_f_k_vs_neighborhood = -logsumexp(log_target_n - log_weights_n)
        delta_f_neighborhood_vs_mix = -np.log(np.sum(w_k))

        f_k_prime[k] = delta_f_k_vs_neighborhood + delta_f_neighborhood_vs_mix

    return f_k_prime


def solve_neighborhood_mbar(states, samples, neighborhoods):
    """variant: apply self-consistent iteration to reweight_from_local_mixtures"""

    f_k = np.zeros(len(states))
    update = partial(reweight_from_local_mixtures, states=states, samples=samples, neighborhoods=neighborhoods)

    for _ in range(1000):  # skip convergence checks, for simplicity
        f_k = update(f_k)

    return f_k - f_k[0]
