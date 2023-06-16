from wip_mbar_variants import solve_mbar, solve_neighborhood_mbar

from dataclasses import dataclass
import numpy as np


@dataclass
class Gaussian:
    mean: float
    sigma: float
    log_Z_offset: float

    def sample(self, n_samples=1000):
        return np.random.randn(n_samples) * self.sigma + self.mean

    def reduced_potential(self, x):
        unnormed_logpdf = -0.5 * (x - self.mean) ** 2 / self.sigma ** 2
        return -unnormed_logpdf - self.log_Z_offset


def make_test_system(rng, K=50, difficulty=1.0):
    """evenly spaced means, independently varying stddev and normalization
    if difficulty = 0, all share same mean and stddev
    """
    means = np.arange(K) * difficulty
    stddevs = 1 + (rng.uniform(0, 1, K) * difficulty)
    log_Z_offsets = rng.standard_normal(K) * 10

    states = []
    for k in range(K):
        states.append(Gaussian(means[k], stddevs[k], log_Z_offsets[k]))

    true_f_k = np.array([-s.log_Z_offset for s in states])
    true_f_k -= true_f_k[0]

    return states, true_f_k


def test_mbar():
    rng = np.random.default_rng(666)
    for K in [10, 20, 50]:
        for difficulty in [0.0, 1.0]:
            print(f"testing K={K}, difficulty={difficulty}")
            states, true_f_k = make_test_system(rng, K, difficulty)
            assert np.std(true_f_k) > 1

            N_k = [100] * K
            samples = [state.sample(n) for (state, n) in zip(states, N_k)]
            x_n = np.concatenate(samples)
            u_kn = np.array([state.reduced_potential(x_n) for state in states])

            est_f_k = solve_mbar(u_kn, N_k)
            residuals = est_f_k - true_f_k
            assert np.std(residuals) < np.std(true_f_k), "didn't improve over predicting zeros"

            if difficulty == 0.0:
                assert np.std(residuals) < 1e-4, "didn't converge in 0-difficulty case"


def get_neighborhoods_by_idx_window(K, window_size=1):
    neighborhoods = []
    for k in range(K):
        lb = max(0, k - window_size)
        ub = min(K, k + window_size + 1)
        neighbors_k = set(range(lb, ub))

        neighborhoods.append(neighbors_k)
    return neighborhoods


def test_neighborhood_mbar():
    # note: exorbitantly slow for large window_size!

    rng = np.random.default_rng(666)

    for K in [10, 20, 50]:
        for difficulty in [0.0, 1.0]:
            for window_size in [5, K]:
                print(f"testing K={K}, difficulty={difficulty}, window_size={window_size}")
                states, true_f_k = make_test_system(rng, K, difficulty)
                assert np.std(true_f_k) > 1

                N_k = [100] * K
                samples = [state.sample(n) for (state, n) in zip(states, N_k)]
                neighborhoods = get_neighborhoods_by_idx_window(K, window_size=window_size)

                est_f_k = solve_neighborhood_mbar(states, samples, neighborhoods)
                residuals = est_f_k - true_f_k
                assert np.std(residuals) < np.std(true_f_k), "didn't improve over predicting zeros"

                if difficulty == 0.0:
                    assert np.std(residuals) < 1e-4, "didn't converge in 0-difficulty case"

                if window_size == K:
                    # compare to standard MBAR
                    x_n = np.concatenate(samples)
                    u_kn = np.array([state.reduced_potential(x_n) for state in states])
                    standard_est_f_k = solve_mbar(u_kn, N_k)
                    assert np.std(est_f_k - standard_est_f_k) < 1e-4, "didn't agree with standard MBAR in limit"
