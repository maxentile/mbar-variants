import numpy as np
from jax_lax_sci_mbar import bootstrap_mbar

# adapted from https://github.com/proteneer/timemachine/pull/1098/files#diff-cb526fade08eed0853a590a554d7ccdb928d264f18cd2a7c7a451c8662ed2084R50
def make_partial_overlap_uniform_ukn_example(dlogZ, n_samples=100):
    """Generate 2-state u_kln matrix for uniform distributions with partial overlap"""

    def u_a(x):
        """Unif[0.0, 1.0], with log(Z) = 0"""
        in_bounds = (x > 0) * (x < 1)
        return np.where(in_bounds, 0, +np.inf)

    def u_b(x):
        """Unif[0.5, 1.5], with log(Z) = dlogZ"""
        x_ = x - 0.5
        return u_a(x_) + dlogZ

    rng = np.random.default_rng(2023)

    x_a = rng.uniform(0, 1, (n_samples,))
    x_b = rng.uniform(0.5, 1.5, (n_samples,))

    assert np.isfinite(u_a(x_a)).all()
    assert np.isfinite(u_b(x_b)).all()

    x = np.concatenate([x_a, x_b])

    u_kn = np.stack([u_a(x), u_b(x)])
    N_k = np.array([n_samples, n_samples])

    return u_kn, N_k

if __name__ == "__main__":
    u_kn, N_k = make_partial_overlap_uniform_ukn_example(dlogZ=123.45, n_samples=2_000)
    f_k, bootstrap_samples = bootstrap_mbar(u_kn, N_k)
    print(f_k, bootstrap_samples.mean(0), bootstrap_samples.std(0))
