import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive


@jax.jit
def compute_alphas(n_on, n_off):
    """Compute the fraction of passengers remaining on the train at each stop.

    ``jax.lax.scan`` carries the passenger load forward, yielding the same
    result as the original Python loop without the interpretive overhead.
    Division by zero is avoided by using ``jnp.divide`` with the ``where``
    argument.
    """

    dtype = jnp.result_type(n_on, n_off, 0.0)
    n_on = n_on.astype(dtype)
    n_off = n_off.astype(dtype)

    def step(load, on_off):
        on, off = on_off
        load_after = load - jnp.minimum(off, load)
        alpha = jax.lax.cond(
            load > 0,
            lambda _: load_after / load,
            lambda _: jnp.array(1.0, dtype=dtype),
            operand=None,
        )
        next_load = load_after + on
        return next_load, alpha

    init_load = jnp.asarray(0, dtype=dtype)
    _, alphas = jax.lax.scan(step, init_load, (n_on, n_off))
    return alphas


@jax.jit
def precompute_survivors(n_on, alpha):
    """Precompute the number of surviving passengers for all station pairs.

    The previous implementation filled the matrix using nested Python loops,
    leading to quadratic Python overhead.  Here we exploit broadcasting to
    construct the same matrix in a single expression.
    """

    dtype = n_on.dtype
    alpha_cum = jnp.cumprod(jnp.concatenate([jnp.ones(1, dtype=dtype), alpha]))
    left = jnp.where(alpha_cum[1:] > 0, n_on / alpha_cum[1:], 0.0)
    right = alpha_cum[1:]
    return jnp.triu(left[:, None] * right[None, :])


def passenger_location_model(n_on, n_off, positions_on):
    """
    We'll assume we've precomputed alpha and the survivors array,
    then define mixture distributions per station using the precomputed logic.
    """
    num_rounds = len(n_on)

    mu = [
        numpyro.sample(
            f"mu_{i}",
            dist.TruncatedNormal(loc=positions_on[i], scale=30.0, low=0.0, high=100.0),
        )
        for i in range(num_rounds)
    ]
    sigma = [
        numpyro.sample(f"sigma_{i}", dist.Exponential(3.0)) for i in range(num_rounds)
    ]

    alpha = compute_alphas(n_on, n_off)

    numpyro.deterministic("alpha", alpha)

    survivors_2d = precompute_survivors(n_on, alpha)

    for k in range(num_rounds):

        total_k = jnp.sum(survivors_2d[: k + 1, k])
        mixture_weights_k = jnp.where(
            total_k > 0, survivors_2d[: k + 1, k] / total_k, jnp.zeros(k + 1)
        )

        final_mix_k = dist.MixtureSameFamily(
            dist.Categorical(probs=mixture_weights_k),
            dist.TruncatedNormal(
                loc=jnp.array(mu[: k + 1]),
                scale=jnp.array(sigma[: k + 1]),
                low=0.0,
                high=100.0,
            ),
        )
        numpyro.sample(f"passenger_location_{k}", final_mix_k)
        numpyro.deterministic(f"mixture_weights_{k}", mixture_weights_k)


def sample_passenger_locations_per_station(
    n_on, n_off, positions_on, path, num_samples=1000, seed=0
):
    rng_key = jax.random.PRNGKey(seed)
    num_rounds = len(n_on)

    return_sites = (
        [f"passenger_location_{i}" for i in range(num_rounds)]
        + [f"mixture_weights_{i}" for i in range(num_rounds)]
        + ["alpha"]
    )

    predictive = Predictive(
        passenger_location_model, num_samples=num_samples, return_sites=return_sites
    )
    samples = predictive(rng_key, n_on, n_off, positions_on)
    passenger_locations_per_station = {}
    mixture_weights_per_station = {}
    for i in range(num_rounds):
        passenger_locations_per_station[path[i + 1]] = samples[
            f"passenger_location_{i}"
        ]
        mixture_weights_per_station[path[i + 1]] = samples[f"mixture_weights_{i}"]

    alpha_samples = samples["alpha"]

    return passenger_locations_per_station, mixture_weights_per_station, alpha_samples


@jax.jit
def expected_passengers_per_station(n_on, n_off):
    """Return expected passenger counts at each station along a path.

    Parameters
    ----------
    n_on, n_off : array-like
        Numbers of passengers boarding and alighting at each stop after the
        initial station.

    Returns
    -------
    jnp.ndarray
        The expected number of passengers remaining on the train upon arrival
        at each station in the path.
    """

    alpha = compute_alphas(n_on, n_off)
    survivors = precompute_survivors(n_on, alpha)
    return survivors.sum(axis=0)


def average_expected_passengers(n_on_list, n_off_list):
    """Average passenger expectations from multiple incoming paths.

    Each element of ``n_on_list``/``n_off_list`` corresponds to an incoming
    branch that terminates at a shared station.  The function computes the
    expected number of passengers arriving via each branch and returns their
    mean.
    """

    expectations = [
        expected_passengers_per_station(n_on, n_off)[-1]
        for n_on, n_off in zip(n_on_list, n_off_list)
    ]
    return jnp.mean(jnp.stack(expectations))
