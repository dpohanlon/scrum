import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive

import jax


def compute_alphas(n_on, n_off):
    """
    α_k = fraction that REMAINS aboard at station k AFTER alighting (before boarding at k).
    We then roll the load forward by boarding at k. Snapshot used by the model:
    'after boarding at k' via survivors (see below).
    """
    alphas = []
    load = 0.0

    for on, off in zip(n_on, n_off):
        L_before = load
        off_eff = min(off, L_before)
        L_after_alight = L_before - off_eff

        alpha_k = 1.0 if L_before == 0 else L_after_alight / L_before
        alphas.append(alpha_k)

        load = L_after_alight + on

    return jnp.array(alphas)


@jax.jit
def precompute_survivors(n_on, alpha):
    """
    Given n_on[r] and alpha[i], compute survivors[r,k] = n_on[r] * product_{j=r+1..k} alpha[j]
    for 0 <= r <= k < num_rounds.
    """
    num_rounds = len(n_on)

    survivors = jnp.zeros((num_rounds, num_rounds))

    alpha_cum = jnp.cumprod(jnp.concatenate([jnp.ones([1]), alpha]))

    def product_of_alphas(r, k):
        return alpha_cum[k + 1] / alpha_cum[r + 1]

    # Fill in survivors
    for r in range(num_rounds):
        for k in range(num_rounds):
            if k >= r:
                s = n_on[r] * product_of_alphas(r, k)
            else:
                s = 0.0
            survivors = survivors.at[r, k].set(s)
    return survivors


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

    return_sites = []
    for i in range(num_rounds):
        return_sites.append(f"passenger_location_{i}")
        return_sites.append(f"mixture_weights_{i}")
    return_sites.append("alpha")

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
