import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive

import jax


# def compute_alphas(n_on, n_off):
#     """
#     Compute alpha[i] = fraction that remain at station i.
#     This requires knowing the total population 'just before' station i's alighting.
#     One easy approach: do a forward pass to find station populations, then alpha_i = 1 - n_off[i]/population_before_i.
#     """
#     num_rounds = len(n_on)
#     population_before = jnp.zeros(
#         num_rounds + 1
#     )  # station 0..num_rounds; station i is before i-th alighting

#     # forward pass to get population before station i
#     # station 0: population_before[0] = 0 (start empty)
#     pop = 0.0
#     new_pop = []
#     for i in range(num_rounds):
#         pop += n_on[i]
#         population_before = population_before.at[i + 1].set(pop)
#         pop -= n_off[i]  # after station i, for next iteration

#     # now alpha_i = fraction that remain
#     alpha = []
#     for i in range(num_rounds):
#         p_i = population_before[i + 1]  # pop before alighting at station i
#         a_i = 1.0 - (n_off[i] / (p_i + 1e-10))
#         a_i = jnp.clip(a_i, 0.0, 1.0)
#         alpha.append(a_i)
#     alpha = jnp.array(alpha)
#     return alpha

def compute_alphas(n_on, n_off):
    """
    Compute alpha[i] = fraction that remain at station i.
    This version is robust to data inconsistencies by ensuring the
    passenger count never drops below zero.
    """
    num_rounds = len(n_on)
    population_before = jnp.zeros(num_rounds + 1)
    pop = 0.0
    for i in range(num_rounds):
        # Add passengers who boarded at station i
        pop += n_on[i]

        # Store the population before alighting at the next station (i+1).
        # Crucially, ensure this value cannot be negative due to prior data errors.
        population_before = population_before.at[i + 1].set(jnp.maximum(0.0, pop))

        # For the next iteration, update the population by subtracting alighters
        pop -= n_off[i]

    # Calculate alpha values using the corrected population counts
    alpha = []
    for i in range(num_rounds):
        # This p_i is now guaranteed to be non-negative
        p_i = population_before[i + 1]

        # Calculate survival fraction. Add a small epsilon to prevent division by zero.
        a_i = 1.0 - (n_off[i] / (p_i + 1e-10))

        # Clip the result to handle edge cases (e.g., if n_off > p_i due to data noise)
        a_i = jnp.clip(a_i, 0.0, 1.0)
        alpha.append(a_i + 1E-6)

    alpha = jnp.array(alpha)
    return alpha


def precompute_survivors(n_on, alpha):
    """
    Given n_on[r] and alpha[i], compute survivors[r,k] = n_on[r] * product_{j=r+1..k} alpha[j]
    for 0 <= r <= k < num_rounds.
    """
    num_rounds = len(n_on)
    # We'll build a 2D array survivors[r,k].
    # If r > k, survivors[r,k] = 0.
    # If r <= k, survivors[r,k] = n_on[r] * product(alpha[r+1..k]).

    # We'll do a naive approach: double loop, or we can do partial products. Let's do partial products for clarity.
    # partial_products[i] = alpha[0]*alpha[1]*...alpha[i-1].
    # We want alpha[r+1]*...*alpha[k].
    # That's partial_products[k+1]/partial_products[r+1], if r < k.

    survivors = jnp.zeros((num_rounds, num_rounds))

    # build a cumulative product array for alpha
    # e.g. alpha_cum[0] = 1, alpha_cum[1] = alpha[0], alpha_cum[2] = alpha[0]*alpha[1], ...
    alpha_cum = jnp.cumprod(jnp.concatenate([jnp.ones([1]), alpha]))
    # alpha_cum[i] = product_{j=0..i-1} alpha[j], with alpha_cum[0] = 1

    def product_of_alphas(r, k):
        # product_{j=r+1..k} alpha[j] = alpha_cum[k+1] / alpha_cum[r+1]
        return alpha_cum[k + 1] / alpha_cum[r + 1]

    # Fill in survivors
    for r in range(num_rounds):
        for k in range(num_rounds):
            if (
                k >= r
            ):  # round r boards at station r, so r must be <= k to exist at station k
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

    # 1) sample mu_i, sigma_i
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

    # 2) compute alpha[i] in python (cannot do it inside the model if it depends on n_off, etc.)
    #    so we do it as "deterministic" from outside. We'll replicate it as a 'deterministic' node for illustration.
    alpha = compute_alphas(n_on, n_off)

    numpyro.deterministic("alpha", alpha)

    # 3) build the survivors array outside the model, or do so in a 'deterministic' node, but let's just do it here
    #    for demonstration.
    survivors_2d = precompute_survivors(n_on, alpha)
    # shape [num_rounds, num_rounds], survivors_2d[r,k] = how many from round r survive to station k

    # 4) For station k, mixture weights = survivors_2d[:,k] / sum(survivors_2d[:,k]) for r=0..k
    #    We define mixture over the first k+1 components.
    for k in range(num_rounds):
        # sum of all survivors at station k
        total_k = jnp.sum(survivors_2d[: k + 1, k])
        mixture_weights_k = jnp.where(
            total_k > 0, survivors_2d[: k + 1, k]/ total_k,
            jnp.zeros(k + 1)
        )

        # define mixture distribution
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

    print('alphas', compute_alphas(n_on, n_off))

    return_sites = []
    for i in range(num_rounds):
        return_sites.append(f"passenger_location_{i}")
        return_sites.append(f"mixture_weights_{i}")
    return_sites.append("alpha")  # might as well see alpha

    predictive = Predictive(
        passenger_location_model, num_samples=num_samples, return_sites=return_sites
    )
    samples = predictive(rng_key, n_on, n_off, positions_on)

    # separate out passenger locations and mixture weights
    passenger_locations_per_station = {}
    mixture_weights_per_station = {}
    for i in range(num_rounds):
        passenger_locations_per_station[path[i + 1]] = samples[
            f"passenger_location_{i}"
        ]
        mixture_weights_per_station[path[i + 1]] = samples[f"mixture_weights_{i}"]

    # alpha as well if you want
    alpha_samples = samples["alpha"]

    return passenger_locations_per_station, mixture_weights_per_station, alpha_samples
