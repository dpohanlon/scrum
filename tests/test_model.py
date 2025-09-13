import pathlib
import sys
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
jax = pytest.importorskip('jax')
jnp = jax.numpy
numpyro = pytest.importorskip('numpyro')

from model import (
    compute_alphas,
    precompute_survivors,
    sample_passenger_locations_per_station,
    average_expected_passengers,
)


def test_compute_alphas():
    n_on = jnp.array([10, 5, 2])
    n_off = jnp.array([3, 8, 1])
    expected = jnp.array([1.0, 0.2, 6 / 7])
    assert jnp.allclose(compute_alphas(n_on, n_off), expected)


def test_precompute_survivors():
    n_on = jnp.array([10, 5, 2])
    n_off = jnp.array([3, 8, 1])
    alpha = compute_alphas(n_on, n_off)
    survivors = precompute_survivors(n_on, alpha)
    expected = jnp.array(
        [
            [10.0, 2.0, 1.7142857],
            [0.0, 5.0, 4.2857143],
            [0.0, 0.0, 2.0],
        ]
    )
    assert jnp.allclose(survivors, expected)


def test_sample_passenger_locations_per_station_shapes():
    n_on = jnp.array([10, 5])
    n_off = jnp.array([3, 8])
    positions_on = jnp.array([10.0, 20.0])
    path = ["A", "B", "C"]
    passengers, weights, alphas = sample_passenger_locations_per_station(
        n_on, n_off, positions_on, path, num_samples=5, seed=0
    )

    assert set(passengers.keys()) == {"B", "C"}
    assert passengers["B"].shape == (5,)
    assert weights["C"].shape == (5, 2)
    assert jnp.allclose(weights["C"].sum(axis=1), 1.0)
    assert alphas.shape == (5, 2)


def test_average_expected_passengers_merging_branches():
    # Two branches converge on a final station. Branch 1 boards 10 and loses 4,
    # branch 2 boards 20 and loses 8.  Expectations at the merge should average.
    n_on_branch1 = jnp.array([10, 0])
    n_off_branch1 = jnp.array([0, 4])
    n_on_branch2 = jnp.array([20, 0])
    n_off_branch2 = jnp.array([0, 8])

    expected = average_expected_passengers(
        [n_on_branch1, n_on_branch2], [n_off_branch1, n_off_branch2]
    )

    assert jnp.isclose(expected, 9.0)
