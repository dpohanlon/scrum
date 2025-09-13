# London Underground Passenger Flow

This project models how passengers propagate through the London Underground using
[JAX](https://github.com/google/jax), [NumPyro](https://github.com/pyro-ppl/numpyro)
and [NetworkX](https://networkx.org/).  Station sequences for each line are
extracted from Transport for London data and used to compute expected passenger
numbers along a path.

## Approach
- Build directed graphs of stations for each line and extract all branches.
- For a chosen branch, propagate boarding and alighting counts with vectorised
  JAX helpers to obtain the fraction of passengers remaining after each stop.
- Use these fractions to sample passenger locations and compute the expected
  number of riders at every station.

## Outputs
Running the model yields:
- **Expected passengers per station** – the predicted number of riders on board
  when the train arrives at each stop.
- **Passenger location samples** – draws from a mixture distribution describing
  where passengers stand within a car.

## Status
Only the **Piccadilly line** is currently included.  Support for additional
lines will be added over time.

## Development
Install dependencies and run the tests with Poetry:

```bash
poetry install --no-root
poetry run pytest -q
```

