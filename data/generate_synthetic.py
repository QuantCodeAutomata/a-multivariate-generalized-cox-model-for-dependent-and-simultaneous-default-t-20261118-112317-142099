"""Random-number-generator utility shared by every experiment.

The paper (Sections 4 and 5) gives explicit mathematical constructions for
generalized Cox default times driven by Levy subordinators, compound Poisson
processes, shot-noise processes and frailty factors; it requires no external
or market dataset. Every experiment is a closed-form computation checked
against a Monte Carlo simulation of the same model, so the only "data" this
project needs is reproducible pseudo-random draws.

Sample paths of each driver (Levy subordinator jump times and marks, Poisson
shock times, CIR intensity paths, frailty draws, ...) are produced by the
model classes in ``src/`` (``levy_subordinator.py``, ``factor_model.py``,
``shot_noise.py``, ``cir_block.py``, ``liu_model.py``): that simulation logic
implements the paper's stochastic representation of each driver and is part
of the method under test, not a data-preparation step, so it is not
duplicated here. This module only fixes the seed used across every
experiment so that results are reproducible from ``src/config.RNG_SEED``.
"""
from __future__ import annotations

import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy default random generator, seeded for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Overrides ``src.config.RNG_SEED`` when provided.
    """
    if seed is None:
        from src.config import RNG_SEED

        seed = RNG_SEED
    assert isinstance(seed, int) and seed >= 0, "seed must be a nonnegative int"
    return np.random.default_rng(seed)
