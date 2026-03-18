"""
LHSSampler: pre-generates n Latin Hypercube samples in the joint
(GenConfig × SimConfig) unit hypercube, then exposes them as lazy iterators.

Usage
-----
    from structure.sampling import LHSSampler

    sampler = LHSSampler(n=100_000, seed=42)

    # Iterate (GenConfig, SimConfig) pairs:
    for gen_cfg, sim_cfg in sampler:
        ...

    # Or access each config stream independently:
    for i, gen_cfg in enumerate(sampler.gen_configs(start_id=0)):
        ...
    for sim_cfg in sampler.sim_configs():
        ...

Notes
-----
- scipy.stats.qmc.LatinHypercube is used when scipy is available.
- Falls back to a per-dimension stratified shuffle (same marginal coverage,
  no inter-dimension correlation control) when scipy is absent.
- The full unit array is generated lazily on first access and cached.
  For n=100k and 69 dims this is ~55 MB (float64) — acceptable.
"""

from __future__ import annotations

from typing import Generator, Iterator, Tuple

import numpy as np

from structure.sampling._gen_map import N_GEN_DIMS, map_gen_config
from structure.sampling._sim_map import N_SIM_DIMS, map_sim_config
from scipy.stats.qmc import LatinHypercube as _LHS

_TOTAL_DIMS = N_GEN_DIMS + N_SIM_DIMS  # 46 + 23 = 69


class LHSSampler:
    """
    Pre-computes n Latin Hypercube samples covering the joint
    GenConfig + SimConfig parameter space.

    Parameters
    ----------
    n        : Number of samples to generate.
    seed     : Global RNG seed (controls LHS design and per-sample seeds).
    start_id : run_id / seed assigned to the first sample (increments by 1).
    """

    def __init__(self, n: int, seed: int = 42, start_id: int = 0) -> None:
        self.n = n
        self.seed = seed
        self.start_id = start_id
        self._unit: np.ndarray | None = None  # lazy, shape (n, _TOTAL_DIMS)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_unit(self) -> np.ndarray:
        """Return cached (n, _TOTAL_DIMS) unit-hypercube array."""
        if self._unit is None:
            self._unit = self._generate()
        return self._unit

    def _generate(self) -> np.ndarray:
        sampler = _LHS(d=_TOTAL_DIMS, seed=self.seed)
        return sampler.random(n=self.n)  # shape (n, _TOTAL_DIMS)

    # ------------------------------------------------------------------
    # Public iterators
    # ------------------------------------------------------------------

    def gen_configs(self, start_id: int | None = None) -> Generator:
        """Yield GenConfig instances, one per sample."""
        from structure.schema import GenConfig

        if start_id is None:
            start_id = self.start_id
        unit = self._get_unit()
        for idx in range(self.n):
            run_id = start_id + idx
            kwargs = map_gen_config(unit[idx, :N_GEN_DIMS], run_id=run_id, seed=run_id)
            yield GenConfig.model_validate(kwargs)

    def sim_configs(self) -> Generator:
        """Yield SimConfig instances, one per sample."""
        from structure.schema import SimConfig

        unit = self._get_unit()
        for idx in range(self.n):
            kwargs = map_sim_config(unit[idx, N_GEN_DIMS:])
            yield SimConfig.model_validate(kwargs)

    def __iter__(self) -> Iterator[Tuple]:
        """Yield (GenConfig, SimConfig) pairs."""
        from structure.schema import GenConfig, SimConfig

        unit = self._get_unit()
        start_id = self.start_id
        for idx in range(self.n):
            run_id = start_id + idx
            gen_kwargs = map_gen_config(
                unit[idx, :N_GEN_DIMS], run_id=run_id, seed=run_id
            )
            sim_kwargs = map_sim_config(unit[idx, N_GEN_DIMS:])
            yield (
                GenConfig.model_validate(gen_kwargs),
                SimConfig.model_validate(sim_kwargs),
            )

    def __len__(self) -> int:
        return self.n

    # ------------------------------------------------------------------
    # Convenience: materialise all at once
    # ------------------------------------------------------------------

    def all_gen_configs(self, start_id: int | None = None) -> list:
        return list(self.gen_configs(start_id=start_id))

    def all_sim_configs(self) -> list:
        return list(self.sim_configs())
