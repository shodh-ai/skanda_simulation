"""
Pipeline orchestrator — Steps 0 through 8.

Single entry point:

    vol = run(sim)

Takes a ResolvedGeneration, chains all generation steps, handles
PercolationFailed retries, and returns a MicrostructureVolume.

Step dependency map:
    Step 0  compute_composition(sim)           → CompositionState
    Step 1  build_domain(comp, sim)            → DomainGeometry
        ↓ (retry loop starts here — seed-dependent steps)
    Step 2  pack_carbon_scaffold(comp, domain, sim, rng)  → PackingResult
    Step 2b rasterize_carbon(packing, domain)             → RasterResult
    Step 3  map_si_vf(comp, domain, sim, carbon_label, packing, rng) → SiMapResult
    Step 4  fill_cbd_binder(comp, domain, sim, carbon_label, si_result, rng) → CBDBinderResult
    Step 5  calender → new si/cbd
    Step 5b rasterize_carbon(packing, domain)   → re-rasterize in final coords
    Step 6  build_sei(comp, domain, sim, carbon_label, si_result, rng) → SEIResult
    Step 7  validate_percolation(...)            → PercolationResult
                ↑ raises PercolationFailed → increment seed, retry from Step 2
    Step 8  assemble_volume(...)                 → MicrostructureVolume

Retry behaviour:
    On PercolationFailed, seed is incremented by 1 and Steps 2–7 are
    re-executed. Steps 0–1 are deterministic and never re-run.
    Maximum retries controlled by max_retries (default: 10).
"""

from __future__ import annotations

import time
from typing import Optional
import numpy as np

from structure.schema import ResolvedGeneration
from structure.data import (
    CompositionState,
    DomainGeometry,
    PackingResult,
    RasterResult,
    SiMapResult,
    CBDBinderResult,
    SEIResult,
    PercolationFailed,
    PercolationResult,
    MicrostructureVolume,
    PipelineResult,
)
from structure.generation import (
    compute_composition,
    build_domain,
    pack_carbon_scaffold,
    map_si_distribution,
    fill_cbd_binder,
    apply_calendering,
    add_sei_shell,
    validate_percolation,
    assemble_volume,
)
from structure.utils import rasterize_carbon


# ---------------------------------------------------------------------------
# PipelineResult — wraps MicrostructureVolume with run metadata
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_generation(
    sim: ResolvedGeneration,
    max_retries: int = 10,
    verbose: bool = True,
) -> PipelineResult:
    """
    Run the full microstructure generation pipeline.

    Args:
        sim         : ResolvedGeneration — all config + materials resolved.
        max_retries : Maximum number of packing+percolation attempts.
                      Each attempt uses seed = sim.seed + attempt_index.
        verbose     : Print per-step progress and summaries if True.

    Returns:
        PipelineResult containing the MicrostructureVolume and run metadata.

    Raises:
        RuntimeError if all max_retries attempts fail percolation.
    """
    t_run_start = time.perf_counter()
    step_times: dict[str, float] = {}
    pipeline_warns: list[str] = []

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    def _timed(name: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        step_times[name] = time.perf_counter() - t0
        _log(f"  [{name}] {step_times[name]:.3f}s")
        return result

    # ── Steps 0–1: deterministic, run once ───────────────────────────────
    _log("\n── Step 0: Composition ──")
    comp: CompositionState = _timed("0_composition", compute_composition, sim)
    _log(comp.summary())

    _log("\n── Step 1: Domain ──")
    domain: DomainGeometry = _timed("1_domain", build_domain, comp)
    _log(domain.summary())

    # ── Steps 2–7: seed-dependent, retried on PercolationFailed ──────────
    volume: Optional[MicrostructureVolume] = None
    seed_used = sim.seed
    attempt = 0

    for attempt in range(max_retries):
        seed = sim.seed + attempt
        seed_used = seed
        rng = np.random.default_rng(seed)

        if attempt > 0:
            _log(f"\n── Retry {attempt}/{max_retries - 1} (seed={seed}) ──")
        else:
            _log(f"\n── Steps 2–7 (seed={seed}) ──")

        try:
            # ── Step 2: Carbon packing ────────────────────────────────────
            _log(" Step 2: Carbon scaffold packing...")
            packing: PackingResult = _timed(
                f"2_packing_attempt{attempt}",
                pack_carbon_scaffold,
                comp,
                domain,
                sim,
                rng,
            )
            _log(packing.summary())
            _collect_warnings(packing.warnings, "Step 2", pipeline_warns)

            # ── Step 2b: Rasterize carbon (pre-calender) ──────────────────
            _log(" Step 2b: Rasterizing carbon scaffold...")
            raster_pre: RasterResult = _timed(
                f"2b_raster_pre_attempt{attempt}",
                rasterize_carbon,
                packing,
                domain,
            )
            _log(raster_pre.summary())
            _collect_warnings(raster_pre.warnings, "Step 2b", pipeline_warns)
            carbon_label_pre = raster_pre.carbon_label

            # ── Step 3: Si VF map ─────────────────────────────────────────
            _log(" Step 3: Si volume-fraction mapping...")
            si_result: SiMapResult = _timed(
                f"3_si_mapper_attempt{attempt}",
                map_si_distribution,
                comp,
                domain,
                sim,
                carbon_label_pre,
                packing,
                rng,
            )
            _log(si_result.summary())
            _collect_warnings(si_result.warnings, "Step 3", pipeline_warns)

            # ── Step 4: CBD + Binder ──────────────────────────────────────
            _log(" Step 4: CBD + binder fill...")
            cbd_result: CBDBinderResult = _timed(
                f"4_cbd_binder_attempt{attempt}",
                fill_cbd_binder,
                comp,
                domain,
                sim,
                carbon_label_pre,
                si_result,
                rng,
            )
            _log(cbd_result.summary())
            _collect_warnings(cbd_result.warnings, "Step 4", pipeline_warns)

            # ── Step 5: Calendering ───────────────────────────────────────
            _log(" Step 5: Calendering...")
            t5 = time.perf_counter()

            # apply_calendering handles both particle mutation and field compression
            si_result, cbd_result = apply_calendering(
                packing.particles, comp, domain, si_result, cbd_result, sim
            )
            _collect_warnings(si_result.warnings, "Step 5-si", pipeline_warns)
            _collect_warnings(cbd_result.warnings, "Step 5-cbd", pipeline_warns)

            # Re-rasterize carbon in final-domain (post-calender) coordinates
            raster_final: RasterResult = rasterize_carbon(packing, domain)
            _collect_warnings(raster_final.warnings, "Step 5-raster", pipeline_warns)
            carbon_label = raster_final.carbon_label

            step_times[f"5_calendering_attempt{attempt}"] = time.perf_counter() - t5
            _log(
                f"  [5_calendering_attempt{attempt}] "
                f"{step_times[f'5_calendering_attempt{attempt}']:.3f}s"
            )

            # ── Step 6: SEI ───────────────────────────────────────────────
            _log(" Step 6: SEI formation...")
            sei_result: SEIResult = _timed(
                f"6_sei_attempt{attempt}",
                add_sei_shell,
                comp,
                domain,
                sim,
                carbon_label,
                si_result,
                rng,
            )
            _log(sei_result.summary())
            _collect_warnings(sei_result.warnings, "Step 6", pipeline_warns)

            # ── Step 7: Percolation ───────────────────────────────────────
            _log(" Step 7: Percolation validation...")
            perc_result: PercolationResult = _timed(
                f"7_percolation_attempt{attempt}",
                validate_percolation,
                comp,
                domain,
                sim,
                carbon_label,
                si_result,
                cbd_result,
                sei_result,
            )
            # ↑ raises PercolationFailed if electronic fraction < threshold
            _log(perc_result.summary())
            _collect_warnings(perc_result.warnings, "Step 7", pipeline_warns)

            # ── Step 8: Assemble MicrostructureVolume ─────────────────────
            _log(" Step 8: Assembling MicrostructureVolume...")
            volume = _timed(
                f"8_assemble_attempt{attempt}",
                assemble_volume,
                comp,
                domain,
                sim,
                packing,
                carbon_label,
                si_result,
                cbd_result,
                sei_result,
                perc_result,
            )
            _collect_warnings(volume.warnings, "Step 8", pipeline_warns)
            _log(volume.summary())

            # Success — exit retry loop
            break

        except PercolationFailed as pf:
            _log(
                f"  PercolationFailed: seed={pf.seed} "
                f"fraction={pf.fraction:.4f} < threshold={pf.threshold:.4f}. "
                f"Retrying with seed={seed + 1}."
            )
            pipeline_warns.append(
                f"Attempt {attempt + 1}: percolation failed "
                f"(seed={pf.seed}, fraction={pf.fraction:.4f}). "
                f"Retrying."
            )
            continue

    # ── Check for total failure ───────────────────────────────────────────
    if volume is None:
        raise RuntimeError(
            f"Pipeline failed: all {max_retries} attempts produced a "
            f"non-percolating structure. "
            f"Seeds tried: {sim.seed} – {sim.seed + max_retries - 1}. "
            f"Consider: (a) increasing target_porosity, "
            f"(b) reducing carbon_particle_d50_nm, "
            f"(c) increasing percolation_min_threshold, "
            f"(d) disabling percolation_enforce."
        )

    elapsed = time.perf_counter() - t_run_start

    result = PipelineResult(
        volume=volume,
        seed_used=seed_used,
        attempts=attempt + 1,
        elapsed_s=elapsed,
        step_times_s=step_times,
        warnings=pipeline_warns,
    )

    _log(result.summary())
    return result


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _collect_warnings(
    warns: list[str],
    step_name: str,
    pipeline_warns: list[str],
) -> None:
    """
    Prefix step warnings with their source step and append to pipeline_warns.
    Warnings already propagate into the result dataclasses — this gives a
    single flat list for the PipelineResult summary.
    """
    pipeline_warns.extend(f"[{step_name}] {w}" for w in warns)
