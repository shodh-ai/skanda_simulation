"""
Full pipeline test — Steps 0-8 (generation) + Steps A-C (simulation).

Output:
  output/microstructure.npz
  output/microstructure.tiff

Run:
  python test.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from structure import (
    load_gen_config,
    load_materials_db,
    resolve_generation,
    load_sim_config,
    resolve_simulation,
    run_generation,
    run_simulation,
    PipelineResult,
    SimulationResult,
)

OUT = Path("output")
OUT.mkdir(exist_ok=True)

# =============================================================================
# LOAD + RESOLVE
# =============================================================================
print("Loading generation config and materials DB...")
cfg = load_gen_config("str_gen_config.yml")
db = load_materials_db("materials_db.yml")
gen = resolve_generation(cfg, db)

print("Loading simulation config...")
sim_cfg = load_sim_config("simulation_config.yml")
sim = resolve_simulation(sim_cfg, db)
print(sim.summary())

# =============================================================================
# RUN GENERATION PIPELINE
# =============================================================================
try:
    result: PipelineResult = run_generation(gen, max_retries=10, verbose=True)
except RuntimeError as e:
    sys.exit(str(e))

# =============================================================================
# SAVE VOLUME
# =============================================================================
npz_path = OUT / "microstructure.npz"
tiff_path = OUT / "microstructure.tiff"

result.volume.save(str(npz_path))
result.volume.save_tiff(str(tiff_path))

print(f"\nSaved → {npz_path}")
print(f"Saved → {tiff_path}")

# =============================================================================
# RUN SIMULATION PIPELINE
# =============================================================================
sim_result: SimulationResult = run_simulation(vol=result.volume, sim=sim, verbose=True)

# =============================================================================
# PRINT RESULTS
# =============================================================================


def _v(val: float, unit: str = "", dec: int = 4) -> str:
    return "N/A" if math.isnan(val) else f"{val:.{dec}f} {unit}".strip()


SEP = "-" * 52
SEP2 = "=" * 52

print(f"\n{SEP2}")
print("  SIMULATION RESULTS")
print(SEP2)

# ── TauFactor ────────────────────────────────────────────────────
if sim_result.taufactor:
    tf = sim_result.taufactor
    print("\nTauFactor")
    print(SEP)
    print(f"  Ionic tortuosity (τ)         {_v(tf.tau_ionic)}")
    print(f"  Electrode porosity (ε)       {_v(tf.epsilon_ionic)}")
    print(f"  {'Effective diffusivity (D_eff)':<38} {tf.D_eff_ionic_m2_s:.4e} m²/s")
    print(f"  Bruggeman exponent (β)       {_v(tf.bruggeman_exponent)}")
    if tf.tau_electronic is not None:
        print(f"  Electronic tortuosity (τ_e)  {_v(tf.tau_electronic)}")
    conv = "converged" if tf.converged else "DID NOT CONVERGE"
    print(f"  Solver                       {conv}  res={tf.residual:.1e}")
    for w in tf.warnings:
        print(f"  ⚠  {w}")

# ── Rate capability ──────────────────────────────────────────────
if sim_result.rate_capability:
    rc = sim_result.rate_capability
    print("\nRate Capability")
    print(SEP)
    for c, q, e in zip(rc.c_rates, rc.capacities_mAh_cm2, rc.energy_densities_mWh_cm2):
        tag = "  ← nominal" if abs(c - 0.2) < 1e-6 else ""
        print(
            f"  {(f'C/{1/c:.0f}' if c <= 1.0 else f'{c:.1f}C'):<6}  "
            f"{_v(q, 'mAh/cm²')}   {_v(e, 'mWh/cm²')}{tag}"
        )
    for w in rc.warnings:
        print(f"  ⚠  {w}")

# ── DCIR ─────────────────────────────────────────────────────────
if sim_result.dcir:
    dc = sim_result.dcir
    print("\nDCIR Pulse")
    print(SEP)
    print(f"  Initial DCIR                 {_v(dc.dcir_mOhm_cm2, 'mΩ·cm²', 2)}")
    print(f"  Voltage drop (ΔV)            {_v(dc.delta_V_mV, 'mV', 2)}")
    print(
        f"  Condition                    SOC={dc.soc_point:.0%}  "
        f"{dc.pulse_c_rate}C  {dc.pulse_duration_s:.0f}s"
    )
    for w in dc.warnings:
        print(f"  ⚠  {w}")

# ── Cycle life ───────────────────────────────────────────────────
if sim_result.cycle_life:
    cl = sim_result.cycle_life
    print("\nCycle Life")
    print(SEP)
    life = (
        f"{cl.projected_cycle_life} cycles"
        if cl.projected_cycle_life
        else f"> {cl.cycles_run} cycles (EOL not reached)"
    )
    print(f"  Projected cycle life         {life}")
    print(
        f"  Capacity fade rate           {_v(cl.capacity_fade_rate_pct_per_cycle, '%/cycle')}"
    )
    print(
        f"  Q initial                    {_v(cl.initial_capacity_mAh_cm2, 'mAh/cm²')}"
    )
    print(
        f"  Q final (cycle {cl.cycles_run})           {_v(cl.final_capacity_mAh_cm2, 'mAh/cm²')}"
    )
    ret = cl.retention_at_final_cycle
    print(
        f"  Retention                    {'N/A' if math.isnan(ret) else f'{ret:.1%}'}"
    )
    for w in cl.warnings:
        print(f"  ⚠  {w}")

# ── Pipeline warnings ────────────────────────────────────────────
if sim_result.warnings:
    print(f"\nPipeline warnings")
    print(SEP)
    for i, w in enumerate(sim_result.warnings, 1):
        print(f"  [{i}] {w}")

# ── Timing ───────────────────────────────────────────────────────
print(f"\nTiming")
print(SEP)
print(
    f"  Generation   {result.elapsed_s:.1f}s  "
    f"({result.attempts} attempt(s), seed={result.seed_used})"
)
print(f"  Simulation   {sim_result.elapsed_s:.1f}s")
for step, t in sim_result.step_times_s.items():
    print(f"    {step:<30} {t:.3f}s")
print(SEP2)
