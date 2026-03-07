"""
Full pipeline test — Steps 0-8.

Output:
  output/microstructure.npz

Run:
  python test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from structure import load_run_config, load_materials_db, resolve, run, PipelineResult

OUT = Path("output")
OUT.mkdir(exist_ok=True)

# =============================================================================
# LOAD + RESOLVE
# =============================================================================
print("Loading config and materials DB...")
cfg = load_run_config("str_gen_config.yml")
db = load_materials_db("materials_db.yml")
sim = resolve(cfg, db)

# =============================================================================
# RUN PIPELINE
# =============================================================================
try:
    result: PipelineResult = run(sim, max_retries=10, verbose=True)
except RuntimeError as e:
    sys.exit(str(e))

# =============================================================================
# SAVE .NPZ
# =============================================================================
result.volume.save(str(OUT / "microstructure.npz"))
print(f"\nSaved → output/microstructure.npz")
print(f"Seed used : {result.seed_used}")
print(f"Attempts  : {result.attempts}")
print(f"Time      : {result.elapsed_s:.1f}s")
