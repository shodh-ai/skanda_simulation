"""
MicrostructureVolume — canonical output container after Step 7.

Replaces the discrete VoxelGrid with a richer multi-phase float32
volume-fraction representation. Every phase is stored as float32
(nx, ny, nz), enabling direct use in:
  - TauFactor  : feed to_pore_mask() or pore_vf directly
  - PyBaMM     : feed carbon_vf, si_vf as microstructure params

pore_vf is NOT an independent field — it is derived at assembly as:
    pore_vf = clip(1 - sum(all solid VF fields), 0, 1)

Self-consistency is validated on assembly and stored as warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import json
import tifffile

from structure.phases import (
    PHASE_GRAPHITE,
    PHASE_SI,
    PHASE_COATING,
    PHASE_CBD,
    PHASE_BINDER,
    PHASE_SEI,
    PHASE_GRAYSCALE,
    _GRAYSCALE_TO_PHASE,
)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class VolumeMetadata:
    """
    Scalar provenance captured at assembly time.
    All fields survive numpy scalar round-trip through .npz.
    Flat by design — no nested objects so load() is trivial.
    """

    # Run identity
    run_id: int
    seed: int

    # Geometry
    voxel_size_nm: float
    voxel_resolution: int
    electrode_thickness_um: float

    # Composition — weight fractions
    wf_si: float
    wf_carbon: float
    wf_additive: float
    wf_binder: float

    # Composition — solid volume fractions
    vf_si: float
    vf_carbon: float
    vf_additive: float
    vf_binder: float

    # Porosity
    target_porosity: float
    measured_porosity: float

    # Capacity
    capacity_total_mah: float
    capacity_si_fraction: float
    volumetric_capacity_mah_cm3: float

    # Percolation
    electronic_fraction: float
    ionic_fraction: float
    electronic_percolating: bool
    ionic_percolating: bool

    # Particle info
    n_carbon_particles: int
    carbon_d50_nm: float
    si_d50_nm: float
    si_coating_enabled: bool
    si_coating_thickness_nm: float

    # ── Carbon electrochemical properties — from GraphiteMaterial in DB ───────
    carbon_li_diffusivity_m2_s: float
    carbon_electrical_conductivity_S_m: float
    carbon_theoretical_capacity_mAh_g: float
    carbon_density_g_cm3: float
    carbon_molar_mass_g_mol: float

    # ── Silicon electrochemical properties — computed at si_d50_nm ────────────
    # D and σ are size-dependent (correlations in SiBaseMaterial).
    # Intrinsic properties (capacity, density, molar_mass, expansion) are fixed.
    si_li_diffusivity_m2_s: float
    si_electrical_conductivity_S_m: float
    si_theoretical_capacity_mAh_g: float
    si_density_g_cm3: float
    si_molar_mass_g_mol: float
    si_volume_expansion_factor: float
    si_young_modulus_GPa: float
    si_poisson_ratio: float


# ---------------------------------------------------------------------------
# MicrostructureVolume
# ---------------------------------------------------------------------------


@dataclass
class MicrostructureVolume:
    """
    Multi-phase VF container for one Si-graphite microstructure.

    All arrays are float32 (nx, ny, nz), values in [0, 1].
    """

    carbon_vf: np.ndarray  # float32
    si_vf: np.ndarray  # float32
    coating_vf: np.ndarray  # float32
    cbd_vf: np.ndarray  # float32
    binder_vf: np.ndarray  # float32
    sei_vf: np.ndarray  # float32
    pore_vf: np.ndarray  # float32

    metadata: VolumeMetadata
    warnings: list[str] = field(default_factory=list)

    # ── Shape ────────────────────────────────────────────────────────────

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.carbon_vf.shape  # type: ignore[return-value]

    # ── Phase fractions ───────────────────────────────────────────────────

    def phase_fractions(self) -> dict[str, float]:
        """
        Mean VF per phase across the full domain.
        Sum equals 1.0 by construction (pore fills the remainder).
        Uses float64 accumulation to avoid float32 rounding on large grids.
        """
        N = float(self.carbon_vf.size)
        return {
            "Pore": float(self.pore_vf.astype(np.float64).sum()) / N,
            "Graphite": float(self.carbon_vf.astype(np.float64).sum()) / N,
            "Silicon": float(self.si_vf.astype(np.float64).sum()) / N,
            "Coating": float(self.coating_vf.astype(np.float64).sum()) / N,
            "CBD": float(self.cbd_vf.astype(np.float64).sum()) / N,
            "Binder": float(self.binder_vf.astype(np.float64).sum()) / N,
            "SEI": float(self.sei_vf.astype(np.float64).sum()) / N,
        }

    # ── TauFactor interface ───────────────────────────────────────────────

    def to_pore_mask(self, threshold: float = 0.5) -> np.ndarray:
        """
        Boolean pore mask for TauFactor tortuosity calculation.

        Args:
            threshold : voxels with pore_vf >= threshold are True.
                        0.5  → majority-pore (standard, conservative)
                        0.1  → any meaningful pore presence (permissive)

        Returns:
            bool (nx, ny, nz)
        """
        return self.pore_vf >= threshold

    # ── Visualization interface ───────────────────────────────────────────

    def dominant_phase_map(self) -> np.ndarray:
        """
        Per-voxel dominant phase label using LABEL_PRIORITY ordering.
        Returns uint8 (nx, ny, nz) with values matching PHASE_* constants.

        Computed on-demand — not stored. Use for cross-section slices
        and 3D rendering. Identical priority semantics to the old
        VoxelGrid.label_map but derived from float fields, not stored.

        Thresholds per phase (same as old voxelizer._THRESHOLDS):
            SEI      ≥ 0.005
            Graphite ≥ 0.50
            Coating  ≥ 0.02
            Silicon  ≥ 0.02
            CBD      ≥ 0.10
            Binder   ≥ 0.15
        """
        out = np.zeros(self.shape, dtype=np.uint8)  # default = PHASE_PORE = 0

        # Apply in ascending LABEL_PRIORITY so higher priority overwrites
        for phase_id, arr, thr in [
            (PHASE_BINDER, self.binder_vf, 0.15),
            (PHASE_CBD, self.cbd_vf, 0.10),
            (PHASE_SI, self.si_vf, 0.02),
            (PHASE_COATING, self.coating_vf, 0.02),
            (PHASE_GRAPHITE, self.carbon_vf, 0.50),
            (PHASE_SEI, self.sei_vf, 0.005),
        ]:
            out[arr >= thr] = phase_id

        return out

    # ── Save ─────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Save all VF arrays + flattened metadata to compressed .npz.

        Array keys : carbon_vf, si_vf, coating_vf, cbd_vf,
                     binder_vf, sei_vf, pore_vf  (all float32)
        Metadata keys : meta_<field_name>  (numpy scalars)
        """
        m = self.metadata
        np.savez_compressed(
            str(path),
            # Phase arrays
            carbon_vf=self.carbon_vf,
            si_vf=self.si_vf,
            coating_vf=self.coating_vf,
            cbd_vf=self.cbd_vf,
            binder_vf=self.binder_vf,
            sei_vf=self.sei_vf,
            pore_vf=self.pore_vf,
            # Metadata scalars — prefixed meta_ to avoid key collisions
            meta_run_id=np.int32(m.run_id),
            meta_seed=np.int32(m.seed),
            meta_voxel_size_nm=np.float32(m.voxel_size_nm),
            meta_voxel_resolution=np.int32(m.voxel_resolution),
            meta_electrode_thickness_um=np.float32(m.electrode_thickness_um),
            meta_wf_si=np.float32(m.wf_si),
            meta_wf_carbon=np.float32(m.wf_carbon),
            meta_wf_additive=np.float32(m.wf_additive),
            meta_wf_binder=np.float32(m.wf_binder),
            meta_vf_si=np.float32(m.vf_si),
            meta_vf_carbon=np.float32(m.vf_carbon),
            meta_vf_additive=np.float32(m.vf_additive),
            meta_vf_binder=np.float32(m.vf_binder),
            meta_target_porosity=np.float32(m.target_porosity),
            meta_measured_porosity=np.float32(m.measured_porosity),
            meta_capacity_total_mah=np.float32(m.capacity_total_mah),
            meta_capacity_si_fraction=np.float32(m.capacity_si_fraction),
            meta_volumetric_capacity_mah_cm3=np.float32(m.volumetric_capacity_mah_cm3),
            meta_electronic_fraction=np.float32(m.electronic_fraction),
            meta_ionic_fraction=np.float32(m.ionic_fraction),
            meta_electronic_percolating=np.bool_(m.electronic_percolating),
            meta_ionic_percolating=np.bool_(m.ionic_percolating),
            meta_n_carbon_particles=np.int32(m.n_carbon_particles),
            meta_carbon_d50_nm=np.float32(m.carbon_d50_nm),
            meta_si_d50_nm=np.float32(m.si_d50_nm),
            meta_si_coating_enabled=np.bool_(m.si_coating_enabled),
            meta_si_coating_thickness_nm=np.float32(m.si_coating_thickness_nm),
            # Carbon electrochemical
            meta_carbon_li_diffusivity_m2_s=np.float32(m.carbon_li_diffusivity_m2_s),
            meta_carbon_electrical_conductivity_S_m=np.float32(
                m.carbon_electrical_conductivity_S_m
            ),
            meta_carbon_theoretical_capacity_mAh_g=np.float32(
                m.carbon_theoretical_capacity_mAh_g
            ),
            meta_carbon_density_g_cm3=np.float32(m.carbon_density_g_cm3),
            meta_carbon_molar_mass_g_mol=np.float32(m.carbon_molar_mass_g_mol),
            # Silicon electrochemical
            meta_si_li_diffusivity_m2_s=np.float32(m.si_li_diffusivity_m2_s),
            meta_si_electrical_conductivity_S_m=np.float32(
                m.si_electrical_conductivity_S_m
            ),
            meta_si_theoretical_capacity_mAh_g=np.float32(
                m.si_theoretical_capacity_mAh_g
            ),
            meta_si_density_g_cm3=np.float32(m.si_density_g_cm3),
            meta_si_molar_mass_g_mol=np.float32(m.si_molar_mass_g_mol),
            meta_si_volume_expansion_factor=np.float32(m.si_volume_expansion_factor),
            warnings_json=np.bytes_(json.dumps(self.warnings)),
        )

    # ── Load ─────────────────────────────────────────────────────────────

    @staticmethod
    def load(path: str | Path) -> "MicrostructureVolume":
        """Load a saved MicrostructureVolume from .npz."""
        d = np.load(str(path))

        meta = VolumeMetadata(
            run_id=int(d["meta_run_id"]),
            seed=int(d["meta_seed"]),
            voxel_size_nm=float(d["meta_voxel_size_nm"]),
            voxel_resolution=int(d["meta_voxel_resolution"]),
            electrode_thickness_um=float(d["meta_electrode_thickness_um"]),
            wf_si=float(d["meta_wf_si"]),
            wf_carbon=float(d["meta_wf_carbon"]),
            wf_additive=float(d["meta_wf_additive"]),
            wf_binder=float(d["meta_wf_binder"]),
            vf_si=float(d["meta_vf_si"]),
            vf_carbon=float(d["meta_vf_carbon"]),
            vf_additive=float(d["meta_vf_additive"]),
            vf_binder=float(d["meta_vf_binder"]),
            target_porosity=float(d["meta_target_porosity"]),
            measured_porosity=float(d["meta_measured_porosity"]),
            capacity_total_mah=float(d["meta_capacity_total_mah"]),
            capacity_si_fraction=float(d["meta_capacity_si_fraction"]),
            volumetric_capacity_mah_cm3=float(d["meta_volumetric_capacity_mah_cm3"]),
            electronic_fraction=float(d["meta_electronic_fraction"]),
            ionic_fraction=float(d["meta_ionic_fraction"]),
            electronic_percolating=bool(d["meta_electronic_percolating"]),
            ionic_percolating=bool(d["meta_ionic_percolating"]),
            n_carbon_particles=int(d["meta_n_carbon_particles"]),
            carbon_d50_nm=float(d["meta_carbon_d50_nm"]),
            si_d50_nm=float(d["meta_si_d50_nm"]),
            si_coating_enabled=bool(d["meta_si_coating_enabled"]),
            si_coating_thickness_nm=float(d["meta_si_coating_thickness_nm"]),
            carbon_li_diffusivity_m2_s=float(d["meta_carbon_li_diffusivity_m2_s"]),
            carbon_electrical_conductivity_S_m=float(
                d["meta_carbon_electrical_conductivity_S_m"]
            ),
            carbon_theoretical_capacity_mAh_g=float(
                d["meta_carbon_theoretical_capacity_mAh_g"]
            ),
            carbon_density_g_cm3=float(d["meta_carbon_density_g_cm3"]),
            carbon_molar_mass_g_mol=float(d["meta_carbon_molar_mass_g_mol"]),
            si_li_diffusivity_m2_s=float(d["meta_si_li_diffusivity_m2_s"]),
            si_electrical_conductivity_S_m=float(
                d["meta_si_electrical_conductivity_S_m"]
            ),
            si_theoretical_capacity_mAh_g=float(
                d["meta_si_theoretical_capacity_mAh_g"]
            ),
            si_density_g_cm3=float(d["meta_si_density_g_cm3"]),
            si_molar_mass_g_mol=float(d["meta_si_molar_mass_g_mol"]),
            si_volume_expansion_factor=float(d["meta_si_volume_expansion_factor"]),
        )
        warnings = json.loads(d["warnings_json"].item().decode("utf-8"))

        return MicrostructureVolume(
            carbon_vf=d["carbon_vf"],
            si_vf=d["si_vf"],
            coating_vf=d["coating_vf"],
            cbd_vf=d["cbd_vf"],
            binder_vf=d["binder_vf"],
            sei_vf=d["sei_vf"],
            pore_vf=d["pore_vf"],
            metadata=meta,
            warnings=warnings,
        )

    # ── TIFF export / import ──────────────────────────────────────────────────
    def save_tiff(self, path: str | Path) -> None:
        """
        Save the microstructure as a grayscale TIFF Z-stack.

        Format mirrors a BSE-SEM image stack:
        - dtype  : uint8
        - shape  : (nz, ny, nx) — Z-slices as pages, standard microscopy
                    convention (depth-first, same as ImageJ / Fiji / Dragonfly)
        - values : grayscale intensities from PHASE_GRAYSCALE, ordered by
                    mean atomic number so brightness ∝ Z̄ (BSE-SEM convention)

        The dominant phase per voxel is determined by dominant_phase_map()
        (same priority logic as the old VoxelGrid label_map). Float VF
        fields are NOT recoverable from the TIFF — use save() / load() for
        full-fidelity round-trips.

        ImageJ metadata is embedded so the stack opens correctly with
        voxel_size_nm written as the spatial calibration.

        Args:
            path : output path, e.g. "output/microstructure.tiff"

        Raises:
            ImportError if tifffile is not installed.
        """

        label_map = self.dominant_phase_map()
        nx, ny, nz = label_map.shape

        # Build grayscale LUT: phase_id → uint8 grey value
        # Max phase ID is 6 (PHASE_SEI) — LUT size = 7
        lut = np.zeros(7, dtype=np.uint8)
        for phase_id, grey in PHASE_GRAYSCALE.items():
            lut[phase_id] = np.uint8(grey)

        # Apply LUT: label_map values are phase IDs (0–6)
        grey_map = lut[label_map]  # uint8 (nx, ny, nz)

        # Reorder to (nz, ny, nx) — standard TIFF/microscopy Z-stack convention
        grey_stack = np.transpose(grey_map, (2, 1, 0))  # (nz, ny, nx)

        vs_um = self.metadata.voxel_size_nm / 1000.0

        # ImageJ metadata: spatial calibration so Fiji reads voxel size correctly
        imagej_metadata = {
            "unit": "um",
            "spacing": vs_um,  # Z spacing
        }

        tifffile.imwrite(
            str(path),
            grey_stack,
            imagej=True,
            resolution=(1.0 / vs_um, 1.0 / vs_um),  # XY pixels per µm
            metadata=imagej_metadata,
            compression="zlib",  # lossless, ~2–4× smaller than uncompressed
            photometric="minisblack",
        )

    @staticmethod
    def load_tiff(path: str | Path) -> np.ndarray:
        """
        Load a grayscale TIFF Z-stack saved by save_tiff().

        Reverses the grayscale → phase ID mapping using nearest-neighbour
        lookup against PHASE_GRAYSCALE values. Returns a uint8 label array
        in the standard (nx, ny, nz) axis convention used by all other
        methods in this class.

        Note:
            Float VF fields (carbon_vf, si_vf, etc.) cannot be recovered
            from a TIFF — this returns only the dominant-phase label map.
            For full round-trip fidelity, use MicrostructureVolume.load().

        Args:
            path : path to a .tiff file saved by save_tiff()

        Returns:
            uint8 (nx, ny, nz) dominant-phase label array with PHASE_* values.

        Raises:
            ImportError if tifffile is not installed.
            ValueError  if the TIFF contains grayscale values not in
                        PHASE_GRAYSCALE (i.e., not generated by save_tiff).
        """

        grey_stack = tifffile.imread(str(path))  # (nz, ny, nx) uint8

        # Validate: all values must be known grayscale levels
        unique_vals = set(np.unique(grey_stack).tolist())
        known_vals = set(_GRAYSCALE_TO_PHASE.keys())
        if unknown := unique_vals - known_vals:
            raise ValueError(
                f"TIFF contains grayscale values not in PHASE_GRAYSCALE: "
                f"{sorted(unknown)}. "
                f"Expected values: {sorted(known_vals)}. "
                f"This TIFF was not generated by MicrostructureVolume.save_tiff()."
            )

        # Build reverse LUT: grey value → phase ID
        # Grey values go up to 255; only 7 are valid (0,30,55,70,90,120,200)
        reverse_lut = np.zeros(256, dtype=np.uint8)
        for grey, phase_id in _GRAYSCALE_TO_PHASE.items():
            reverse_lut[grey] = np.uint8(phase_id)

        label_stack = reverse_lut[grey_stack]  # (nz, ny, nx) uint8 phase IDs

        return np.transpose(label_stack, (2, 1, 0))

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        m = self.metadata
        fracs = self.phase_fractions()
        nx, ny, nz = self.shape
        N = nx * ny * nz

        mb = (
            sum(
                a.nbytes
                for a in [
                    self.carbon_vf,
                    self.si_vf,
                    self.coating_vf,
                    self.cbd_vf,
                    self.binder_vf,
                    self.sei_vf,
                    self.pore_vf,
                ]
            )
            / 1e6
        )

        solid_sum = (
            self.carbon_vf.astype(np.float64)
            + self.si_vf.astype(np.float64)
            + self.coating_vf.astype(np.float64)
            + self.cbd_vf.astype(np.float64)
            + self.binder_vf.astype(np.float64)
            + self.sei_vf.astype(np.float64)
        )
        overlap_frac = float((solid_sum > 1.01).sum()) / N

        lines = [
            "=" * 66,
            " MICROSTRUCTURE VOLUME",
            "=" * 66,
            f"  Shape           : {nx}×{ny}×{nz} = {N:,} voxels",
            f"  Voxel size      : {m.voxel_size_nm:.2f} nm",
            f"  Memory (float32): {mb:.1f} MB",
            "",
            "  Phase volume fractions (mean VF across domain):",
            f"  {'Phase':<12} {'Mean VF':>10}  Bar (40 cols = 100%)",
            f"  {'-'*12} {'-'*10}  " + "-" * 40,
        ]

        for name, frac in fracs.items():
            bar = "█" * int(frac * 40)
            lines.append(f"  {name:<12} {frac:>10.4f}  {bar}")

        por_delta = abs(m.measured_porosity - m.target_porosity)
        lines += [
            "",
            f"  Porosity : measured={m.measured_porosity:.4f}  "
            f"target={m.target_porosity:.4f}  "
            f"Δ={por_delta*100:.2f} pp",
            f"  Overlap  : solid_sum > 1.01 in " f"{overlap_frac*100:.3f}% of voxels",
            "",
            "  Percolation:",
            f"    Electronic : {m.electronic_fraction:.4f} "
            f"({'✓ PASS' if m.electronic_percolating else '✗ FAIL'})",
            f"    Ionic      : {m.ionic_fraction:.4f} "
            f"({'✓ PASS' if m.ionic_percolating else '✗ FAIL'})",
            "",
            f"  Capacity : {m.capacity_total_mah:.3e} mAh total | "
            f"Si={m.capacity_si_fraction*100:.1f}% | "
            f"{m.volumetric_capacity_mah_cm3:.1f} mAh/cm³",
            f"  Particles: {m.n_carbon_particles} graphite | "
            f"C d50={m.carbon_d50_nm/1000:.1f}µm | "
            f"Si d50={m.si_d50_nm:.0f}nm",
        ]

        if self.warnings:
            lines += ["", f"  ⚠ {len(self.warnings)} WARNING(s) (serialized to .npz):"]
            lines += [f"    [{i+1}] {w}" for i, w in enumerate(self.warnings)]

        lines.append("=" * 66)
        return "\n".join(lines)
