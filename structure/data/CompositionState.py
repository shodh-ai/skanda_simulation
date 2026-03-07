from dataclasses import dataclass, field
from structure.constants import _NM3_TO_CM3, _RSA_JAMMING_LIMIT


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class CompositionState:
    """
    Fully resolved composition for one simulation run.
    Produced by compute_composition(); consumed by every generation step.
    Never modified after construction.
    """

    # ── Weight fractions (stored for reference / logging) ──────────────────
    wf_si: float
    wf_carbon: float
    wf_additive: float
    wf_binder: float

    # ── Solid volume fractions (sum = 1.0 exactly) ─────────────────────────
    vf_si: float
    vf_carbon: float
    vf_additive: float
    vf_binder: float

    # ── Domain geometry ────────────────────────────────────────────────────
    domain_L_nm: float  # cube edge in nm (= coating_thickness_um × 1000)
    voxel_resolution: int  # 64 | 128 | 256
    V_domain_nm3: float
    V_solid_nm3: float
    porosity: float

    # ── Per-phase volumes in domain (nm³) ──────────────────────────────────
    V_si_nm3: float
    V_carbon_nm3: float
    V_additive_nm3: float
    V_binder_nm3: float

    # ── Carbon particle geometry ────────────────────────────────────────────
    carbon_d50_nm: float
    carbon_a_nm: float  # basal semi-axis = d50/2
    carbon_c_nm: float  # thickness semi-axis = a / aspect_ratio
    carbon_aspect_ratio: float
    carbon_V_median_nm3: float  # volume of the median particle (d50)
    carbon_V_mean_nm3: float  # PSD-corrected mean volume
    carbon_size_cv: float  # coefficient of variation for carbon particle size distribution (CV = σ/μ)

    # ── Si particle geometry ───────────────────────────────────────────────
    si_d50_nm: float
    si_r_nm: float  # = d50/2
    si_V_median_nm3: float
    si_V_mean_nm3: float  # PSD-corrected mean volume

    # ── Particle counts ────────────────────────────────────────────────────
    N_carbon: int  # target for explicit RSA packing
    N_si: int  # informational only — Si is placed statistically, not by RSA

    # ── Mass per phase in domain (g) ───────────────────────────────────────
    mass_si_g: float
    mass_carbon_g: float
    mass_additive_g: float
    mass_binder_g: float

    # ── Moles per phase in domain ──────────────────────────────────────────
    mol_si: float
    mol_carbon: float
    mol_additive: float
    mol_binder: float

    # ── Theoretical capacity ───────────────────────────────────────────────
    capacity_si_mah: float
    capacity_carbon_mah: float
    capacity_total_mah: float
    capacity_si_fraction: float  # Si share of total theoretical capacity
    volumetric_capacity_mah_cm3: float  # mAh / cm³ of electrode volume

    # ── Pre-calendering geometry ────────────────────────────────────────────
    compression_ratio: float
    L_z_pre_nm: float  # expanded Z domain before calendering
    phi_solid_pre: float  # solid vol fraction before calendering
    phi_carbon_pre: float  # carbon-only vol fraction in pre-calender box (RSA target)

    # ── Validation warnings (populated by _validate) ───────────────────────
    # Default must be last since it has a default value.
    validation_warnings: list[str] = field(default_factory=list)

    # ── Derived properties ─────────────────────────────────────────────────

    @property
    def voxel_size_nm(self) -> float:
        """Physical size of one output voxel in nm."""
        return self.domain_L_nm / self.voxel_resolution

    @property
    def V_domain_cm3(self) -> float:
        return self.V_domain_nm3 * _NM3_TO_CM3

    @property
    def si_in_voxels(self) -> float:
        """Si d50 expressed in voxel units — <1.0 means sub-voxel (statistical fill)."""
        return self.si_d50_nm / self.voxel_size_nm

    @property
    def carbon_in_voxels(self) -> float:
        """Carbon d50 expressed in voxel units."""
        return self.carbon_d50_nm / self.voxel_size_nm

    # ── Summary ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            "=" * 62,
            "  COMPOSITION STATE",
            "=" * 62,
            f"  Domain          : {self.domain_L_nm/1000:.1f} µm cube"
            f"  ({self.V_domain_cm3:.3e} cm³)",
            f"  Voxel size      : {self.voxel_size_nm:.1f} nm/voxel"
            f"  ({self.voxel_resolution}³)",
            "",
            "  Weight fractions (of total dry electrode mass):",
            f"    Si            : {self.wf_si:.4f}",
            f"    C-matrix      : {self.wf_carbon:.4f}",
            f"    Additive (CB) : {self.wf_additive:.4f}",
            f"    Binder        : {self.wf_binder:.4f}",
            f"    ─────────────   {self.wf_si+self.wf_carbon+self.wf_additive+self.wf_binder:.4f}",
            "",
            "  Solid volume fractions (of total solid):",
            f"    Si            : {self.vf_si:.4f}  ({self.vf_si*100:.2f}%)",
            f"    C-matrix      : {self.vf_carbon:.4f}  ({self.vf_carbon*100:.2f}%)",
            f"    Additive (CB) : {self.vf_additive:.4f}  ({self.vf_additive*100:.2f}%)",
            f"    Binder        : {self.vf_binder:.4f}  ({self.vf_binder*100:.2f}%)",
            f"    ─────────────   {self.vf_si+self.vf_carbon+self.vf_additive+self.vf_binder:.4f}",
            "",
            "  Particle counts:",
            f"    N_carbon      : {self.N_carbon}"
            f"  (explicit RSA,"
            f" d50={self.carbon_d50_nm/1000:.1f}µm,"
            f" AR={self.carbon_aspect_ratio:.1f},"
            f" {self.carbon_in_voxels:.0f} vx)",
            f"    N_Si          : {self.N_si:.2e}"
            f"  (statistical fill,"
            f" d50={self.si_d50_nm:.0f}nm,"
            f" {self.si_in_voxels:.2f} vx ← sub-voxel)",
            "",
            "  Moles in domain:",
            f"    Si            : {self.mol_si:.3e} mol",
            f"    C (graphite)  : {self.mol_carbon:.3e} mol",
            f"    CB (additive) : {self.mol_additive:.3e} mol",
            f"    Binder        : {self.mol_binder:.3e} mol",
            "",
            "  Theoretical capacity:",
            f"    Si            : {self.capacity_si_mah:.4e} mAh"
            f"  ({self.capacity_si_fraction*100:.1f}% of total)",
            f"    C-matrix      : {self.capacity_carbon_mah:.4e} mAh",
            f"    Total         : {self.capacity_total_mah:.4e} mAh",
            f"    Volumetric    : {self.volumetric_capacity_mah_cm3:.2f} mAh/cm³",
            "",
            "  Pre-calendering domain:",
            f"    compression   : {self.compression_ratio:.2f}",
            f"    L_z_pre       : {self.L_z_pre_nm/1000:.1f} µm"
            f"  (final: {self.domain_L_nm/1000:.1f} µm)",
            f"    φ_solid_pre   : {self.phi_solid_pre:.3f}"
            f"  ({self.phi_solid_pre*100:.1f}%)"
            f"  [RSA limit ≈ {_RSA_JAMMING_LIMIT*100:.0f}%]",
        ]
        if w := self.validation_warnings:
            lines += ["", f"  ⚠  {len(w)} WARNING(s):"]
            lines += [f"     [{i+1}] {msg}" for i, msg in enumerate(w)]
        lines.append("=" * 62)
        return "\n".join(lines)

    def raise_if_critical(self) -> None:
        """
        Re-raise validation warnings as errors for any critical issues.
        Call this if you want strict mode (e.g., in batch generation).
        """
        if critical := [
            w for w in self.validation_warnings if w.startswith("[CRITICAL]")
        ]:
            raise ValueError(
                f"CompositionState has {len(critical)} critical issue(s):\n"
                + "\n".join(critical)
            )
