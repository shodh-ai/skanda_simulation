from dataclasses import dataclass


@dataclass
class ParticleCrackParams:
    """Particle cracking defect parameters."""

    enabled: bool
    crack_probability: float
    crack_width_nm: float

    def __post_init__(self):
        if self.enabled:
            if not (0.0 <= self.crack_probability <= 0.2):
                raise ValueError(
                    f"crack_probability must be in [0.0, 0.2], got {self.crack_probability}"
                )

            if not (20.0 <= self.crack_width_nm <= 200.0):
                raise ValueError(
                    f"crack_width_nm must be in [20, 200], got {self.crack_width_nm}"
                )


@dataclass
class BinderAgglomerationParams:
    """Binder agglomeration defect parameters."""

    enabled: bool
    agglomeration_probability: float

    def __post_init__(self):
        if self.enabled:
            if not (0.0 <= self.agglomeration_probability <= 0.3):
                raise ValueError(
                    f"agglomeration_probability must be in [0.0, 0.3], got {self.agglomeration_probability}"
                )


@dataclass
class DelaminationParams:
    """Electrode-current collector delamination parameters."""

    enabled: bool
    delamination_fraction: float

    def __post_init__(self):
        if self.enabled:
            if not (0.0 <= self.delamination_fraction <= 0.2):
                raise ValueError(
                    f"delamination_fraction must be in [0.0, 0.2], got {self.delamination_fraction}"
                )


@dataclass
class PoreClusteringParams:
    """Non-uniform porosity clustering parameters."""

    enabled: bool
    clustering_degree: float

    def __post_init__(self):
        if self.enabled:
            if not (0.0 <= self.clustering_degree <= 1.0):
                raise ValueError(
                    f"clustering_degree must be in [0.0, 1.0], got {self.clustering_degree}"
                )


@dataclass
class DefectParams:
    """Defect and variation parameters for microstructure diversity."""

    particle_cracks: ParticleCrackParams
    binder_agglomeration: BinderAgglomerationParams
    delamination: DelaminationParams
    pore_clustering: PoreClusteringParams
