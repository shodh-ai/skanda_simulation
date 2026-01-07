from enum import Enum


class AnodeType(str, Enum):
    """Anode material types."""

    GRAPHITE = "graphite"
    SILICON_COMPOSITE = "silicon_composite"
    HARD_CARBON = "hard_carbon"
    SOFT_CARBON = "soft_carbon"
    LTO = "lto"


class GraphiteType(str, Enum):
    """Graphite subtypes."""

    NATURAL = "natural"
    ARTIFICIAL = "artificial"
    MCMB = "mcmb"


class ConductiveAdditiveType(str, Enum):
    """Conductive additive types."""

    CARBON_BLACK = "carbon_black"
    SUPER_P = "super_p"
    CNT = "cnt"
    GRAPHENE = "graphene"


class DistributionMode(str, Enum):
    """Conductive additive distribution modes."""

    AGGREGATE = "aggregate"
    DISPERSED = "dispersed"
    NETWORK = "network"


class BinderType(str, Enum):
    """Binder types."""

    PVDF = "pvdf"
    CMC = "cmc"
    SBR = "sbr"
    PAI = "pai"


class BinderDistribution(str, Enum):
    """Binder distribution patterns."""

    UNIFORM = "uniform"
    PATCHY = "patchy"
    NECKS = "necks"


class SiliconMorphology(str, Enum):
    """Silicon particle morphology."""

    SPHERICAL = "spherical"
    IRREGULAR = "irregular"
    POROUS = "porous"


class CarbonMatrixType(str, Enum):
    """Carbon matrix types for Si composite."""

    GRAPHITE = "graphite"
    HARD_CARBON = "hard_carbon"
    SOFT_CARBON = "soft_carbon"


class SiliconDistribution(str, Enum):
    """Silicon distribution in composite."""

    EMBEDDED = "embedded"
    SURFACE_ANCHORED = "surface_anchored"
    CORE_SHELL = "core_shell"


class CoatingType(str, Enum):
    """Coating types."""

    CARBON = "carbon"
    SILICON_OXIDE = "silicon_oxide"
    NONE = "none"


class PrecursorType(str, Enum):
    """Hard carbon precursor types."""

    BIOMASS = "biomass"
    RESIN = "resin"
    POLYMER = "polymer"


class ParticleShape(str, Enum):
    """Particle shape types."""

    SPHERICAL = "spherical"
    IRREGULAR = "irregular"
    FIBER = "fiber"
    CUBIC = "cubic"
    SHEET = "sheet"


class SoftCarbonPrecursor(str, Enum):
    """Soft carbon precursor types."""

    COAL_TAR_PITCH = "coal_tar_pitch"
    PETROLEUM_PITCH = "petroleum_pitch"
    MESOCARBON = "mesocarbon"


class LTOSynthesis(str, Enum):
    """LTO synthesis methods."""

    SOLID_STATE = "solid_state"
    HYDROTHERMAL = "hydrothermal"
    SOL_GEL = "sol_gel"


class CarbonNanomaterialType(str, Enum):
    """Carbon nanomaterial types."""

    GRAPHENE = "graphene"
    CNT = "cnt"
    BOTH = "both"
