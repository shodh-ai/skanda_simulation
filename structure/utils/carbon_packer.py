import math
import numpy as np
from scipy.optimize import minimize_scalar
from structure.data import OblateSpheroid, DomainGeometry


def _make_spheroid(
    center: np.ndarray,
    a: float,
    c: float,
    R: np.ndarray,
) -> OblateSpheroid:
    """
    Construct an OblateSpheroid with correctly computed A_inv.
    Always use this factory rather than constructing OblateSpheroid directly.
    """
    p = OblateSpheroid(center=center, a=a, c=c, R=R)
    p.recompute_shape_matrix()
    return p


def _spheroids_overlap(
    p1: OblateSpheroid,
    p2: OblateSpheroid,
    domain: DomainGeometry,
) -> bool:
    """
    Test overlap between two oblate spheroids using the
    Perram-Wertheim (1985) separating hyperplane criterion.

    Two spheroids do NOT overlap iff there exists s ∈ [0,1] such that:
        F(s) = s(1-s) d·M(s)⁻¹·d > 1
    where M(s) = (1-s)A⁻¹ + s B⁻¹  and  d = center_2 - center_1.

    Step 1 (fast reject): bounding sphere check.
        If |d| > a1 + a2, cannot possibly overlap.
    Step 2 (exact test): maximize F(s) over [0,1].
        If max F < 1 → overlap. Else → no overlap.

    Periodic boundary in X, Y handled via minimum-image displacement.
    """
    # Fast bounding sphere rejection (uses MIC displacement for periodicity)
    d = domain.min_image_vector(p1.center, p2.center)
    dist = float(np.linalg.norm(d))
    if dist > (p1.bounding_radius + p2.bounding_radius):
        return False

    # Exact Perram-Wertheim test
    A_inv = p1.A_inv
    B_inv = p2.A_inv

    def neg_F(s: float) -> float:
        M_inv = (1.0 - s) * A_inv + s * B_inv
        M = np.linalg.inv(M_inv)
        return -(s * (1.0 - s) * float(d @ M @ d))

    result = minimize_scalar(
        neg_F, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-6}
    )
    F_max = -result.fun
    return F_max < 1.0


def _sample_size(
    d50_nm: float,
    size_cv: float,
    rng: np.random.Generator,
    aspect_ratio: float = 5.0,
) -> tuple[float, float]:
    """
    Draw one (a, c) pair from the log-normal PSD.

    The aspect ratio is fixed per the config — only the overall size
    (d50) varies between particles, not their flatness.

        d  ~ LogNormal(median=d50, cv=size_cv)
        a  = d / 2
        c  = a / aspect_ratio
    """
    sigma_ln = math.sqrt(math.log(1.0 + size_cv**2))
    mu_ln = math.log(d50_nm)
    d = float(rng.lognormal(mu_ln, sigma_ln))
    a = d / 2.0
    c = a / aspect_ratio
    return a, c


def _sample_rotation(kappa: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample a 3×3 rotation matrix R such that the c-axis (R[:, 2])
    is drawn from vMF(μ=[0,0,1], κ).

    κ = 0     → isotropic (random orientation)
    κ = large → c-axis closely aligned with Z

    The two basal axes (R[:, 0], R[:, 1]) are chosen to form a
    right-handed orthonormal frame with the sampled c-axis.
    """
    c_axis = _sample_vmf_z(kappa, rng)

    # Build orthonormal frame: Gram-Schmidt with a random transverse vector
    v = rng.normal(size=3)
    v -= v.dot(c_axis) * c_axis
    norm_v = float(np.linalg.norm(v))
    if norm_v < 1e-10:
        # c_axis nearly parallel to v — use a fixed fallback
        v = np.array([1.0, 0.0, 0.0])
        v -= v.dot(c_axis) * c_axis
        norm_v = float(np.linalg.norm(v))
    e1 = v / norm_v
    e2 = np.cross(c_axis, e1)

    return np.column_stack([e1, e2, c_axis])


def _sample_vmf_z(kappa: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample one unit vector from vMF(μ=[0,0,1], κ) using Wood (1994).

    Returns:
        Unit vector np.ndarray of shape (3,).
    """
    if kappa < 1e-6:
        # Uniform on sphere — Muller method
        u = rng.normal(size=3)
        return u / np.linalg.norm(u)

    # Wood (1994) algorithm
    dim = 3
    b = (-2.0 * kappa + math.sqrt(4.0 * kappa**2 + (dim - 1) ** 2)) / (dim - 1)
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (dim - 1) * math.log(1.0 - x0**2)

    while True:
        z = float(rng.beta((dim - 1) / 2.0, (dim - 1) / 2.0))
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = float(rng.uniform())
        if kappa * w + (dim - 1) * math.log(max(1.0 - x0 * w, 1e-300)) - c >= math.log(
            u
        ):
            break

    phi = float(rng.uniform(0.0, 2.0 * math.pi))
    sin_theta = math.sqrt(max(0.0, 1.0 - w**2))
    return np.array([sin_theta * math.cos(phi), sin_theta * math.sin(phi), w])


def _od_to_kappa(orientation_degree: float) -> float:
    """
    Map orientation_degree ∈ [0, 1] to vMF concentration κ ≥ 0.

    Mapping: κ = -10 × ln(1 - orientation_degree)
    od=0.0  → κ=0     (isotropic — uniform sphere)
    od=0.60 → κ≈9.2   (strong Z preference)
    od=0.95 → κ≈30.0  (near-perfect alignment)
    od=1.0  → κ=∞     (clamped to 1000 — effectively perfect)

    The clamp at κ=1000 means od=0.99999 and od=1.0 are numerically
    identical. RunConfig.validate_orientation_degree_clamp() warns the
    user at config-load time when od >= 1.0 or od > 0.95.

    Calibrated so that mean|cos θ| matches the qualitative
    orientation_degree label (0=random, 1=perfect).
    """
    if orientation_degree <= 0.0:
        return 0.0
    if orientation_degree >= 1.0:
        return 1000.0
    kappa = -10.0 * math.log(1.0 - orientation_degree)
    return min(kappa, 1000.0)
