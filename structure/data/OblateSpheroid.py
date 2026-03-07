import math
import numpy as np
from dataclasses import dataclass, field
from structure.phases import PHASE_GRAPHITE


@dataclass
class OblateSpheroid:
    """
    One graphite flake particle in nm coordinates.

    Attributes:
        center  : [x, y, z] position in nm (pre-calendering coordinates)
        a       : basal semi-axis (nm) — the two equal long axes
        c       : thickness semi-axis (nm) = a / aspect_ratio
        R       : 3×3 rotation matrix; columns are the body-frame axes
                  in the lab frame. The c-axis is R[:, 2].
        A_inv   : cached inverse shape matrix = inv(R @ diag(a⁻²,a⁻²,c⁻²) @ R.T)
                  pre-computed once to avoid repeated inversion during overlap checks
        phase_id: always PHASE_GRAPHITE = 1
    """

    center: np.ndarray
    a: float
    c: float
    R: np.ndarray
    _A_inv: np.ndarray | None = field(repr=False, default=None)
    phase_id: int = PHASE_GRAPHITE

    @property
    def A_inv(self) -> np.ndarray:
        """
        Inverse shape matrix. Raises RuntimeError if stale (invalidated
        after a/c/R modification without recompute_shape_matrix() call).
        """
        if self._A_inv is None:
            raise RuntimeError(
                "OblateSpheroid.A_inv is None — shape matrix has been "
                "invalidated after a geometry modification (a, c, or R changed) "
                "without a subsequent call to recompute_shape_matrix(). "
                "Call p.recompute_shape_matrix() before using this particle."
            )
        return self._A_inv

    @A_inv.setter
    def A_inv(self, value: np.ndarray | None) -> None:
        self._A_inv = value

    @property
    def volume_nm3(self) -> float:
        return (4.0 / 3.0) * math.pi * self.a**2 * self.c

    @property
    def aspect_ratio(self) -> float:
        return self.a / self.c

    @property
    def c_axis(self) -> np.ndarray:
        """Unit vector along the c-axis (thickness direction) in lab frame."""
        return self.R[:, 2]

    @property
    def bounding_radius(self) -> float:
        """Radius of the smallest enclosing sphere. Used for fast rejection."""
        return self.a  # a ≥ c always for oblate

    def invalidate_shape_matrix(self) -> None:
        """
        Explicitly mark A_inv as stale before modifying a, c, or R.
        Forces a loud RuntimeError on any consumer that reads A_inv
        before recompute_shape_matrix() is called.
        """
        self._A_inv = None

    def recompute_shape_matrix(self) -> None:
        """
        Recompute A_inv from current a, c, R.
        Must be called after any modification to a, c, or R.
        """
        D = np.diag([1.0 / self.a**2, 1.0 / self.a**2, 1.0 / self.c**2])
        self._A_inv = np.linalg.inv(self.R @ D @ self.R.T)
