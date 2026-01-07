"""
Discrete particle packing for graphite.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def generate_particle_packing(
    shape: tuple,
    target_porosity: float,
    aspect_ratio: float,
    orientation_degree: float,
    seed: int,
) -> np.ndarray:
    """Generate overlapping particles to achieve target porosity."""

    print(f"  Generating particle packing...")

    rng = np.random.default_rng(seed)
    nz, ny, nx = shape

    solid = np.zeros(shape, dtype=bool)

    # Particle size - SMALLER for more particles
    min_dimension = min(nz, ny, nx)
    particle_radius = min_dimension / 10

    print(f"    Particle radius: {particle_radius:.1f} voxels")

    target_solid = 1.0 - target_porosity

    # Generate grid of centers
    spacing = particle_radius * 1.2  # Closer spacing = more overlap
    centers = []

    n_z = int(nz / spacing) + 1
    n_y = int(ny / spacing) + 1
    n_x = int(nx / spacing) + 1

    for iz in range(n_z):
        for iy in range(n_y):
            for ix in range(n_x):
                z = int(iz * spacing + rng.uniform(-spacing * 0.3, spacing * 0.3))
                y = int(iy * spacing + rng.uniform(-spacing * 0.3, spacing * 0.3))
                x = int(ix * spacing + rng.uniform(-spacing * 0.3, spacing * 0.3))

                if 0 <= z < nz and 0 <= y < ny and 0 <= x < nx:
                    centers.append((z, y, x))

    print(f"    Generated {len(centers)} particle centers")

    # Place particles
    count = 0
    for center in centers:
        current_solid = np.sum(solid) / solid.size
        if current_solid >= target_solid:
            break

        # Size variation
        size = particle_radius * rng.uniform(0.6, 1.4)

        # Ellipsoid
        if aspect_ratio > 1.5:
            r_z = size / np.sqrt(aspect_ratio)
            r_xy = size
        else:
            r_z = size
            r_xy = size

        # Orientation
        if rng.random() < orientation_degree:
            theta = rng.normal(0, 0.1)
        else:
            theta = rng.uniform(0, np.pi)

        # Create particle with irregularity
        particle = _create_particle(center, r_z, r_xy, theta, shape, rng)

        solid = np.logical_or(solid, particle)
        count += 1

    print(f"    Placed {count} particles")

    # Smooth slightly
    solid = gaussian_filter(solid.astype(float), sigma=0.5) > 0.5

    actual_porosity = 1.0 - np.mean(solid)
    print(f"    Final porosity: {actual_porosity:.3f}")

    return solid


def _create_particle(center, r_z, r_xy, theta, shape, rng):
    """Create irregular, angular particle (more realistic)."""
    cz, cy, cx = center
    nz, ny, nx = shape

    margin = int(max(r_z, r_xy) * 1.5) + 2
    z_min = max(0, cz - margin)
    z_max = min(nz, cz + margin)
    y_min = max(0, cy - margin)
    y_max = min(ny, cy + margin)
    x_min = max(0, cx - margin)
    x_max = min(nx, cx + margin)

    if z_max <= z_min or y_max <= y_min or x_max <= x_min:
        return np.zeros(shape, dtype=bool)

    zz, yy, xx = np.mgrid[z_min:z_max, y_min:y_max, x_min:x_max]

    zz = zz - cz
    yy = yy - cy
    xx = xx - cx

    # Rotation
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    zz_rot = zz * cos_t - xx * sin_t
    xx_rot = zz * sin_t + xx * cos_t

    # Base ellipsoid distance
    dist = (
        (zz_rot / (r_z + 1e-6)) ** 2
        + (yy / (r_xy + 1e-6)) ** 2
        + (xx_rot / (r_xy + 1e-6)) ** 2
    )

    # ENHANCED IRREGULARITY for angular appearance
    noise = rng.random((z_max - z_min, y_max - y_min, x_max - x_min))
    noise = gaussian_filter(noise, sigma=1.2)  # Smaller sigma = more angular

    # Stronger irregularity = more faceted/angular
    threshold = 1.0 + 0.25 * (noise - 0.5) * 2  # Increased from 0.15 to 0.25

    inside = dist <= threshold

    particle = np.zeros(shape, dtype=bool)
    particle[z_min:z_max, y_min:y_max, x_min:x_max] = inside

    return particle
