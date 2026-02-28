The numbers give us the constraints everything else is designed around. Let me write the full plan.

---

## Key Numbers (Default Config)

| Quantity                         | Value                    | Implication                                  |
| -------------------------------- | ------------------------ | -------------------------------------------- |
| Domain                           | 50,000³ nm               | 50 µm cube                                   |
| Pre-calendering Z                | 71,429 nm                | domain expands before RSA, compresses after  |
| φ_solid (pre-calender)           | **42%**                  | RSA achievable without jamming               |
| φ_solid (post-calender)          | **60%**                  | physically correct density                   |
| Carbon particles                 | **346** oblate spheroids | explicit RSA feasible                        |
| Si particles                     | **12,000,000**           | RSA completely infeasible → statistical fill |
| voxel_size_nm                    | 390 nm/voxel             | Si (100nm = 0.26 vx) → sub-voxel             |
| Si share of theoretical capacity | **51.7%**                | at only 10 wt% — dominant despite small mass |

---

## Strategy Discussion

### Why expand Z before packing, then calender

Post-calendering you need φ_solid = 60% which is near random close packing (64% for spheres, less for flakes). RSA stops making progress near jamming. The physics solution: pack into the pre-calendering domain (42% target), then compress Z by `compression_ratio`. This matches the real manufacturing process exactly — wet coat is thicker, calendering densifies it.

### Why carbon is explicit and Si is statistical

346 oblate spheroids: an O(N²) RSA overlap check runs in microseconds — trivially explicit. 12 million Si spheres at 100nm: RSA at this count in a 50µm domain is computationally infeasible, and at 390nm/voxel you literally cannot resolve individual Si particles anyway. The statistically correct representation is a Si volume-fraction field — every voxel gets a float between 0 and 1 saying how much of it is Si. This is also physically more meaningful: it encodes the local Si loading, not an artificial hard-boundary voxel assignment.

### Why oblate spheroids and not sphere clumps

Aspect ratio 5 flakes need a representation. Sphere clumps (approximating the oblate with multiple spheres) give ~O(k²N²) overlap checks where k is spheres per particle — slow and inaccurate. Exact oblate spheroid overlap via the **Perram-Wertheim (1985) separating hyperplane criterion** gives the analytical test as a 1D optimization of a smooth function, solvable with `scipy.optimize.minimize_scalar` in microseconds. Correct geometry, fast enough for N=346.

### Why CBD and binder are phase-fill, not RSA

Carbon black aggregates at 200nm are still sub-voxel. The 2.33% volume fraction means they're a thin percolating film between the carbon scaffold particles. RSA for them would be both infeasible (count) and meaningless (resolution). Instead they are generated as a spatial phase-fill using a Gaussian Random Field (GRF) conditioned on the interstitial geometry.

---

## The 8-Step Plan

### Step 0 — [CompositionCalculator](./step_0.md)

```
Inputs: sim.composition (wt fracs), sim.silicon, sim.carbon, sim.additive, sim.binder
```

Converts weight fractions → volume fractions → particle counts → moles. This runs once before anything else and determines every count/volume the subsequent steps use.

```
wt_frac[i] / density[i]
─────────────────────── = vf_solid[i]    (solid volume fraction of phase i)
   Σ wt_frac[j] / density[j]

N_carbon = (vf_C × V_solid) / V_oblate_spheroid(d50, aspect_ratio)
N_Si     = (vf_Si × V_solid) / V_sphere(d50)     → stored but NOT used for RSA

moles[i] = (vf[i] × V_solid × density[i]) / MW[i]
capacity  = mol_Si × F × 3579_mAh_g / MW_Si  +  mol_C × F × 372_mAh_g / MW_C
```

Outputs stored in a `CompositionState` dataclass. Capacity is a sanity check — if it's wildly off, the composition was misconfigured.

---

### Step 1 — `DomainGeometry`

```
pre_calender_domain = Box(
    x = coating_thickness_nm,
    y = coating_thickness_nm,
    z = coating_thickness_nm / compression_ratio     # expanded Z
)
```

All particle placement happens in this expanded box. The Z compression comes later as a single coordinate transform. Periodic boundaries in X and Y (electrode is laterally continuous). Hard wall boundaries in Z (between current collector and air interface).

---

### Step 2 — `CarbonScaffoldPacker` (RSA oblate spheroids)

**Data structure for each particle:**

```python
@dataclass
class OblateSpheroid:
    center:     np.ndarray   # [x, y, z] in nm
    a:          float        # basal semi-axis (nm)  = d50/2
    c:          float        # thickness semi-axis   = a / aspect_ratio
    R:          np.ndarray   # 3×3 rotation matrix (orientation)
    phase_id:   int          = 1
```

**Orientation sampling:**

`orientation_degree` from config controls c-axis alignment toward Z. Sample from a **von Mises-Fisher distribution** on the unit sphere with concentration parameter κ derived from `orientation_degree`:

```
κ = 0             → isotropic (orientation_degree=0)
κ = 5             → moderate alignment (orientation_degree=0.6)
κ → ∞             → perfect alignment (orientation_degree=1.0)
```

c-axis direction sampled from vMF(μ=, κ), then construct rotation matrix. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/111797836/4dbd9121-eeaa-4988-bb1c-bedc1eb2108f/str_gen_config.yml)

**PSD sampling:**

d50 from config → log-normal distribution with `carbon_particle_size_cv`. Each particle gets its own (a, c) drawn from this distribution.

**Overlap test (Perram-Wertheim criterion):**

```python
def spheroids_overlap(p1: OblateSpheroid, p2: OblateSpheroid) -> bool:
    # Step 1: fast bounding sphere rejection
    dist = np.linalg.norm(p1.center - p2.center)
    if dist > (p1.a + p2.a):
        return False                       # guaranteed no overlap

    # Step 2: Perram-Wertheim separating hyperplane test
    # Build shape matrices A, B for each spheroid:
    #   A = R @ diag(1/a², 1/a², 1/c²) @ R.T
    A = p1.R @ np.diag([1/p1.a**2, 1/p1.a**2, 1/p1.c**2]) @ p1.R.T
    B = p2.R @ np.diag([1/p2.a**2, 1/p2.a**2, 1/p2.c**2]) @ p2.R.T
    d = p2.center - p1.center

    def F(s):
        M = (1-s) * np.linalg.inv(A) + s * np.linalg.inv(B)
        return s * (1-s) * d @ np.linalg.inv(M) @ d

    # Maximize F over [0,1]; if max(F) < 1 → overlap
    result = minimize_scalar(lambda s: -F(s), bounds=(0,1), method='bounded')
    return (-result.fun) < 1.0
```

**Spatial indexing:** Uniform 3D grid with cell size = 2 × max(a). Each cell stores particle indices. Overlap checks only against particles in neighboring cells → reduces O(N²) to ~O(N) in practice.

**RSA loop:**

```
while N_placed < N_carbon:
    propose (center, orientation)
    check overlap against spatial grid neighbors
    if no overlap:
        add to particle list + update spatial grid
    else:
        reject and try again (with max_attempts limit)
```

Target: fill pre-calendering domain to φ_solid_pre = 42%. RSA for oblate spheroids saturates around 30-40% depending on aspect ratio and orientation — at 42% we may need to allow slight particle overlap and resolve later via short MD relaxation, OR reduce aspect ratio slightly in placement then adjust post-calendering.

---

### Step 3 — `SiVfMapper` (statistical, not RSA)

Generates a Si volume-fraction map `si_vf[Nx, Ny, Nz]` where `Nx = voxel_resolution = 128`.

**For `si_distribution="embedded"`:**

```
1. Initialize si_vf = 0 everywhere
2. For each carbon particle:
     mark interior voxels using analytical oblate spheroid membership test
     assign si_vf_local = target_Si_vf_inside_C + N(0, uniformity_cv²)
3. If si_void_enabled:
     zero out a thin shell of thickness = void_radius around Si
     (implemented as morphological dilation of Si region then subtraction)
4. If si_coating_enabled:
     add a 1-voxel-thick shell at coating_thickness_nm scaled to voxels
     mark as phase_id=COATING
5. Apply overall normalization: sum(si_vf × V_voxel) == V_Si_target
```

Where `target_Si_vf_inside_C = vf_Si / vf_C` (Si volume per unit carbon volume).

**For `si_distribution="surface_anchored"`:**
Si vf concentrated in a band of depth ≈ 2×r_Si from each carbon particle surface, using a distance-transform from the carbon surface.

**For `si_distribution="core_shell"`:**
Carbon is placed as a shell around Si — the RSA order flips: Si cores placed first, then carbon shells grown around them.

---

### Step 4 — `CBDBinder_Fill` (GRF phase fill)

Fills interstitial space (not occupied by carbon or Si-void zones) with CBD.

For `conductive_additive_distribution="network"`:

- Generate a **Gaussian Random Field** in the interstitial space with correlation length ≈ `aggregate_size_nm / voxel_size_nm`
- Threshold the GRF to give the correct CBD volume fraction
- Ensure the thresholded field percolates (check with BFS) — retry if not

For `binder_distribution="necks"`:

- Find carbon particle contact zones using proximity detection between particle surfaces
- Concentrate binder vf in these contact regions using a Gaussian kernel centered at contact points

---

### Step 5 — `CalenderingTransform`

**Coordinate transform** on all particle centers and axes:

```python
def calender(particles, compression_ratio, particle_deformation):
    for p in particles:
        # Translate Z center
        p.center[2] *= compression_ratio

        # Deform particle shape: flatten in Z, expand in XY (volume-conserving)
        deform_factor = 1.0 + particle_deformation * (1/compression_ratio - 1)
        p.c *= (1.0 / deform_factor)   # thinner
        p.a *= (deform_factor ** 0.5)  # wider (volume conserved: a²c = const)

        # Update orientation: c-axis more aligned to Z after compression
        # Mix existing R toward Z-aligned by orientation_enhancement factor
        p.R = blend_toward_z_aligned(p.R, orientation_enhancement)
```

**Si/CBD vf maps** are scaled the same way: `zoom(si_vf, [1.0, 1.0, compression_ratio])`.

---

### Step 6 — `SEIShellAdder`

```
1. Compute surface voxels of carbon + Si using binary dilation - original
2. Assign SEI phase to surface voxels
3. Scale thickness: thickness_voxels = sei_thickness_nm / voxel_size_nm
   (at 390nm/voxel: 15nm SEI = 0.038 voxels → sub-voxel statistical shell)
4. Store as sei_vf map rather than binary, modulated by uniformity_cv GRF
```

---

### Step 7 — `PercolationValidator`

```
1. Build binary solid mask (carbon + Si + CBD)
2. BFS from all voxels on the Z=0 face
3. Compute: percolating_fraction = N_reachable_solid / N_total_solid
4. If percolating_fraction < min_threshold:
     raise PercolationFailed(run_id, seed) → caller retries with new seed
```

---

### Step 8 — `Voxelizer`

```
Inputs: particle list (physical nm coords) + vf maps for sub-voxel phases
Output: label_map uint8 (128³) + si_vf_map float16 (128³)

For each carbon particle:
    rasterize oblate spheroid → analytical voxel fill using
    parametric inside-test: (x-cx)²/a² + (y-cy)²/a² + (z-cz)²/c² ≤ 1
    after applying inverse rotation

Priority rule (when phases overlap at a voxel):
    SEI > Carbon > Si_vf_map > CBD > Pore
```

---

## Module Layout

```
structure/
  generation/
    __init__.py            # generate_si_graphite(sim: ResolvedSimulation) → np.ndarray
    composition.py         # Step 0: CompositionCalculator + CompositionState
    domain.py              # Step 1: DomainGeometry box definition
    carbon_packer.py       # Step 2: CarbonScaffoldPacker (RSA + oblate spheroids)
    si_mapper.py           # Step 3: SiVfMapper
    cbd_binder.py          # Step 4: CBDBinder_Fill (GRF)
    calendering.py         # Step 5: CalenderingTransform (replaces old voxel-space version)
    sei.py                 # Step 6: SEIShellAdder
    percolation.py         # Step 7: PercolationValidator
    voxelizer.py           # Step 8: Voxelizer
    utils.py               # OblateSpheroid dataclass, spatial grid, vMF sampler, GRF
```

The top-level `generate_si_graphite(sim)` calls steps 0–8 in sequence, handles the percolation retry loop, and returns `(label_map, si_vf_map, metadata_dict)` where `metadata_dict` carries the moles, capacity, actual porosity, and coordination number for use as ML labels.
