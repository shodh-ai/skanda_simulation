# Large Physics Model (LPM) for Porous Media: A Complete Technical Reference

**Harshit Sandilya | Pre-Meeting Technical Documentation | March 2026**

---

## Preface: The Architecture of This Document

You are an ML/DL/RL engineer who has already built a microstructure pipeline and a PyBaMM-based 1D cell-level simulator. This document builds directly on top of that experience. Every physics concept is introduced through the lens of what you already know — operators, function approximators, differentiable solvers, dataset generation — and only introduces new notation when unavoidable.

The document is structured as five layers, each building on the previous:

1. **Layer 1 — The Big Picture**: What a "Large Physics Model for Porous Media" actually is
2. **Layer 2 — Porous Media Physics**: The PDEs and macroscopic quantities you care about
3. **Layer 3 — Lattice Boltzmann Methods (LBM)**: Your GPU-native data factory
4. **Layer 4 — Fourier Neural Operators (FNO)**: Your neural surrogate
5. **Layer 5 — Inverse Design**: Closing the design loop

Throughout, the Li-ion battery anode is used as the running example to ground everything.

---

# LAYER 1: THE BIG PICTURE

## 1.1 What You Are Actually Building

Think of this as three interoperating systems:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     UNIVERSAL POROUS MEDIA ENGINE                       │
│                                                                         │
│  [3D Voxel Geometry]  ──→  [LBM GPU Solver]  ──→  [3D Physics Fields]   │
│         ↑                        (ground truth)             ↓           │
│  [Inverse Design]  ←──  [Fourier Neural Operator]  ←─────────           │
│  (generate geometry)       (learned surrogate, ~1000x faster)           │
└─────────────────────────────────────────────────────────────────────────┘
```

The three parts are:

- **Data Factory**: JAX-based LBM (JAX-LaB or XLB) running on IndiaAI GPU nodes, generating millions of (geometry, physics-fields) pairs
- **Forward LPM**: An FNO trained to map geometry → full 3D physics fields at any resolution
- **Inverse Design**: A generative model or gradient-based optimizer that maps desired properties → geometry

## 1.2 Why This Is Better Than What You Have

Your current PyBaMM pipeline does:

```
3D Voxel → TauFactor (scalars: τ, ε, SA) → PyBaMM DFN (1D, CPU) → Voltage curve
```

This new LPM pipeline does:

```
3D Voxel → LBM (full 3D: velocity/pressure/saturation fields, GPU) → FNO surrogate
```

The differences are fundamental:
| Dimension | PyBaMM/TauFactor | LBM + FNO (LPM) |
|---|---|---|
| Spatial resolution | 1D (x only, through electrode) | Full 3D field |
| Physics | Effective medium (homogenized) | Pore-scale, resolved |
| Multiphase | No (single-phase electrolyte assumed) | Yes (Shan-Chen) |
| Wettability | No | Yes (contact angle control) |
| Speed at inference | Fast (1D ODE/PDE) | ~1000x faster than LBM via FNO |
| Applications | Batteries only | Batteries, bio-scaffolds, catalysts |

## 1.3 The "Same Math" Claim

Your brief says "battery electrodes, bio-scaffolds, and catalysts are mathematically the same thing." Here is why this is literally true.

All three are instances of the problem: **fluid transport through a tortuous porous solid**. The governing equations in all three cases are:

- **Stokes flow**: ∇P = μ∇²u (slow viscous flow in pores)
- **Continuity**: ∇·u = 0
- **Convection-diffusion**: ∂c/∂t + u·∇c = D∇²c (species transport)
- **Darcy's Law** (upscaled): u = -(K/μ)∇P

The only things that change between applications are:

- The geometry (electrode particle packing vs. scaffold vs. pellet)
- The fluid phases (electrolyte vs. culture medium vs. reactant gas)
- The boundary reactions (intercalation vs. cell growth vs. catalysis)

This is why a single "universal porous media engine" trained on diverse geometries can serve all three.

---

# LAYER 2: POROUS MEDIA PHYSICS

## 2.1 What Is Porous Media?

A porous medium is any solid material with a connected network of voids (pores) through which fluids can flow. For your anode (graphite or silicon), it looks like this:

- ~20–40 μm thick electrode
- Spherical/irregular graphite particles (radius 3–10 μm)
- Pores filled with liquid electrolyte (LiPF₆ in EC/DMC)
- Porosity ε typically 0.25–0.35 (25–35% of volume is pore space)

The key geometric descriptors:

- **Porosity ε**: Volume fraction of pore space. ε = V_pore / V_total
- **Tortuosity τ**: How much longer the actual fluid path is vs. the straight-line path. τ = (L_actual/L_straight)². Always τ ≥ 1.
- **Specific surface area SA**: Area of solid-pore interface per unit volume [m²/m³]
- **Permeability K**: Resistance of the medium to fluid flow [m²]. It links pressure gradient to flow via Darcy's Law.

**The Kozeny-Carman correlation** (your baseline sanity check) gives a rough estimate:

```
K ≈ ε³ / (180 · (1-ε)² · SA²)       [Kozeny-Carman equation]
```

This is why 20% porosity microstructures are so challenging — at ε = 0.20, the K value drops ~10× compared to ε = 0.35.

## 2.2 From PyBaMM (1D) to LBM (3D): The Bridge

You already know the PyBaMM Doyle-Fuller-Newman (DFN) model. Let's connect it explicitly to what you're building.

### The DFN Model (What PyBaMM Solves)

The DFN treats the electrode as a 1D continuum with "effective" transport properties. The key equations are:

**Electrolyte concentration transport**:

```
ε · ∂c_e/∂t = ∂/∂x [D_e,eff · ∂c_e/∂x] + (1 - t₊) · j_n
```

**Electrolyte potential**:

```
∂/∂x [κ_eff · ∂φ_e/∂x] + ∂/∂x [κ_D,eff · ∂(ln c_e)/∂x] + j_n = 0
```

The key terms `D_e,eff` and `κ_eff` are the _effective diffusivity_ and _effective conductivity_, and they are related to microstructure by:

```
D_e,eff = ε · D_e / τ        (Bruggeman relation, commonly used approximation)
```

**This is the bridge**: τ (tortuosity) comes from your 3D microstructure. TauFactor computes τ from your voxel grid. That τ goes into PyBaMM. But PyBaMM never sees the actual pore structure — it only sees the aggregated effect through τ.

**What LBM gives you instead**: Rather than computing τ and plugging it in, LBM directly simulates the 3D electrolyte flow in the actual pore geometry. You get the full velocity field u(x,y,z), pressure field P(x,y,z), and concentration field c(x,y,z). You can then extract K, τ, D_eff as derived quantities — but you also have the full spatiotemporal data that PyBaMM throws away.

### Why This Matters for Batteries Specifically

The DFN model breaks down at:

- **Fast charge/discharge** (C-rate > 2C): Concentration gradients in pores become large, non-linear
- **Partial wetting**: Not all pores are filled with electrolyte at time zero. Shan-Chen LBM captures the capillary-driven infiltration process that determines _which_ pores are active
- **Tortuosity anisotropy**: τ is not a scalar — it's a tensor. The pore structure of a rolled electrode has different τ in the through-plane direction (what matters for Li⁺ transport) vs. in-plane
- **Dendrite/degradation**: Pore geometry changes with cycling. A true LPM can predict evolving transport as geometry changes

## 2.3 The Governing Equations Your LBM Will Solve

At the pore scale, fluid flow in your electrode is governed by the **incompressible Navier-Stokes equations**:

```
ρ(∂u/∂t + u·∇u) = -∇P + μ∇²u + F_body    [momentum]
∇·u = 0                                      [incompressibility]
```

For the slow creeping flow in electrode pores (very low Reynolds number, Re ≈ 10⁻³ to 10⁻²), the inertial terms disappear and this simplifies to **Stokes flow**:

```
-∇P + μ∇²u = 0
∇·u = 0
```

For two-phase flow (electrolyte + gas, or two immiscible solvents), you additionally need:

- **Capillary pressure**: P_c = P_non-wetting − P_wetting = 2σ·cos(θ)/r (Young-Laplace)
- **Wetting boundary conditions**: contact angle θ at solid-fluid interface

LBM solves all of this without explicitly tracking interfaces, which is the key advantage over VOF or level-set methods.

---

# LAYER 3: LATTICE BOLTZMANN METHODS

## 3.1 Why LBM, Not OpenFOAM / FEM?

From an ML perspective, think of LBM as a **mesh-free, locally parallel cellular automaton** that happens to recover the Navier-Stokes equations in the macroscopic limit. Compared to classical solvers:

| Property                | FEM/FVM (OpenFOAM)                         | LBM                                                        |
| ----------------------- | ------------------------------------------ | ---------------------------------------------------------- |
| Grid                    | Body-conforming, complex mesh generation   | Regular Cartesian grid, no meshing                         |
| Parallelism             | Domain decomposition, MPI, hard to GPU-ize | Trivially parallel: each node independent at each timestep |
| Multiphase interfaces   | Explicit tracking (VOF, LS)                | Implicit, no tracking needed                               |
| Complex geometry        | Needs re-meshing                           | Just mark solid nodes as bounceback                        |
| JAX/PyTorch integration | Very hard (C++ codebase)                   | Native (JAX-LaB, XLB are pure JAX)                         |
| Differentiability       | Not differentiable                         | Fully differentiable via JAX autograd                      |

The differentiability is critical for your Phase 3 inverse design work.

## 3.2 LBM Core Concept: Particles on a Lattice

LBM is based on the **Boltzmann kinetic theory** of gases, but discretized. Instead of tracking individual fluid molecules, it tracks a **probability distribution function** f_i(x, t): the probability of finding a fluid "particle" at position x, at time t, moving in direction i.

Think of it like this: at each grid cell, you store not just a single velocity vector (as in Navier-Stokes) but a _histogram of velocities_ — how much fluid "population" is moving in each of the 19 discrete directions.

### 3.3 The D3Q19 Lattice

**D3Q19** means: **3** spatial **D**imensions, **19** velocity directions **Q**.

The 19 directions are: 1 rest (zero velocity) + 6 face-connected + 12 edge-connected neighbors. (D3Q27 adds 8 corner-connected, but costs more; D3Q15 is cheaper but less isotropic.)

Why D3Q19 for porous media?

- **More isotropic than D3Q15**: Better resolves pressure isotropy and flow in arbitrary directions through tortuous pores
- **Less expensive than D3Q27**: 19 vs 27 distributions stored per node
- **Proven stability**: Decades of validation in porous media literature
- **Supported by both JAX-LaB and XLB**: Direct implementation available

Each node stores a vector f = [f₀, f₁, ..., f₁₈] (19 floats). The lattice speed c = Δx/Δt is normalized to 1. The speed of sound c_s = 1/√3.

## 3.4 The LBM Algorithm: Two Steps per Timestep

The full LBM update is elegantly simple — just two steps:

### Step 1: Collision (Local, fully parallel)

At each node, independently update the distribution functions based on how far they are from equilibrium:

```python
# BGK (single relaxation time) collision:
f_i_star(x, t) = f_i(x, t) - (1/τ) * [f_i(x, t) - f_i_eq(x, t)]
```

Where f_i_eq is the **Maxwell-Boltzmann equilibrium distribution**:

```
f_i_eq(ρ, u) = w_i · ρ · [1 + (e_i·u)/c_s² + (e_i·u)²/(2c_s⁴) - u²/(2c_s²)]
```

Here w_i are lattice weights (e.g., for D3Q19: w=1/3 for rest, w=1/18 for face, w=1/36 for edge), and e_i are the 19 discrete velocity vectors.

### Step 2: Streaming (Non-local, but trivially parallel)

Move each population to its neighboring node in its velocity direction:

```python
# Streaming:
f_i(x + e_i·Δt, t + Δt) = f_i_star(x, t)
```

### Macroscopic Recovery

Density and momentum are zero-th and first moments of f:

```
ρ(x,t) = Σ_i f_i(x,t)
ρ·u(x,t) = Σ_i f_i(x,t) · e_i
```

Via Chapman-Enskog multi-scale analysis, one can prove that this LBE recovers the incompressible Navier-Stokes equations in the low-Mach-number limit, with kinematic viscosity:

```
ν = c_s² · (τ - 0.5) · Δt
```

This is the single most important equation for LBM users: **viscosity is controlled by τ**. τ > 0.5 for stability, and practically τ ∈ [0.5, 2].

## 3.5 Why BGK Fails and MRT Saves You

### The BGK Problem

BGK (Bhatnagar-Gross-Krook) uses a **single relaxation time τ** for all 19 moments of f. This is the simplest possible collision model, but it has a fatal flaw for porous media: **numerical instability in narrow constrictions and at low porosities**.

Physically, in a pore with a narrow throat (common in 20% porosity microstructures), the local Reynolds number spikes momentarily as fluid accelerates through the bottleneck. BGK, relaxing all moments at the same rate, cannot dissipate the resulting numerical artifacts, and the simulation **diverges**.

The root cause: BGK couples the _shear_ modes (physically meaningful, controlled by viscosity) with _energy_ and _ghost_ modes (numerical artifacts). When all are relaxed at the same rate, you can't independently tune away the artifacts without messing up the physics.

### The MRT Solution

MRT (Multi-Relaxation Time) transforms the 19 distribution functions into **moment space** using a transformation matrix M:

```
m = M · f        (transform to moment space)
```

The 19 moments have physical meaning:

- m₀ = ρ (density)
- m₁, m₂, m₃ = j_x, j_y, j_z (momentum)
- m₄ = energy
- m₅...m₈ = energy flux (non-conserved)
- m₉...m₁₄ = stress tensor components
- m₁₅...m₁₈ = ghost/anti-symmetric modes

MRT relaxes **each moment independently** with its own relaxation rate s_i:

```
# MRT collision:
f_i_star = f_i - M⁻¹ · S · M · (f_i - f_i_eq)
```

Where S = diag(s₀, s₁, ..., s₁₈) is the diagonal relaxation matrix.

The key freedom: you fix s_ν (shear viscosity rate) to match your target viscosity, then freely tune s_bulk, s_energy, s_ghost to **maximize numerical stability** without touching the physical viscosity. This is why MRT survives the extreme constrictions in your 20% porosity structures.

**In code (JAX-LaB convention)**:

```python
# JAX-LaB MRT setup
S = jnp.diag(jnp.array([
    s_rho,     # density (conserved, set = 1)
    s_e,       # energy
    s_eps,     # energy square
    s_rho,     # jx (conserved)
    s_q,       # energy flux x
    s_rho,     # jy (conserved)
    s_q,       # energy flux y
    s_rho,     # jz (conserved)
    s_q,       # energy flux z
    1.0/tau,   # Pxx (shear viscosity!)
    s_pi,      # Pi_xx - Pi_yy
    1.0/tau,   # Pxy
    1.0/tau,   # Pxz
    1.0/tau,   # Pyz
    s_pi,      # Pi_xx + Pi_yy - 2*Pi_zz
    s_t,       # Qxxx
    s_t,       # Qyyy
    s_t,       # Qzzz
    s_om       # omega (anti-symmetric)
]))
```

The shear relaxation rate (=1/τ) controls viscosity. All other s values are tuned for stability, typically set to 1.0–1.8 for robustness.

## 3.6 Multiphase LBM: The Shan-Chen Model

For battery electrolyte infiltration and bio-reactor filling, you need to model **two immiscible fluids** (e.g., electrolyte liquid + air) competing for pore space. This is where the **Shan-Chen pseudopotential method** comes in.

### Key Idea: Interaction Force

Shan-Chen adds an **inter-particle interaction force** between fluid components. For component k interacting with component k':

```
F_ff^(k,k') = -A_{kk'} ∇U^(k') - (1-A_{kk'}) g_{kk'} ψ^k(x) ∇ψ^(k')(x)
```

The key object is the **pseudopotential** ψ^k, which is a function of local density:

```
ψ^k(ρ) = sqrt( 2(α^k · P_EOS^k - c_s² · ρ^k) / g_{kk} )
```

Here P_EOS is the pressure from a thermodynamic equation of state (Carnahan-Starling, Peng-Robinson, etc.). The EOS encodes the fluid's phase behavior — at what densities it separates into liquid vs. vapor phases.

### What This Gives You Physically

1. **Phase separation**: If two fluid elements have incompatible densities, the interaction force pushes them apart. This is how a droplet forms without explicit interface tracking.
2. **Surface tension**: The interaction force creates a pressure jump across the interface, quantified by Laplace's law: ΔP = σ/R.
3. **Contact angle (wettability)**: By setting virtual densities at solid nodes, you control whether the fluid prefers to wet (θ < 90°) or dewet (θ > 90°) the solid. For a hydrophilic battery electrode, θ_electrolyte < 90°.

### The Improved Virtual Density Scheme (JAX-LaB)

Traditional Shan-Chen wettability used a single uniform "solid density" to control contact angle, but this breaks down at high density ratios and produces non-physical films. JAX-LaB implements the **improved virtual density scheme** (Li et al. 2019):

```
ρ_s = φ · ρ_ave(x_s)     if φ ≥ 1  (θ ≤ 90°, wetting)
ρ_s = ρ_ave(x_s) - Δρ   if Δρ ≥ 0  (θ > 90°, non-wetting)
```

Where ρ_ave is a weighted average of neighboring fluid densities. By setting φ and Δρ spatially, you can specify **per-surface wettability** — essential for heterogeneous battery electrodes where carbon black and active material particles have different contact angles.

JAX-LaB achieves this while maintaining contact angles from 5° to 170°, density ratios >10⁷, and spurious currents <2×10⁻³. That last point matters because spurious currents (unphysical velocities at interfaces) corrupt your transport measurements.

## 3.7 What Your LBM Simulation Produces (The Data)

For a single 3D porous geometry (say, 256³ voxels of your graphite anode), one LBM run produces:

**Steady-state single-phase flow** (one run per geometry):

- Velocity field u(x,y,z): 3 × N³ floats
- Pressure field P(x,y,z): 1 × N³ floats
- Derived scalar: permeability tensor K_xx, K_yy, K_zz [from K = -μ·(Q·L)/(A·ΔP)]
- Derived scalar: tortuosity τ_x, τ_y, τ_z

**Two-phase imbibition run** (time-series):

- Saturation field S(x,y,z,t): liquid volume fraction at each voxel, over time
- Capillary pressure P_c(S) curve
- Relative permeability k_r,wetting(S), k_r,non-wetting(S) curves

This data is the **ground truth** your FNO will learn to approximate.

## 3.8 JAX-LaB vs. XLB: What to Know

Both are JAX-based, both scale to multi-GPU. The key differences:

| Feature               | JAX-LaB                                          | XLB                                    |
| --------------------- | ------------------------------------------------ | -------------------------------------- |
| Primary use case      | Multiphase porous media (geoscience focus)       | General-purpose, physics-based ML      |
| Multiphase            | Native Shan-Chen with improved EOS + wettability | No native multiphase (single-phase)    |
| Collision models      | BGK, MRT, Cascaded (CLBM)                        | BGK, MRT, KBC, TNSE                    |
| Backends              | JAX (CPU/GPU/TPU)                                | JAX + NVIDIA Warp                      |
| Multi-GPU sharding    | pjit along x-axis                                | pjit along x-axis                      |
| Performance           | Giga-LUPS on multi-GPU                           | Giga-LUPS, scales to billions of cells |
| Python-ML integration | Native (JAX pytrees)                             | Native (JAX arrays)                    |

**For your project**: JAX-LaB is the right solver for Phase 2 (data factory). It has everything you need — MRT, D3Q19, Shan-Chen with wettability, porous-media benchmarks, and JAX-native multi-GPU scaling. XLB is worth benchmarking for single-phase permeability runs because of its Warp backend performance.

### The pjit Multi-GPU Setup

Both solvers shard along the x-axis. For a 512³ simulation on 8 GPUs:

```python
import jax
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec as P

# Data layout: (Nx, Ny, Nz, Q) = (512, 512, 512, 19)
# Shard: split Nx across 8 GPUs → each GPU gets (64, 512, 512, 19)
devices = jax.devices()  # 8 GPUs
mesh = Mesh(devices, axis_names=('x',))

# Ensure Nx % n_gpus == 0 (512 / 8 = 64 ✓)
# JAX-LaB handles ghost-cell communication across shards automatically
```

The OOM micro-benchmark you need to run: sweep (Nx, Ny, Nz) from 64³ to 512³ on a single 8-GPU node, watch per-GPU memory usage (target < 80% of 80 GB for H100).

---

# LAYER 4: FOURIER NEURAL OPERATORS (FNO)

## 4.1 Why Standard Neural Networks Are Wrong for PDEs

From your ML background: you know a standard CNN trained on 64×64 images won't automatically generalize to 128×128 images — it learns finite-dimensional mappings. For PDE surrogates, this is disastrous:

- Your training geometries might be 64³ voxels
- Your inference-time geometry might be 128³
- You want the same model to work on both

More fundamentally: a PDE solution u(x) is a **function**, not a vector. The "correct" surrogate should learn a map between infinite-dimensional function spaces.

## 4.2 Neural Operators: The Right Abstraction

A **neural operator** G_θ learns a map between infinite-dimensional function spaces:

```
G_θ: A → U
```

Where A is a space of input functions (e.g., permeability fields a(x)) and U is the space of output functions (e.g., pressure fields u(x)).

The key property: **discretization invariance**. G_θ is parameterized on function spaces, not on specific grids. When you discretize a(x) to train on 64³ points, the learned operator automatically applies at 128³, 256³, or any other resolution.

This is proved rigorously: neural operators are **universal approximators** of continuous operators between Banach spaces (Kovachki et al. 2023, JMLR).

## 4.3 The FNO Architecture

The FNO (Li et al. 2021) parameterizes the integral kernel (the core operation in a neural operator) **in Fourier space**.

### Formal Definition

An FNO layer maps: v*t(x) → v*{t+1}(x) via:

```
v_{t+1}(x) = σ( W · v_t(x)  +  (K · v_t)(x) )
               └─────────┘       └───────────┘
               local linear    non-local convolution
               (like residual)  (like attention — global)
```

The non-local part is a **Fourier-space multiplication**:

```
(K · v)(x) = F⁻¹[ R_φ(k) · F[v](k) ]
```

Where:

- F[·] is the Discrete Fourier Transform (FFT)
- R_φ(k) is a learnable weight tensor in frequency space (only low-frequency modes k ≤ k_max are kept)
- F⁻¹[·] is the inverse FFT

### Why Fourier Space?

For PDEs, low-frequency modes (global structure) carry most of the physics. High frequencies (fine details) are relatively unimportant and expensive to compute. By learning in Fourier space:

- The model captures **long-range correlations** (which convolution can't) at the cost of a single FFT
- Frequency truncation at k_max acts as an implicit regularizer
- Resolution invariance follows naturally — FFT/iFFT can be computed at any grid size

### FNO for Darcy/Porous Media Flow

For your battery electrode case, the forward FNO maps:

```
a(x) [permeability field or binary voxel mask]  →  u(x) [pressure/velocity field]
```

Input a(x) is essentially your 3D microstructure — either:

- Binary: 1 at pore nodes, 0 at solid nodes
- Permeability: K(x) = K_fluid at pore, K_solid ≈ 0 at solid

Output u(x) is the full 3D pressure (and optionally velocity) field, from which you can derive K, τ, relative perms as scalars.

The NVIDIA Modulus Darcy tutorial does exactly this in 2D. Your LPM does it in 3D with multiphase outputs.

### Complete FNO Architecture (What You'll Code)

```
Input: a(x) ∈ R^(Nx × Ny × Nz × 1)  [voxel geometry]

Step 1: Lift to higher-dimensional feature space
  v₀(x) = P · a(x)     [learnable linear, R¹ → R^d_v]
  v₀ ∈ R^(Nx × Ny × Nz × d_v)

Step 2: L FNO layers
  For l = 0 to L-1:
    v_{l+1}(x) = σ( W_l · v_l(x) + F⁻¹[R_l · F[v_l]] )

    where:
      W_l: pointwise linear (d_v × d_v weight matrix)
      R_l: complex weight tensor in Fourier space (k_max³ × d_v × d_v)
      σ: GELU or ReLU activation

Step 3: Project to output
  u(x) = Q · v_L(x)     [learnable linear, R^d_v → R^c_out]
  u ∈ R^(Nx × Ny × Nz × c_out)  [pressure, velocity, saturation...]

Parameters:
  d_v: width (channels), typically 32-64 for 3D
  L: depth (layers), typically 4
  k_max: Fourier mode cutoff, typically 8-16 for 3D
  c_out: output channels (1 for pressure, 3 for velocity, etc.)
```

### Losses and Metrics

Train with relative L2 loss on field predictions:

```
L = ‖u_pred - u_true‖_L2 / ‖u_true‖_L2
```

Also track:

- **Permeability relative error**: |K_pred - K_true| / K_true (derived from predicted pressure gradient)
- **Velocity RMSE**: per-component velocity field error
- **Saturation MAE**: for multiphase outputs

Empirical targets from the literature: <5% relative error on permeability, <10% on full velocity fields.

## 4.4 Resolution Invariance: What This Means in Practice

This is the killer feature. When your FNO is trained on 64³ geometries:

- The FFT is computed on 64³ points, keeping modes up to k_max
- R_l is learned in Fourier space (shape k_max³ × d_v × d_v)

At inference time with 128³ geometry:

- FFT is computed on 128³ points
- Low-frequency modes (k ≤ k_max) are extracted — same shape as training
- R_l is applied — same parameters as training
- iFFT is computed on 128³ points

The weights R_l are identical. No re-training. This is zero-shot generalization to higher resolution, which is fundamentally impossible with standard CNNs.

## 4.5 Scaling to a Large Physics Model

An "LPM" is an FNO trained on a **massively diverse dataset**:

- Thousands of different geometry classes (sphere packs, reconstructed SEM images, synthetic fractal structures, anode architectures, bio-scaffold designs, catalyst pellets)
- Multiple porosity levels (ε = 0.15 to 0.60)
- Multiple fluid pairs (electrolyte/gas, water/oil, reactant/product)
- Multiple flow regimes (single-phase Darcy, two-phase drainage, imbibition)

A model trained on all of this simultaneously will develop "foundational" representations of pore-scale physics, analogous to how LLMs develop language representations from diverse text.

The architecture scales up by increasing: d_v (width), L (depth), k_max (Fourier modes). A large 3D LPM might have d_v=128, L=6, k_max=16.

---

# LAYER 5: INVERSE DESIGN

## 5.1 The Inverse Problem Statement

**Forward problem**: Given geometry a(x), predict physics u(x)

```
G_θ: a(x) → u(x)     [FNO, learned]
```

**Inverse problem**: Given desired physics or macroscopic properties p* (e.g., target permeability K*, target relative perm curve), find geometry a(x) that achieves it.

```
Find a(x) such that: Φ(G_θ(a(x))) ≈ p*
```

Where Φ is a functional that extracts macroscopic properties from fields.

This is used to: design the optimal anode microstructure for fast charging, design a bio-scaffold with target nutrient transport, design a catalyst support with optimal pressure drop vs. surface area.

## 5.2 Approach 1: Gradient-Based Optimization through FNO

Since FNO is differentiable (built on JAX/PyTorch), you can compute ∂L/∂a — the gradient of your physics objective with respect to the geometry.

```
Algorithm: Gradient-based inverse design
-----------------------------------------
1. Initialize: a(x) ← continuous relaxation of binary geometry
2. Forward pass: u(x) = G_θ(a(x))
3. Compute loss: L = ‖Φ(u(x)) - p*‖²
4. Backward pass: ∂L/∂a = ∂L/∂u · ∂G_θ/∂a   [via autograd]
5. Update: a ← a - η · ∂L/∂a
6. Project: a ← threshold(a) to binary [0,1]
7. Repeat until convergence
```

This is essentially **topology optimization** using a neural surrogate. Since G_θ is ~1000× faster than LBM, you can afford thousands of optimization steps.

**Practical challenge**: The binary constraint (a ∈ {0,1}) makes this non-convex. Techniques: SIMP relaxation (a ∈ [0,1] with penalization), stochastic binary projections, or annealed thresholding.

## 5.3 Approach 2: Generative Model for Inverse Design

Train a **generative model** conditioned on desired properties p\* to directly sample from the distribution of geometries that achieve those properties.

```
P(a | p*) ∝ P(p* | a) · P(a)
```

Current state-of-the-art approaches:

- **3D-GAN + RL**: Train a GAN to generate realistic microstructures. Use actor-critic RL to control the GAN's latent space, navigating it to achieve target properties (permeability, porosity, surface area). This is essentially what the GAN-AC paper you can reference does for sandstone.
- **Diffusion Model for Porous Materials**: A 2024 RSC paper (d3ta06274k) shows diffusion models outperform GANs by >2000× on structure validity for inverse design, generating novel porous materials with target void fraction and transport properties.
- **Latent FNO + Variational Autoencoder**: Encode geometry into latent space z with VAE, train FNO in latent space, then invert using conditional decoder.

For your project, the recommended first step is the **gradient-based approach** (simpler, directly uses your FNO), with the generative model as a later enhancement for Phase 3.

---

# LAYER 6: SYSTEM INTEGRATION — FROM PyBaMM TO LPM

## 6.1 The Full Stack, Grounded in Li-ion Anode

Let's walk through the complete pipeline using your graphite anode as the concrete example.

### Step 0: What You Already Have

```
[Microstructure Generator] → [3D Voxel Grid: 256³, graphite anode, ε ≈ 0.30]
                                  ↓
                          [TauFactor → τ, ε, SA]
                                  ↓
                          [PyBaMM DFN → Voltage(t)]
```

### Step 1 (Phase 1 — Solver Setup): Configure JAX-LaB for Your Geometry

```python
import jax
import jax.numpy as jnp
from jaxlab import LBMSolver, MRTCollision, D3Q19, ShanChen

# Load your existing voxel grid (from your microstructure generator)
voxel_grid = load_voxel_grid("graphite_anode_256.npy")  # shape (256, 256, 256)
# 1 = pore (fluid), 0 = solid (graphite particle)

# Configure D3Q19 MRT solver
solver = LBMSolver(
    lattice=D3Q19(),
    collision=MRTCollision(
        tau=0.7,           # ν = c_s² * (τ - 0.5) = 0.0667, physical viscosity
        # Stability-optimized relaxation rates for 20% porosity:
        s_e=1.2,           # energy mode
        s_eps=1.0,         # energy-square mode
        s_q=1.4,           # energy-flux modes
        s_nu=1.0/tau,      # MUST match shear viscosity
        s_pi=1.8,          # stress modes (tune for stability)
    ),
    multiphase=ShanChen(   # For electrolyte infiltration
        eos='carnahan_starling',
        contact_angle=30.0,  # degrees, electrolyte wets graphite
        surface_tension=0.01
    )
)

# Domain setup
solver.set_geometry(voxel_grid)
solver.set_boundary('inlet', axis='z', type='pressure', P=1.001)
solver.set_boundary('outlet', axis='z', type='pressure', P=1.000)
```

### Step 2 (Phase 2 — Data Factory): Generate Ground Truth Dataset

```python
# Vary geometry inputs from your microstructure generator:
geometry_params = {
    'porosity': np.linspace(0.20, 0.45, 20),
    'particle_radius_mean': [3.0, 5.0, 7.0, 10.0],  # μm
    'particle_shape': ['sphere', 'ellipsoid', 'irregular'],
    'particle_size_std': [0.5, 1.5, 3.0]  # polydispersity
}

# For each geometry: run LBM, extract fields
dataset = []
for params in geometry_param_sweep(geometry_params, n=50000):
    voxel = microstructure_generator(**params)          # Your existing pipeline
    fields = solver.run(voxel, n_steps=10000)           # LBM steady state
    K = compute_permeability(fields, params)            # Darcy K tensor
    tau = compute_tortuosity(fields)                    # geometric tortuosity
    rel_perm = solver.run_drainage(voxel, n_steps=50000)  # two-phase
    dataset.append({
        'geometry': voxel,          # input to FNO
        'pressure': fields.P,       # output
        'velocity': fields.u,       # output
        'saturation': fields.S_t,   # output (two-phase)
        'K': K,                     # scalar label
        'tau': tau                  # scalar label
    })
```

### Step 3 (Phase 3 — LPM Training): Train the Forward FNO

```python
import torch
from neuralop.models import FNO3d

model = FNO3d(
    n_modes=(16, 16, 16),   # k_max
    hidden_channels=64,      # d_v
    n_layers=4,              # L
    in_channels=1,           # binary voxel
    out_channels=4,          # (P, u_x, u_y, u_z)
)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for epoch in range(500):
    for batch in dataloader:
        pred = model(batch['geometry'])
        loss = relative_L2_loss(pred, batch['fields'])
        loss.backward()
        optimizer.step()
```

### Step 4: Bridging Back to PyBaMM (or Replacing It)

Once you have the FNO trained:

**Option A: Use FNO to compute effective parameters for PyBaMM**

```python
# Given a new geometry:
voxel = new_anode_design()
fields = fno_model(voxel)               # microseconds inference
K, tau, D_eff = compute_effective_params(fields)

# Feed into PyBaMM DFN as before:
param["Negative electrode tortuosity factor [1]"] = tau
param["Negative electrode permeability [m2]"] = K
voltage = pybamm_dfn.simulate(param)
```

**Option B: Replace PyBaMM entirely with full 3D LPM**

- Train FNO outputs to include concentration c_e(x,y,z,t) and reaction rate j_n(x,y,z,t)
- Integrate directly over volume to get cell-level voltage
- No need for τ/K effective parameters — the 3D physics is directly captured

Option B is your ultimate goal (the true LPM). Option A is a good intermediate milestone that validates your FNO using your existing PyBaMM validation framework.

---

# APPENDIX: KEY EQUATIONS REFERENCE

## A.1 LBM Core Equations

**LBE (Lattice Boltzmann Equation)**:

```
f_i(x + e_i·Δt, t + Δt) = f_i(x,t) + Ω_i(x,t)
```

**Equilibrium distribution (D3Q19)**:

```
f_i_eq = w_i · ρ · [1 + (e_i·u)/c_s² + (e_i·u)²/(2c_s⁴) - u²/(2c_s²)]

w_0 = 1/3  (rest)
w_{1-6} = 1/18  (face neighbors)
w_{7-18} = 1/36  (edge neighbors)
c_s = 1/√3
```

**BGK collision**:

```
Ω_i^BGK = -(1/τ)(f_i - f_i_eq)
```

**MRT collision**:

```
Ω_i^MRT = -M⁻¹ · S · M · (f_i - f_i_eq)
```

**Viscosity-relaxation relationship**:

```
ν = c_s² · (τ - 0.5) · Δt
```

**Permeability (Darcy)**:

```
K = -μ · (Q · L) / (A · ΔP)
Q = volume flow rate [m³/s]
L = domain length [m]
A = cross-sectional area [m²]
ΔP = applied pressure difference
```

## A.2 FNO Key Equations

**Neural Operator layer**:

```
v_{t+1}(x) = σ( W_v_t(x) + F⁻¹[R · F[v_t]](x) )
```

**Fourier truncation**:

```
F[v](k) for k ∈ {0, 1, ..., k_max-1}^3  (retain low modes only)
```

**Relative L2 loss**:

```
L_rel = ‖u_pred - u_true‖₂ / ‖u_true‖₂
```

## A.3 Shan-Chen Key Equations

**Pseudopotential with EOS**:

```
ψ^k(ρ) = sqrt( 2(α^k · P_EOS^k - c_s² · ρ^k) / g_{kk} )
```

**Inter-component force**:

```
F_ff^(k,k') = -A_{kk'}∇U^(k') - (1-A_{kk'})g_{kk'}ψ^k ∇ψ^(k')
```

**Young-Laplace (capillary pressure)**:

```
ΔP = 2σ·cos(θ)/r  (capillary tube)
ΔP = σ/R  (droplet, 3D: Laplace)
```

---

# APPENDIX B: YOUR IMMEDIATE NEXT STEPS

## B.1 Before the Meeting: Key Concepts to Be Able to Explain

1. **Why D3Q19 + MRT?** D3Q19 is 3D with 19 velocity directions — more isotropic than D3Q15, cheaper than D3Q27. MRT independently relaxes each of the 19 velocity moments, enabling stability in low-porosity constrictions where BGK diverges.

2. **Why JAX-LaB over XLB for Phase 2?** JAX-LaB extends XLB with native Shan-Chen multiphase, improved virtual-density wettability, and 5 EOS implementations — all needed for battery electrolyte infiltration and bio-reactor wetting.

3. **Why FNO?** Unlike CNN surrogates, FNO is discretization-invariant (same model at any resolution), handles long-range correlations via Fourier-space kernels, and has proven ~1000× speedup over classical PDE solvers.

4. **How does LPM extend beyond batteries?** The governing equations (Stokes + convection-diffusion in porous media) are identical for batteries, bio-scaffolds, and catalysts. Only boundary conditions and EOS change.

5. **Inverse design approach**: Gradient-based optimization through differentiable FNO + JAX-LaB in Phase 3.

## B.2 The Micro-Benchmark You Need to Run

Before committing to JAX-LaB, run this benchmark on a single multi-GPU node:

```python
# Benchmark script: JAX-LaB pjit scaling
import jax
import time

sizes = [64, 128, 256, 512]
n_gpus = len(jax.devices())

for N in sizes:
    voxel = generate_test_geometry(N, porosity=0.20)  # worst case

    t0 = time.time()
    result = jax_lab_solver.run(voxel, collision='MRT', n_steps=1000)
    t_per_step = (time.time() - t0) / 1000

    mlups = (N**3 / t_per_step) / 1e6  # Million Lattice Updates Per Second
    mem = measure_peak_gpu_memory()

    print(f"N={N}, GPUs={n_gpus}: {mlups:.1f} MLUPS, {mem:.1f} GB/GPU")

# Target: no OOM at 256³ on 8×H100. Prefer >100 MLUPS for practical data generation.
```

Check: does the sharding (nx/n_gpus must be integer) work? Any inter-GPU communication bottleneck (should be minimal with nearest-neighbor-only stencil)?
