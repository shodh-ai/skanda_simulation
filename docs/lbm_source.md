# Pore-Scale LBM and JAX Foundations for Li-Ion Porous Electrode Simulation

## 1. Scope and Modeling Goals

This report compiles the mathematical equations, lattice Boltzmann method (LBM) literature, and JAX implementation patterns relevant to building a pore-scale, LBM-based simulator for lithium-ion battery porous electrodes, without relying on PyBaMM or Fourier Neural Operators (FNOs). The focus is on:

- Single- and multiphase LBM for Stokes flow and advection–diffusion in complex 3D porous media.
- Pore-scale electrochemical transport and reaction in Li-ion electrodes using LBM.
- JAX primitives and existing JAX-based LBM/CFD libraries for high-performance, differentiable implementations.

The intent is to ground each modeling choice in published work so that the implementation can be justified in terms of established theory.

---

## 2. Core Governing Equations

### 2.1 Porous Media and Darcy-Scale Relations

At the Darcy scale, single-phase flow in porous media is often modeled with Darcy’s law:

\[ \mathbf{q} = -\frac{K}{\mu} \nabla p, \quad \mathbf{q} = \varepsilon \mathbf{u}, \]

where \(K\) is permeability, \(\mu\) dynamic viscosity, \(\varepsilon\) porosity, and \(\mathbf{u}\) is the intrinsic phase-average velocity.[1][2]

The Kozeny–Carman relation provides an empirical link between permeability, porosity, and specific surface area \(S_A\):

\[
K \approx \frac{\varepsilon^3}{C_K (1-\varepsilon)^2 S_A^2},
\]

where \(C_K \approx 180\) for packed spheres.[1]

Effective transport in porous electrodes is often parameterized by porosity and tortuosity \(\tau\). A common Bruggeman-type relation is:

\[
D*{\mathrm{e,eff}} = \frac{\varepsilon}{\tau} D_e, \quad \kappa*{\mathrm{eff}} = \frac{\varepsilon}{\tau} \kappa,
\]

with \(\tau\) obtained either from geometric models or direct pore-scale simulations.[3][4]

### 2.2 Navier–Stokes and Stokes Flow

At the pore scale, incompressible fluid flow obeys the Navier–Stokes equations:

\[
\rho \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u}\cdot\nabla \mathbf{u} \right) = -\nabla p + \mu \nabla^2 \mathbf{u} + \mathbf{F},
\]
\[
\nabla \cdot \mathbf{u} = 0.
\]

In low-Reynolds-number porous media (creeping flow), inertial terms are negligible, yielding Stokes flow:

\[
-\nabla p + \mu \nabla^2 \mathbf{u} + \mathbf{F} = 0, \quad \nabla \cdot \mathbf{u} = 0.
\]

### 2.3 Multiphase and Capillarity

For two immiscible fluids (e.g., electrolyte and gas), capillary pressure is typically modeled by Young–Laplace:

\[
P*c = P*{\mathrm{nonwet}} - P\_{\mathrm{wet}} = \frac{2 \sigma \cos\theta}{r},
\]

with surface tension \(\sigma\), contact angle \(\theta\), and characteristic pore radius \(r\).[5][6]

Wetting in porous electrodes is strongly influenced by contact angle and pore geometry; pore-scale LBM models directly resolve filling dynamics and trapped gas distributions, which in turn affect electrochemical performance.[7][8][9]

### 2.4 Electrochemical Transport and Kinetics

Pore-scale Li-ion models typically couple ionic transport in the electrolyte, electronic transport in the solid, and interfacial reactions at the active-material/electrolyte interface.[10][11]

A common continuum description uses Nernst–Planck-type equations for ionic species in the electrolyte:

\[
\frac{\partial c_i}{\partial t} + \nabla \cdot \mathbf{N}\_i = R_i,
\]
\[
\mathbf{N}\_i = -D_i \nabla c_i - z_i u_i c_i \nabla \phi_e + c_i \mathbf{u},
\]

where \(c_i\) is concentration, \(D_i\) diffusion coefficient, \(z_i\) charge number, \(u_i\) mobility, \(\phi_e\) electrolyte potential, and \(\mathbf{u}\) fluid velocity.[11][12]

Charge conservation in the solid and electrolyte phases leads to equations of the form:

\[
\nabla \cdot (\sigma*s \nabla \phi_s) = a j_n,
\]
\[
\nabla \cdot (\kappa*{\mathrm{eff}} \nabla \phi*e + 2 R T (1-t*+) F^{-1} \kappa\_{\mathrm{eff}} \nabla \ln c_e) = -a j_n,
\]

where \(\sigma*s\) is solid conductivity, \(\kappa*{\mathrm{eff}}\) effective electrolyte conductivity, \(a\) interfacial area per volume, and \(j_n\) interfacial current density.[12][3]

Interfacial kinetics are typically described by Butler–Volmer:

\[
j_n = j_0 \left[ \exp\left(\frac{\alpha_a F \eta}{R T}\right) - \exp\left(-\frac{\alpha_c F \eta}{R T}\right) \right],
\]

with \(j_0\) exchange current density, \(\eta\) overpotential, transfer coefficients \(\alpha_a, \alpha_c\), Faraday’s constant \(F\), gas constant \(R\), and temperature \(T\).[10][11]

These equations are implemented in different ways in pore-scale LBM frameworks, but the physical content is consistent across the literature.[13][14][10]

---

## 3. Lattice Boltzmann Method: Core Formulations

### 3.1 Basic Lattice Boltzmann Equation (BGK)

The single-relaxation-time (BGK) lattice Boltzmann equation for the density distribution function \(f_i(\mathbf{x}, t)\) in direction \(i\) is:

\[
f_i(\mathbf{x} + \mathbf{e}\_i \Delta t, t + \Delta t) = f_i(\mathbf{x}, t) - \omega \left(f_i(\mathbf{x}, t) - f_i^{\mathrm{eq}}(\mathbf{x}, t)\right) + F_i \Delta t,
\]

where \(\mathbf{e}\_i\) are discrete velocities, \(\omega = 1/\tau\) is the relaxation rate, \(f_i^{\mathrm{eq}}\) is the local equilibrium, and \(F_i\) lattice forcing term.[15][16]

On a D3Q19 lattice, there are 19 velocity directions: one rest, six face-centered, and twelve edge-centered directions. The equilibrium distribution for incompressible isothermal flow is typically[17][15]

\[
f_i^{\mathrm{eq}}(\rho, \mathbf{u}) = w_i \rho \left[1 + \frac{\mathbf{e}_i \cdot \mathbf{u}}{c_s^2} + \frac{(\mathbf{e}_i \cdot \mathbf{u})^2}{2 c_s^4} - \frac{\mathbf{u}^2}{2 c_s^2} \right],
\]

where \(w_i\) are lattice weights and \(c_s = 1/\sqrt{3}\) is the lattice speed of sound.[16][15]

Macroscopic variables are recovered as

\[
\rho = \sum_i f_i, \quad \rho \mathbf{u} = \sum_i f_i \mathbf{e}\_i + \frac{\Delta t}{2} \mathbf{F}.
\]

The kinematic viscosity is related to \(\tau\) by

\[
\nu = c_s^2 (\tau - 1/2) \Delta t.
\]

### 3.2 Multiple-Relaxation-Time (MRT) LBM

The MRT LBM improves numerical stability by relaxing different moments at different rates.[17][16]

Define the moment vector \(\mathbf{m} = M \mathbf{f}\), where \(M\) is a transformation matrix from distribution space to moment space. Collision in moment space is

\[
\mathbf{m}' = \mathbf{m} - S (\mathbf{m} - \mathbf{m}^{\mathrm{eq}}),
\]

where \(S = \mathrm{diag}(s*0, s_1, \dots, s*{18})\) is a diagonal relaxation matrix, with specific entries tied to conserved (density, momentum) and non-conserved (energy, shear, ghost) modes.[18][16]

Back in distribution space:

\[
\mathbf{f}' = M^{-1} \mathbf{m}'.
\]

D’Humières provides detailed construction of the D3Q19 MRT model and guidelines for choosing relaxation rates, including separating the shear viscosity rate from bulk and higher-order modes. For porous media, more recent evaluations show that appropriate choices of a small set of free parameters can yield permeability estimates that are independent of \(\tau\) and robust in complex micro-tomographic pore spaces.[19][18][16][17]

### 3.3 Forcing Schemes and Boundary Conditions

Body forces (e.g., pressure gradients) can be implemented using Guo forcing or equivalent schemes, which add source terms to the collision operator in a way that maintains second-order accuracy.[20]

For porous media flows, the typical boundary conditions are:

- No-slip walls via halfway bounce-back at solid nodes.[18][15]
- Pressure or velocity boundaries at inlets/outlets (e.g., Zou–He or its MRT variants).[20]

Recent work on D3Q19 MRT in micro-tomographic pore spaces examines combinations of relaxation parameters and bounce-back schemes with respect to accuracy, convergence rate, and viscosity independence, providing concrete tables of recommended parameter sets.[18]

### 3.4 Scalar Transport and Advection–Diffusion LBM

For species transport (e.g., lithium concentration), a separate distribution function \(g_i\) is used, often with a simpler lattice (D3Q7 or D3Q19) and BGK or MRT collision:[21][20]

\[
g_i(\mathbf{x} + \mathbf{e}\_i \Delta t, t + \Delta t) = g_i(\mathbf{x}, t) - \omega_g (g_i - g_i^{\mathrm{eq}}) + S_i \Delta t.
\]

The equilibrium is typically

\[
g_i^{\mathrm{eq}} = w_i C \left[1 + \frac{\mathbf{e}_i \cdot \mathbf{u}}{c_s^2} \right],
\]

leading to an advection–diffusion equation at the macroscopic level, suitable for modeling ionic concentration fields coupled to the hydrodynamic LBM.[21][10]

---

## 4. Multiphase LBM and Shan–Chen Models

### 4.1 Original Shan–Chen Pseudopotential Model

The Shan–Chen pseudopotential model introduces an interparticle force based on a density-dependent pseudopotential \(\psi(\rho)\):

\[
\mathbf{F}(\mathbf{x}) = -G \psi(\mathbf{x}) \sum_i w_i \psi(\mathbf{x} + \mathbf{e}\_i \Delta t) \mathbf{e}\_i,
\]

where \(G\) is an interaction strength parameter controlling phase separation and effective surface tension.[22][6]

This force is incorporated into the LBM via a forcing scheme (e.g., Shan–Chen’s original formulation or Guo’s forcing), yielding coexisting liquid and gas phases with a diffuse interface.[5][22]

The Shan–Chen model has been widely used for two-phase flow in porous media, including simulations of displacement, relative permeability, and capillary phenomena.[23][24][6]

### 4.2 Improved Pseudopotential and Forcing Schemes

A large body of work refines the Shan–Chen model to improve thermodynamic consistency, increase stable density ratios, and reduce spurious currents. Key contributions include:

- Critical reviews and improvements of pseudopotential models, clarifying the role of equation of state (EOS), mechanical stability, and forcing schemes.[25][22]
- 3D pseudopotential models capable of handling large density ratios (\(> 700\)) with improved forcing to match Maxwell constructions.[25]
- Recent multi-component improved pseudopotential LBM for displacement in 3D porous media, validated on sphere packs and micro-tomographic samples.[26][27]

### 4.3 Wettability and Contact Angle Control

Contact angle control in pseudopotential LBM has evolved from simple constant "solid density" schemes to more advanced approaches:

- Early work used uniform solid densities to tune effective contact angle, but suffered from unphysical thin films and limited density ratios.[22][5]
- Improved virtual-density schemes adjust an effective solid density based on neighboring fluid densities, enabling accurate control of contact angles across wide ranges (e.g., \(5^\circ\) to \(170^\circ\)) and high density ratios, while strongly reducing spurious currents.[28][29]
- Additional developments include curved-boundary schemes and immersed-boundary-based wetting conditions for complex geometries, designed to maintain mass conservation and low spurious currents at high density ratios.[30][28]

These schemes are directly relevant to battery electrodes, where electrolyte contact angle on active material and binder surfaces strongly influences electrolyte infiltration and wetting.[31][7]

---

## 5. LBM for Porous Electrodes and Li-Ion Batteries

### 5.1 Electrolyte Transport and Wetting

Several studies apply LBM to electrolyte transport and wetting in Li-ion porous electrodes:

- An early study applied multiphase LBM to electrolyte transport in 2D porous electrodes, showing that wettability and air removal strongly influence electrolyte distribution and suggesting that tuning porosity, particle size, and contact angle can improve wetting.[32]
- A three-dimensional, pore-resolved LBM model was developed to simulate electrolyte infiltration in cathode microstructures derived from tomography and stochastic generation, providing insights into how porosity, pore size distribution, and connectivity affect filling speed, trapped gas volumes, and their influence on electrochemical behavior.[9][7]
- Follow-up work by the same group used LBM to systematically study electrolyte filling of Li-ion electrodes, analyzing the impact of particle size, binder distribution, and wetting behavior, and connecting capillary pressure–saturation curves to optimized filling conditions.[8][33]

These works validate LBM as a tool for accurately capturing electrolyte infiltration and wetting in realistic Li-ion electrode microstructures.

### 5.2 Ion and Electron Transport and Discharge Performance

LBM has been extended beyond pure hydrodynamics to simulate coupled ion and electron transport and electrochemical reactions:

- Two-dimensional lattice Boltzmann models coupling ion transport in the electrolyte and electron transport in active particles have been used to investigate discharge processes in randomly reconstructed porous electrodes, highlighting how particle size, porosity, and particle shape affect local lithium concentration, potential distribution, and macroscopic discharge curves.[34][10]
- A 3D pore-scale LBM model for Li-ion electrodes simulates transport of ions and electrons, as well as electrochemical reactions at the active-material/electrolyte interface, in realistic NMC electrodes. The model’s predicted discharge curves match experimental data on the same electrodes, and simulations are used to quantify how surface area, pore size distribution, porosity, and tortuosity impact performance and utilization.[35][36][11]
- Pore-scale LBM models have also been applied to reactive transport in electrodes of Li–O2 and other chemistries, showing how microstructure affects discharge capacity, reaction front propagation, and pore blockage.[14]

These results provide a validated blueprint for using LBM to compute cell-level discharge behavior and inform microstructure design.

### 5.3 Electrolyte Filling and Manufacturing-Relevant Questions

Several recent works specifically address electrolyte filling and manufacturing using LBM:

- A 3D-resolved Shan–Chen LBM model simulating electrolyte filling in realistic electrode stacks demonstrates how electrode mesostructure, separator design, and process conditions determine filling time, residual gas distribution, and resulting electrochemical performance.[7][9]
- A related homogenized LBM approach combines grayscale and multi-component Shan–Chen models to represent unresolved sub-grid pores, maintaining control over interfacial tension and wettability while enabling larger-scale simulations.[27][37]
- Machine-learning surrogates have been trained on LBM-generated infiltration data to accelerate screening of electrode architectures, demonstrating the potential of LBM-based data factories for inverse design workflows.[38]

These works justify using Shan–Chen-based LBM plus advanced wettability models as the reference approach for modeling electrolyte infiltration and wetting at the pore scale in Li-ion electrodes.

### 5.4 Comparison to Porous Electrode and DFN Models

Porous electrode models following Doyle–Fuller–Newman (DFN) represent the electrode as a 1D homogenized medium with effective parameters \(D*{\mathrm{e,eff}},\kappa*{\mathrm{eff}}, \tau, a\), derived from microstructure. Recent work has highlighted regimes where the DFN fails to predict voltage accurately, especially at high C-rates and elevated temperatures, and proposed homogenized Poisson–Nernst–Planck-based alternatives.[39][40][12]

Pore-scale LBM studies comparing to DFN/P2D models show:

- For homogeneous structures, DFN and pore-scale LBM predictions agree well.
- For heterogeneous electrodes, DFN underestimates structure-sensitive transport–reaction coupling and non-uniform utilization, especially at high C-rates, even when using structure-based tortuosity.[41]

These results support using LBM as the primary predictive tool, with DFN-like equations serving as a conceptual reference rather than an implementation dependency.

---

## 6. Nonequilibrium Thermodynamics of Porous Electrodes

While the project aims to avoid PyBaMM, the broader theory of nonequilibrium thermodynamics in porous electrodes is still useful as a conceptual and mathematical foundation.

A generalized porous electrode framework based on nonequilibrium thermodynamics derives transport and reaction equations from variational principles, including bounds on effective diffusivity in terms of porosity and tortuosity.[3]

Key takeaways for the pore-scale LBM design are:

- Effective transport properties computed from LBM (e.g., \(D*{\mathrm{e,eff}},\kappa*{\mathrm{eff}}\)) should respect theoretical bounds in terms of \(\varepsilon\) and \(\tau\).[3]
- Reaction boundary conditions at active-material/electrolyte interfaces should be formulated in a way consistent with local Nernst potentials and Butler–Volmer kinetics.
- Pore-scale simulations can be interpreted within the homogenized theory framework, providing a bridge between microstructural physics and cell-level behavior.

---

## 7. JAX Ecosystem for LBM and CFD

### 7.1 JAX-LaB: A Direct Precedent

JAX-LaB is a high-performance, differentiable LBM library built on JAX, targeting multiphase and multiphysics flows in porous media.[42][43][44]

Key features directly relevant to a Li-ion electrode simulator include:

- D3Q19 lattice and MRT collision implementations for single-phase flow, validated on permeability benchmarks and capillary flows.[43]
- Shan–Chen pseudopotential multiphase model with EOS-based thermodynamic consistency and density ratios greater than \(10^7\) while maintaining low spurious currents.[42][43]
- Improved virtual-density wettability control scheme (following Li et al. 2019) integrated into the interaction force, enabling accurate contact angle control on flat and curved surfaces without unphysical films.[28][43]
- Multi-GPU scaling using JAX sharding primitives (e.g., `pjit`), demonstrated for large 3D porous media simulations.[43][42]

JAX-LaB provides a concrete example of how to structure JAX-based LBM kernels, manage data layouts, and integrate with physics-based machine learning, and can be used either as a reference implementation or as a dependency to be extended.

### 7.2 XLB and Other LBM/CFD Libraries in JAX

- **XLB** (Accelerated Lattice Boltzmann) is a JAX-based, GPU-accelerated LBM framework from Autodesk, focused on physics-based ML, offering BGK, MRT, and more advanced collision models with support for NVIDIA Warp backends.[45]
- **JAX-Fluids** is a fully differentiable CFD solver for compressible single- and two-phase Navier–Stokes flows in 3D, written entirely in JAX and supporting automatic differentiation and GPU/TPU execution.[46][47]
- **JAX-CFD** from Google provides building blocks for Navier–Stokes solvers, including finite-difference stencils, multigrid solvers, and example differentiable CFD simulations.[48]
- Tutorial implementations demonstrate simple LBM solvers in JAX for 2D flows (e.g., D2Q9 vortex street), showing how collision and streaming can be expressed using JAX array operations and `jit`.[49]

These libraries demonstrate patterns for implementing PDE solvers in JAX with performance competitive with traditional C++/CUDA codes while leveraging automatic differentiation.

### 7.3 JAX Parallelization Primitives

JAX supports several key transformations for parallel and high-performance computation:

- `jax.jit`: just-in-time compilation of pure functions for single-device acceleration.
- `jax.vmap`: vectorization across a batch dimension, pushing the mapped axis through primitive operations to generate SIMD-like code.
- `jax.pmap`: parallel map across multiple devices (GPUs/TPUs), suitable for data-parallel LBM where the domain is partitioned along one axis.[50][51][52]
- `pjit` and `sharding` APIs: finer-grained control over how arrays and computations are partitioned across device meshes, used in JAX-LaB and JAX-Fluids for multi-dimensional domain decomposition.[47][43]

The JAX documentation provides detailed descriptions and examples of `pmap`, including how to shard inputs across devices and how collectives operate in SPMD programs. Educational resources demonstrate combining `pmap` with `jit` and `vmap` to achieve nested parallelism (e.g., across devices and within-device vectorization).[53][51][54]

---

## 8. Recommended JAX Implementation Patterns for LBM

Based on the above libraries and documentation, several concrete patterns emerge for implementing the LBM solver in JAX:

1. **Data Layout:** Use a 4D array for distributions, e.g., `f.shape = (Nx, Ny, Nz, Q)` for D3Q19, where `Q=19`. For multi-component or scalar fields, add component or field dimensions as needed.[42][43]
2. **Collision Step:** Implement collision as a pure function operating on local distributions (and optionally macroscopic fields), written in terms of JAX array operations such as `jnp.einsum`, `@` with precomputed matrices (for MRT), or broadcasting arithmetic. This function is then vectorized across the full lattice via native broadcasting or `vmap`.
3. **Streaming Step:** Implement streaming using `jnp.roll`, `jnp.take`, or custom indexing along spatial axes, taking care to handle boundaries separately. Many JAX LBM examples treat streaming as a combination of axis shifts and masking.[49][42]
4. **Boundary Conditions:** Encode solid masks and boundary conditions as boolean or integer arrays, and apply them via masking operations after collision/streaming. Bounce-back conditions can be implemented as local swaps of opposite directions at solid nodes, based on a precomputed mapping of indices.[18][42]
5. **JIT Compilation:** Wrap the main timestep function in `jax.jit` to compile the entire update (collision + streaming + boundary conditions) into a single XLA computation. This improves performance significantly and is standard in JAX-based LBM/CFD codes.[47][42]
6. **Multi-GPU Sharding:** For large 3D domains, shard the domain along one axis (e.g., x) using `pmap` or `pjit`, such that each device hosts a slab with halo cells. Ghost-cell exchanges (halo swaps) can be implemented using JAX collectives or cross-replica shuffles, as demonstrated in JAX-LaB and XLB.[45][43][42]
7. **Differentiability:** Ensure all state updates are expressed in terms of JAX primitives so that autograd can compute gradients with respect to parameters (e.g., wettability, microstructure descriptors), enabling inverse design.

---

## 9. Consolidated Equation Set for the Planned Solver

This section lists the main equations that the proposed LBM-based battery solver will implement, along with their literature foundations.

### 9.1 Single-Phase Flow and Permeability

- D3Q19 MRT LBM for incompressible Stokes flow, with viscosity determined by \(\nu = c_s^2 (\tau - 1/2) \Delta t\).[16][17]
- No-slip boundary conditions via halfway bounce-back at solid nodes.[15][18]
- Permeability tensor components computed from steady-state flow under pressure gradient using Darcy’s law:

  \[ K\_{ii} = -\mu \frac{Q_i L_i}{A_i \Delta p_i}.\][6][18]

- Tortuosity tensor \(\tau\_{ij}\) estimated by comparing effective to intrinsic transport in each principal direction.[4][18]

### 9.2 Scalar Transport (Electrolyte Concentration)

- Advection–diffusion LBM for lithium concentration in electrolyte using a separate distribution \(g_i\).[21][10]
- Effective diffusion tensor \(D\_{\mathrm{e,eff}}\) extracted from steady-state fluxes in homogeneous concentration gradients.[11][21]

### 9.3 Multiphase Electrolyte–Gas Flow and Wetting

- Shan–Chen pseudopotential two-phase LBM with appropriate EOS to match equilibrium densities and surface tension.[6][25]
- Improved virtual-density wettability scheme enabling accurate contact-angle control and low spurious currents, following Li et al. and its implementation in JAX-LaB.[28][43]
- Capillary pressure–saturation curves \(P_c(S)\) and relative permeability functions \(k_r(S)\) derived from drainage and imbibition simulations in electrode microstructures.[37][26]

### 9.4 Electrochemical Coupling at the Pore Scale

- Nernst–Planck-type equations for lithium ions in electrolyte, approximated via advection–diffusion LBM with additional migration terms captured in effective parameters or extended forcing.[10][11]
- Ohmic electron transport in the solid phase, which can be solved either via finite-volume/finite-difference methods on the solid subdomain or via a second LBM for electrons (as in some battery LBM works).[13][10]
- Butler–Volmer interfacial kinetics at active-material/electrolyte interfaces, providing source terms \(j_n(x,y,z)\) for charge and mass conservation.[11][10]
- Macroscopic observables (voltage, current, capacity, internal resistance) obtained by integrating local currents and potentials over the electrode domain, following pore-scale battery LBM models.[41][10][11]

### 9.5 Degradation and Cycle-Life-Relevant Physics (Future Extension)

- SEI growth and Li plating can be added via additional reaction channels and evolving solid phases, informed by nonequilibrium thermodynamic models and phase-field-based porous electrode theories.[14][3]
- Changes in pore geometry due to degradation can be incorporated by periodically updating solid masks and rerunning LBM to obtain time-evolving \(K, \tau, D\_{\mathrm{e,eff}}\).

---

## 10. Key References by Topic

### 10.1 LBM Foundations and MRT

- D. D’Humières, “Multiple-relaxation-time lattice Boltzmann models in three dimensions,” provides a canonical exposition of MRT LBM, including D3Q19 moment bases and relaxation strategies.[17][16]
- Evaluations of SRT and MRT schemes for flow in porous media and micro-tomographic pore spaces provide guidance on parameter choices and expected accuracy.[2][1][18]

### 10.2 Multiphase LBM and Wettability

- Shan & Chen’s original multicomponent pseudopotential model and its application to flow in complex 3D geometries.[6]
- Critical reviews and improved pseudopotential models addressing thermodynamic consistency and large density ratios.[22][25]
- Improved wettability schemes, including virtual-density and curved-boundary methods, with demonstrations of reduced spurious currents and accurate contact-angle control.[30][5][28]

### 10.3 Pore-Scale LBM for Batteries and Electrodes

- LBM simulations of ion and electron transport in Li-ion porous electrodes during discharge, exploring effects of particle size, porosity, and microstructure.[34][13][10]
- 3D pore-scale LBM models for Li-ion electrodes capturing coupled transport and electrochemistry, validated against experimental discharge curves.[36][35][11]
- LBM studies of electrolyte infiltration and wetting in Li-ion electrodes, including full 3D tomography-based simulations.[33][8][32][9][7]

### 10.4 Theoretical Porous Electrode and DFN Context

- Reviews of porous electrode modeling and its applications to Li-ion batteries, including the original Doyle–Fuller–Newman model and subsequent developments.[40][12]
- Nonequilibrium thermodynamics of porous electrodes, providing rigorous bounds and variational formulations for effective transport.[3]
- Analyses of regimes where DFN-like models fail, underscoring the need for pore-scale approaches.[39][41]

### 10.5 JAX-Based LBM and CFD

- JAX-LaB as a direct template for JAX-based multiphase LBM in porous media, including Shan–Chen, improved wettability, and multi-GPU scaling.[44][43][42]
- XLB from Autodesk as another JAX-based LBM framework focused on physics-based ML and advanced collision models.[45]
- JAX-Fluids and JAX-CFD as general-purpose differentiable CFD toolkits demonstrating best practices for JAX-based PDE solvers.[46][48][47]
- JAX documentation and tutorials on `jit`, `vmap`, and `pmap` for parallelizing array-based computations across devices.[51][52][54][50]

These references collectively provide a solid foundation for implementing a pore-scale, LBM-based Li-ion electrode simulator entirely in JAX, with all major modeling decisions grounded in established literature.
