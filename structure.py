"""
Structure Generation Module

Generates 3D microstructures using:
- GRF: Gaussian Random Field (fast, default)
- physics: FiPy convection-diffusion-evaporation (accurate, GPU optional)
- physics_gpu: JAX-based solver (fast, less accurate)
"""
import os
import copy
import numpy as np
import tifffile
from scipy.fft import ifftn 

# JAX 
JAX_AVAILABLE = False
JAX_GPU = False
try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    from functools import partial
    JAX_AVAILABLE = True
    try:
        jax.devices('gpu')
        JAX_GPU = True
    except:
        pass
except ImportError:
    pass

# FiPy for physics  simulation
FIPY_AVAILABLE = False
try:
    from fipy import (
        CellVariable, FaceVariable, Grid3D,
        TransientTerm, DiffusionTerm, ConvectionTerm, ImplicitSourceTerm,
    )
    FIPY_AVAILABLE = True
except ImportError:
    pass

# PyAmgx for GPU accelerated FiPy solvers
PYAMGX_AVAILABLE = False
_pyamgx = None
try:
    import pyamgx as _pyamgx
    _pyamgx.initialize()
    from fipy.solvers.solver import Solver as _FipySolver
    from fipy.matrices.scipyMatrix import _ScipyMeshMatrix
    from fipy.tools import numerix
    from scipy.sparse import linalg
    PYAMGX_AVAILABLE = True
except ImportError:
    pass


# GPU Solver for FiPy 

if PYAMGX_AVAILABLE:
    class GPUAMGXSolver(_FipySolver):
        """GPU-accelerated AMG solver using PyAMGX for FiPy."""
        
        CONFIG = {
            "config_version": 2,
            "determinism_flag": 1,
            "solver": {
                "algorithm": "AGGREGATION",
                "solver": "AMG",
                "selector": "SIZE_2",
                "monitor_residual": 1,
                "max_levels": 1000,
                "cycle": "V",
                "smoother": "BLOCK_JACOBI",
                "presweeps": 2,
                "postsweeps": 2,
            }
        }
        
        def __init__(self, tolerance=1e-6, iterations=100):
            super().__init__(tolerance=tolerance, criterion="default", 
                           iterations=iterations, precon=None)
            
            config = copy.deepcopy(self.CONFIG)
            config["solver"]["max_iters"] = iterations
            config["solver"]["tolerance"] = tolerance
            config["solver"]["convergence"] = "RELATIVE_INI_CORE"
            
            self.cfg = _pyamgx.Config().create_from_dict(config)
            self.resources = _pyamgx.Resources().create_simple(self.cfg)
            self.x_gpu = _pyamgx.Vector().create(self.resources)
            self.b_gpu = _pyamgx.Vector().create(self.resources)
            self.A_gpu = _pyamgx.Matrix().create(self.resources)
        
        @property
        def _matrixClass(self):
            return _ScipyMeshMatrix
        
        def _rhsNorm(self, L, x, b):
            return numerix.L2norm(b)
        
        def _matrixNorm(self, L, x, b):
            return linalg.norm(L.matrix, ord=numerix.inf)
        
        def _solve_(self, L, x, b):
            self.x_gpu.upload(x)
            self.b_gpu.upload(b)
            self.A_gpu.upload_CSR(L)
            
            solver = _pyamgx.Solver().create(self.resources, self.cfg)
            solver.setup(self.A_gpu)
            solver.solve(self.b_gpu, self.x_gpu)
            self.x_gpu.download(x)
            solver.destroy()
            
            return x
        
        def _cleanup(self):
            pass
        
        def __del__(self):
            for attr in ['A_gpu', 'b_gpu', 'x_gpu', 'cfg']:
                if hasattr(self, attr):
                    getattr(self, attr).destroy()
else:
    class GPUAMGXSolver:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyAMGX not available")


# === Utility Functions ===

def prune_isolated_voxels(solid, min_neighbors=2):
    """Remove isolated solid voxels with fewer than min_neighbors."""
    neighbors = np.zeros_like(solid, dtype=int)
    neighbors[1:] += solid[:-1]
    neighbors[:-1] += solid[1:]
    neighbors[:, 1:] += solid[:, :-1]
    neighbors[:, :-1] += solid[:, 1:]
    neighbors[:, :, 1:] += solid[:, :, :-1]
    neighbors[:, :, :-1] += solid[:, :, 1:]
    return solid & (neighbors >= min_neighbors)


def gaussian_random_field(shape, psd_power=1.0, anisotropy=(1.0, 1.0, 1.0), seed=42):
    """Generate a 3D Gaussian Random Field."""
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape

    kz = np.fft.fftfreq(nz)[:, None, None]
    ky = np.fft.fftfreq(ny)[None, :, None]
    kx = np.fft.fftfreq(nx)[None, None, :]

    k2 = ((kz * anisotropy[0])**2 + (ky * anisotropy[1])**2 + (kx * anisotropy[2])**2)
    k2[0, 0, 0] = 1e-12

    amplitude = 1.0 / (k2 ** (psd_power / 2.0))
    phase = rng.random(shape) * 2.0 * np.pi
    F = (np.cos(phase) + 1j * np.sin(phase)) * amplitude
    field = ifftn(F).real

    return (field - field.mean()) / (field.std() + 1e-12)


# === JAX GPU Physics Solver ===

if JAX_AVAILABLE:
    @partial(jit, static_argnums=(1,))
    def _diffusion_step(binder, dx=1.0):
        """Compute diffusion term."""
        return (
            jnp.roll(binder, 1, 0) + jnp.roll(binder, -1, 0) +
            jnp.roll(binder, 1, 1) + jnp.roll(binder, -1, 1) +
            jnp.roll(binder, 1, 2) + jnp.roll(binder, -1, 2) - 6 * binder
        ) / (dx * dx)

    @jit
    def _apply_bc(field):
        """Apply no-flux boundary conditions."""
        field = field.at[0].set(field[1])
        field = field.at[-1].set(field[-2])
        field = field.at[:, 0].set(field[:, 1])
        field = field.at[:, -1].set(field[:, -2])
        field = field.at[:, :, 0].set(field[:, :, 1])
        field = field.at[:, :, -1].set(field[:, :, -2])
        return field


def generate_physics_structure_gpu(shape, params, seed):
    """GPU-accelerated physics-based structure generation using JAX."""
    if not JAX_AVAILABLE:
        raise ImportError("JAX required. Install: pip install jax jaxlib")
    
    nz, ny, nx = shape
    rng = np.random.default_rng(seed)
    
    # Initial structure from GRF
    field = gaussian_random_field(shape, psd_power=1.5, anisotropy=(0.8, 1.0, 1.0), seed=seed)
    solid = field > np.quantile(field, params["target_porosity"])
    
    # Parameters
    drying = params.get("drying_intensity", 0.3)
    nsteps = params.get("time_steps", 40)
    dt = params.get("dt", 0.05)
    
    # Initialize binder
    binder = jnp.ones(shape) * 0.1 * (1.0 + 0.5 * solid.astype(float))
    
    # Evaporation profile
    z = jnp.arange(nz).reshape(-1, 1, 1) / (nz - 1)
    evap = 0.2 * drying * jnp.exp(-5.0 * (1.0 - z))
    top_mask = (z > 0.8).astype(float)
    
    @jit
    def step(b):
        b_new = b + dt * (_diffusion_step(b) - evap * b)
        return _apply_bc(jnp.maximum(b_new, 0.0))
    
    for i in range(nsteps):
        prev = float(jnp.sum(binder))
        binder = step(binder)
        lost = max(0, prev - float(jnp.sum(binder)))
        if lost > 0:
            binder = binder + (0.08 * drying * lost / jnp.sum(top_mask)) * top_mask
    
    # Create final structure
    binder_np = np.array(binder)
    binder_norm = (binder_np - binder_np.min()) / (binder_np.max() - binder_np.min() + 1e-12)
    dens_mask = rng.random(shape) < (drying * binder_norm)
    final_solid = prune_isolated_voxels(solid | dens_mask, min_neighbors=2)
    
    return (~final_solid).astype(np.uint8)


    # FiPy Physics Solver

def generate_physics_structure(shape, params, seed, use_gpu_fipy=False):
    """Generate microstructure using FiPy convection-diffusion-evaporation."""
    if not FIPY_AVAILABLE:
        raise ImportError("FiPy required. Install: pip install fipy")
    
    # gpu solver setup
    solver = None
    if use_gpu_fipy and PYAMGX_AVAILABLE:
        solver = GPUAMGXSolver(tolerance=1e-6, iterations=100)
    
    nz, ny, nx = shape
    rng = np.random.default_rng(seed)
    
    # Initial structure from GRF
    field = gaussian_random_field(shape, psd_power=1.5, anisotropy=(0.8, 1.0, 1.0), seed=seed)
    solid = field > np.quantile(field, params["target_porosity"])
    
    # FiPy mesh (uses x,y,z ordering)
    mesh = Grid3D(nx=nx, ny=ny, nz=nz, dx=1.0, dy=1.0, dz=1.0)
    solid_flat = solid.transpose(2, 1, 0).ravel()
    
    # Parameters
    drying = params.get("drying_intensity", 0.3)
    nsteps = params.get("time_steps", 40)
    dt = params.get("dt", 0.05)
    v_scale = params.get("velocity_scale", 0.6)
    
    # Variables
    binder = CellVariable(name="binder", mesh=mesh, value=0.1)
    binder.setValue(binder.value * (1.0 + 0.5 * solid_flat))
    
    D = CellVariable(mesh=mesh, value=params.get("diff_pore", 1.0))
    D.setValue(params.get("diff_solid", 0.1), where=solid_flat)
    
    velocity = FaceVariable(mesh=mesh, rank=1, value=(0.0, 0.0, v_scale * drying))
    
    z = np.array(mesh.cellCenters[2])
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-12)
    evap = CellVariable(mesh=mesh, value=np.exp(-5.0 * (1.0 - z_norm)))
    k_evap = 0.2 * drying
    
    # PDE equation
    eq = TransientTerm() == DiffusionTerm(coeff=D) - ConvectionTerm(coeff=velocity) - ImplicitSourceTerm(k_evap * evap)
    
    top_mask = z_norm > 0.8

    # Time stepping
    for step in range(nsteps):
        prev_total = float(np.sum(binder.value))
        
        if solver:
            eq.solve(var=binder, dt=dt, solver=solver)
        else:
            eq.solve(var=binder, dt=dt)
        
        after_total = float(np.sum(binder.value))
        
        # Redeposition at top
        lost = max(0.0, prev_total - after_total)
        if lost > 0:
            vals = binder.value.copy()
            vals[top_mask] += 0.08 * drying * lost / (np.sum(top_mask) + 1e-9)
            binder.setValue(vals)
        
    
    # Convert to structure
    binder_field = np.array(binder.value).reshape((nx, ny, nz)).transpose(2, 1, 0)
    binder_norm = (binder_field - binder_field.min()) / (binder_field.max() - binder_field.min() + 1e-12)
    
    dens_mask = rng.random(shape) < (drying * binder_norm)
    final_solid = prune_isolated_voxels(solid | dens_mask, min_neighbors=2)
    
    return (~final_solid).astype(np.uint8)


# Entry Point 

def generate_structure(run_id, config, output_path):
    """
    Generate a 3D microstructure and save as TIFF.
    
    Methods:
    - "GRF": Gaussian Random Field (fast)
    - "physics": FiPy simulation (accurate, GPU optional via use_gpu_fipy)
    - "physics_gpu": JAX simulation (fast, less accurate)
    """
    params = config["structure"]
    shape = tuple(map(int, params["volume_shape"]))
    method = params.get("method", "GRF").lower()
    seed = config["general"]["base_seed"] + run_id
    
    if method == "physics":
        if not FIPY_AVAILABLE:
            raise ImportError("FiPy required. Install: pip install fipy")
        use_gpu = config.get("general", {}).get("use_gpu_fipy", False)
        binary_vol = generate_physics_structure(shape, params, seed, use_gpu_fipy=use_gpu)
    
    elif method == "physics_gpu":
        if not JAX_AVAILABLE:
            raise ImportError("JAX required. Install: pip install jax jaxlib")
        binary_vol = generate_physics_structure_gpu(shape, params, seed)
    
    else:  # GRF
        field = gaussian_random_field(
            shape=shape,
            psd_power=params["psd_power"],
            anisotropy=tuple(params["anisotropy"]),
            seed=seed,
        )
        binary_vol = (field <= np.quantile(field, params["target_porosity"])).astype(np.uint8)
    
    # Save
    filename = f"sample_{run_id:04d}.tif"
    filepath = os.path.join(output_path, filename)
    
    if config["general"]["save_images"]:
        tifffile.imwrite(filepath, binary_vol * 255)
    
    return filepath, binary_vol
