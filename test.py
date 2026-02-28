from structure import load_run_config, load_materials_db, resolve, compute_composition

cfg = load_run_config("str_gen_config.yml")
db = load_materials_db("materials_db.yml")
sim = resolve(cfg, db)

comp = compute_composition(sim)
print(comp.summary())

# Access downstream
print(comp.N_carbon)  # → 263  (PSD-corrected)
print(comp.phi_solid_pre)  # → 0.420  (safe for RSA)
print(comp.si_in_voxels)  # → 0.26  (triggers INFO warning, expected)
print(comp.voxel_size_nm)  # → 390.6 nm
