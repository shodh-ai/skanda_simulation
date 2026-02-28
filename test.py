from structure import (
    load_run_config,
    load_materials_db,
    resolve,
    compute_composition,
    build_domain,
)

cfg = load_run_config("str_gen_config.yml")
db = load_materials_db("materials_db.yml")
sim = resolve(cfg, db)

comp = compute_composition(sim)
print(comp.summary())

domain = build_domain(comp)
print(domain.summary())
