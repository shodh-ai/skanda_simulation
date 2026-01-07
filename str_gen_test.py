from structure.schema import (
    AnodeType,
    Geometry,
    GraphiteParams,
    GraphiteType,
    ConductiveAdditiveType,
    BinderType,
    DistributionMode,
    BinderDistribution,
    GenerationParams,
    CalenderingParams,
    SEILayerParams,
    ContactParams,
    PercolationParams,
    DefectParams,
    ParticleCrackParams,
    BinderAgglomerationParams,
    DelaminationParams,
    PoreClusteringParams,
)
from structure import generate_anode_microstructure
import matplotlib.pyplot as plt


def example_graphite_generation():
    """Example: Generate a graphite anode microstructure."""

    # Define geometry
    geometry = Geometry(
        shape=(256, 256, 256), field_of_view_x_um=20.0, coating_thickness_um=50.0
    )

    # Define graphite parameters
    graphite_params = GraphiteParams(
        graphite_type=GraphiteType.ARTIFICIAL,
        primary_particle_size_um=15.0,
        secondary_particle_size_um=20.0,
        size_distribution=0.25,
        aspect_ratio=5.0,
        d002_spacing_nm=0.336,
        crystallite_lc_nm=150.0,
        crystallite_la_nm=50.0,
        orientation_degree=0.6,
        target_porosity=0.2,
        # Conductive additive (inline)
        conductive_additive_type=ConductiveAdditiveType.CARBON_BLACK,
        conductive_additive_wt_frac=0.02,
        conductive_additive_particle_size_nm=40.0,
        conductive_additive_distribution=DistributionMode.NETWORK,
        # Binder (inline)
        binder_type=BinderType.PVDF,
        binder_wt_frac=0.05,
        binder_distribution=BinderDistribution.NECKS,
        binder_film_thickness_nm=15.0,
    )

    # Define generation parameters
    generation = GenerationParams(
        calendering=CalenderingParams(
            compression_ratio=0.7,
            particle_deformation=0.3,
            orientation_enhancement=0.2,
        ),
        sei_layer=SEILayerParams(
            enabled=True,
            thickness_nm=15.0,
            uniformity=0.3,
        ),
        contacts=ContactParams(
            coordination_number=6.0,
            contact_area_fraction=0.10,
        ),
        tortuosity_manual=None,
        percolation=PercolationParams(
            enforce_percolation=True,
            min_percolation=0.95,
        ),
    )

    # Define defect parameters
    defects = DefectParams(
        particle_cracks=ParticleCrackParams(
            enabled=False,
            crack_probability=0.0,
            crack_width_nm=50.0,
        ),
        binder_agglomeration=BinderAgglomerationParams(
            enabled=False,
            agglomeration_probability=0.0,
        ),
        delamination=DelaminationParams(
            enabled=False,
            delamination_fraction=0.0,
        ),
        pore_clustering=PoreClusteringParams(
            enabled=False,
            clustering_degree=0.0,
        ),
    )

    # Generate microstructure
    binary_volume = generate_anode_microstructure(
        run_id=0,
        seed=42,
        geometry=geometry,
        active_type=AnodeType.GRAPHITE,
        anode_params=graphite_params,
        generation=generation,
        defects=defects,
    )

    print(f"\nGenerated microstructure shape: {binary_volume.shape}")
    print(f"Data type: {binary_volume.dtype}")
    print(f"Porosity: {binary_volume.mean():.3f}")

    return binary_volume


if __name__ == "__main__":
    print("Example: Graphite Anode Generation")
    print("=" * 60)
    microstructure = example_graphite_generation()
    mid_index = microstructure.shape[0] // 2
    slice_img = microstructure[mid_index, :, :]

    plt.figure(figsize=(8, 8))
    # Using 'gray' colormap: 0 (black) usually pores, 1 (white) usually solid
    plt.imshow(slice_img, cmap="gray", interpolation="nearest")
    plt.title(f"Microstructure Slice (Index {mid_index})")
    plt.colorbar(label="Phase")

    # Save the image
    output_filename = "microstructure_slice.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nSaved slice image to: {output_filename}")
    print("\n✓ Generation complete!")
