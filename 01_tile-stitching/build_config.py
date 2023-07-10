import os
from os.path import join

import questionary
import yaml


def get_mesh_integration_config():
    dt = float(
        questionary.text(
            "mesh_integration_config.dt:",
            default="0.001",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    gamma = float(
        questionary.text(
            "mesh_integration_config.gamma:",
            default="0.0",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    k0 = float(
        questionary.text(
            "mesh_integration_config.k0:",
            default="0.01",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    k = float(
        questionary.text(
            "mesh_integration_config.k:",
            default="0.1",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    stride = int(
        questionary.text(
            "mesh_integration_config.stride:",
            default="20",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    num_iters = int(
        questionary.text(
            "mesh_integration_config.num_iters:",
            default="1000",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    max_iters = int(
        questionary.text(
            "mesh_integration_config.max_iters:",
            default="20000",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    stop_v_max = float(
        questionary.text(
            "mesh_integration_config.stop_v_max:",
            default="0.001",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    dt_max = float(
        questionary.text(
            "mesh_integration_config.dt_max:",
            default="100.0",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    prefer_orig_order = questionary.confirm(
        "mesh_integration_config.prefer_orig_order:", default=True
    ).ask()
    start_cap = float(
        questionary.text(
            "mesh_integration_config.start_cap:",
            default="1.0",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    final_cap = float(
        questionary.text(
            "mesh_integration_config.final_cap:",
            default="10.0",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    remove_drift = questionary.confirm(
        "mesh_integration_config.remove_drift:",
        default=True,
    ).ask()

    from s02_register_tiles import MeshIntegrationConfig

    return MeshIntegrationConfig(
        dt=dt,
        gamma=gamma,
        k0=k0,
        k=k,
        stride=stride,
        num_iters=num_iters,
        max_iters=max_iters,
        stop_v_max=stop_v_max,
        dt_max=dt_max,
        prefer_orig_order=prefer_orig_order,
        start_cap=start_cap,
        final_cap=final_cap,
        remove_drift=remove_drift,
    )


def get_registration_config():
    overlaps_x = [
        int(x.strip())
        for x in questionary.text(
            "registration_config.overlaps_x:",
            default="200, 300, 400",
            validate=lambda v: v.replace(",", "").replace(" ", "").isdigit(),
        )
        .ask()
        .split(",")
    ]
    overlaps_y = [
        int(x.strip())
        for x in questionary.text(
            "registration_config.overlaps_y:",
            default="200, 300, 400",
            validate=lambda v: v.replace(",", "").replace(" ", "").isdigit(),
        )
        .ask()
        .split(",")
    ]
    min_overlap = int(
        questionary.text(
            "registration_config.min_overlap:",
            default="20",
            validate=lambda v: v.isdigit(),
        ).ask()
    )
    min_range = [
        int(x.strip())
        for x in questionary.text(
            "registration_config.min_range:",
            default="10, 100, 0",
            validate=lambda v: v.replace(",", "").replace(" ", "").isdigit(),
        )
        .ask()
        .split(",")
    ]
    patch_size = [
        int(x.strip())
        for x in questionary.text(
            "registration_config.patch_size:",
            default="80, 80",
            validate=lambda v: v.replace(",", "").replace(" ", "").isdigit(),
        )
        .ask()
        .split(",")
    ]
    batch_size = int(
        questionary.text(
            "registration_config.batch_size:",
            default="8000",
            validate=lambda v: v.isdigit(),
        ).ask()
    )
    min_peak_ratio = float(
        questionary.text(
            "registration_config.min_peak_ratio:",
            default="1.4",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    min_peak_sharpness = float(
        questionary.text(
            "registration_config.min_peak_sharpness:",
            default="1.4",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    max_deviation = int(
        questionary.text(
            "registration_config.max_deviation:",
            default="5",
            validate=lambda v: v.isdigit(),
        ).ask()
    )
    max_magnitude = int(
        questionary.text(
            "registration_config.max_magnitude:",
            default="0",
            validate=lambda v: v.isdigit(),
        ).ask()
    )
    min_patch_size = int(
        questionary.text(
            "registration_config.min_patch_size:",
            default="10",
            validate=lambda v: v.isdigit(),
        ).ask()
    )
    max_gradient = float(
        questionary.text(
            "registration_config.max_gradient:",
            default="-1.0",
            validate=lambda v: v.replace(".", "").replace("-", "").isdigit(),
        ).ask()
    )
    reconcile_flow_max_deviation = float(
        questionary.text(
            "registration_config.reconcile_flow_max_deviation:",
            default="-1",
            validate=lambda v: v.replace(".", "").replace("-", "").isdigit(),
        ).ask()
    )

    from s02_register_tiles import RegistrationConfig

    return RegistrationConfig(
        overlaps_X=overlaps_x,
        overlaps_y=overlaps_y,
        min_overlap=min_overlap,
        min_range=min_range,
        patch_size=patch_size,
        batch_size=batch_size,
        min_peak_ratio=min_peak_ratio,
        min_peak_sharpness=min_peak_sharpness,
        max_deviation=max_deviation,
        max_magnitude=max_magnitude,
        min_patch_size=min_patch_size,
        max_gradient=max_gradient,
        reconcile_flow_max_deviation=reconcile_flow_max_deviation,
    )


def get_warp_config():
    margin = int(
        questionary.text(
            "warp_config.margin:", default="20", validate=lambda v: v.isdigit()
        ).ask()
    )
    use_clahe = questionary.confirm(
        "warp_config.use_clahe:",
        default=True,
    )
    if use_clahe:
        kernel_size = int(
            questionary.text(
                "warp_config.kernel_size:",
                default="1024",
                validate=lambda v: v.isdigit(),
            ).ask()
        )
        clip_limit = float(
            questionary.text(
                "warp_config.clip_limit:",
                default="1024",
                validate=lambda v: v.replace(".", "").isdigit(),
            ).ask()
        )
        nbins = int(
            questionary.text(
                "warp_config.nbins:", default="256", validate=lambda v: v.isdigit()
            ).ask()
        )
    else:
        kernel_size = 1024
        clip_limit = 0.01
        nbins = 256

    warp_parallelism = int(
        questionary.text(
            "warp_config.warp_parallelism:", default="5", validate=lambda v: v.isdigit()
        ).ask()
    )

    from s03_warp_tiles import WarpConfig

    return WarpConfig(
        margin=margin,
        use_clahe=use_clahe,
        kernel_size=kernel_size,
        clip_limit=clip_limit,
        nbins=nbins,
        warp_parallelism=warp_parallelism,
    )


def get_acquistion_config():
    sbem_root_dir = questionary.path("Path to the SBEM root directory:").ask()
    acquisition = questionary.text("Acquisition name:", default="run_0").ask()
    tile_grid = questionary.text("Tile Grid ID:", default="g0001").ask()
    thickness = float(
        questionary.text(
            "sbem_config.Thickness:",
            default="25",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    resolution_xy = float(
        questionary.text(
            "sbem_config.Resolution [XY]:",
            default="11",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    tile_width = float(
        questionary.text(
            "sbem_config.Tile width:",
            default="3072",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    tile_height = float(
        questionary.text(
            "sbem_config.Tile height:",
            default="2304",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    tile_overlap = float(
        questionary.text(
            "sbem_config.Tile overlap:",
            default="200",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )

    from prefect_tile_stitching import AcquisitionConfig

    return AcquisitionConfig(
        sbem_root_dir=sbem_root_dir,
        acquisition=acquisition,
        tile_grid=tile_grid,
        thickness=thickness,
        resolution_xy=resolution_xy,
        tile_width=tile_width,
        tile_height=tile_height,
        tile_overlap=tile_overlap,
    )


def build_config():
    user_name = questionary.text("User name:").ask()
    output_dir = questionary.path("Path to the output directory:").ask()
    acquisition_config = get_acquistion_config()

    section_dir = join(output_dir, "sections")
    mesh_integration_config = get_mesh_integration_config()
    registration_config = get_registration_config()

    stitched_section_dir = join(output_dir, "stitched-sections")
    warp_config = get_warp_config()

    os.makedirs(section_dir, exist_ok=True)
    os.makedirs(stitched_section_dir, exist_ok=True)

    config = dict(
        user=user_name,
        output_dir=output_dir,
        acquisition_config=dict(acquisition_config),
        mesh_integration_config=dict(mesh_integration_config),
        registration_config=dict(registration_config),
        warp_config=dict(warp_config),
        max_parallel_jobs=10,
    )

    with open("tile-stitching.config", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
