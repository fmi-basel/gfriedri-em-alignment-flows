import questionary
import yaml

from .parameter_config import FlowFieldEstimationConfig, MeshIntegrationConfig


def get_flow_field_estimation_config():
    patch_size = int(
        questionary.text(
            "patch_size:", default="160", validate=lambda v: v.isdigit()
        ).ask()
    )
    stride = int(
        questionary.text("stride", default="20", validate=lambda v: v.isdigit()).ask()
    )
    batch_size = int(
        questionary.text(
            "batch_size", default="256", validate=lambda v: v.isdigit()
        ).ask()
    )
    min_peak_ratio = float(
        questionary.text(
            "min_peak_ratio",
            default="1.6",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    min_peak_sharpness = float(
        questionary.text(
            "min_peak_sharpness",
            default="1.6",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    max_magnitude = float(
        questionary.text(
            "max_magnitude",
            default="80",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    max_deviation = float(
        questionary.text(
            "max_deviation",
            default="20",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    max_gradient = float(
        questionary.text(
            "max_gradient", default="0", validate=lambda v: v.replace(".", "").isdigit()
        ).ask()
    )
    min_patch_size = int(
        questionary.text(
            "min_patch_size", default="400", validate=lambda v: v.isdigit()
        ).ask()
    )

    return FlowFieldEstimationConfig(
        patch_size=patch_size,
        stride=stride,
        batch_size=batch_size,
        min_peak_ratio=min_peak_ratio,
        min_peak_sharpness=min_peak_sharpness,
        max_magnitude=max_magnitude,
        max_deviation=max_deviation,
        max_gradient=max_gradient,
        min_patch_size=min_patch_size,
    )


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
            default="100000",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    stop_v_max = float(
        questionary.text(
            "mesh_integration_config.stop_v_max:",
            default="0.005",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    dt_max = float(
        questionary.text(
            "mesh_integration_config.dt_max:",
            default="1000.0",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )
    prefer_orig_order = questionary.confirm(
        "mesh_integration_config.prefer_orig_order:", default=True
    ).ask()
    start_cap = float(
        questionary.text(
            "mesh_integration_config.start_cap:",
            default="0.01",
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
    block_size = int(
        questionary.text(
            "mesh_integration_config.block_size:",
            default="25",
            validate=lambda v: v.replace(".", "").isdigit(),
        ).ask()
    )

    return MeshIntegrationConfig(
        dt=dt,
        gamma=gamma,
        k0=k0,
        k=k,
        num_iters=num_iters,
        max_iters=max_iters,
        stop_v_max=stop_v_max,
        dt_max=dt_max,
        start_cap=start_cap,
        final_cap=final_cap,
        prefer_orig_order=prefer_orig_order,
        block_size=block_size,
    )


def get_warp_config():
    warp_start_section = int(
        questionary.text(
            "Warp start section:",
            default="0",
            validate=lambda x: x.isdigit() and int(x) >= 0,
        ).ask()
    )
    warp_end_section = int(
        questionary.text(
            "Warp end section:",
            default="9",
            validate=lambda x: x.isdigit() and int(x) >= 0,
        ).ask()
    )
    output_dir = questionary.path("Path to the output directory:").ask()
    volume_name = questionary.text(
        "Volume name:",
        default="fine_aligned_volume.zarr",
    ).ask()

    return dict(
        warp_start_section=warp_start_section,
        warp_end_section=warp_end_section,
        output_dir=output_dir,
        volume_name=volume_name,
    )


def main():
    stitched_sections_dir = questionary.path(
        "Path to the stitched sections directory:"
    ).ask()
    output_dir = questionary.path("Path to the output directory:").ask()

    config = dict(
        stitched_sections_dir=stitched_sections_dir,
        output_dir=output_dir,
        ffe_conf=get_flow_field_estimation_config().dict(),
        mi_conf=get_mesh_integration_config().dict(),
    )
    warp_config = get_warp_config()
    warp_config["stitched_sections_dir"] = stitched_sections_dir

    with open("fine_alignment_config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    with open("warp_config.yaml", "w") as f:
        yaml.safe_dump(warp_config, f, sort_keys=False)
