import questionary
import yaml
from parameter_config import MeshIntegrationConfig


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
            default="50",
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


def build_config():
    user_name = questionary.text("User name:").ask()
    stitched_sections_dir = questionary.path(
        "Path to the stitched sections directory:"
    ).ask()

    mic = get_mesh_integration_config()

    flow_stride = int(
        questionary.text(
            "flow_stride", default="40", validate=lambda v: v.isdigit()
        ).ask()
    )

    output_dir = questionary.path("Path to the output directory:").ask()

    config = dict(
        user=user_name,
        stitched_sections_dir=stitched_sections_dir,
        mesh_integration=mic.dict(),
        flow_stride=flow_stride,
        output_dir=output_dir,
    )

    with open("relax_meshes.config", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


if __name__ == "__main__":
    build_config()
