import questionary
import yaml


def build_config():
    user_name = questionary.text("User name:").ask()
    stitched_sections_dir = questionary.path(
        "Path to the stitched sections directory:"
    ).ask()
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
    flow_stride = int(
        questionary.text(
            "Flow stride:",
            default="40",
            validate=lambda x: x.isdigit() and int(x) > 0,
        ).ask()
    )
    block_size = int(
        questionary.text(
            "Block size:",
            default="50",
            validate=lambda x: x.isdigit() and int(x) > 0,
        ).ask()
    )
    map_zarr_dir = questionary.path(
        "Path to the map zarr directory:",
        default="",
    ).ask()
    output_dir = questionary.path("Path to the output directory:").ask()
    volume_name = questionary.text(
        "Volume name:",
        default="fine_aligned_volume.zarr",
    ).ask()

    config = dict(
        user=user_name,
        stitched_sections_dir=stitched_sections_dir,
        warp_start_section=warp_start_section,
        warp_end_section=warp_end_section,
        flow_stride=flow_stride,
        block_size=block_size,
        map_zarr_dir=map_zarr_dir,
        output_dir=output_dir,
        volume_name=volume_name,
    )

    with open("warp_fine_aligned_sections.config", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


if __name__ == "__main__":
    build_config()
