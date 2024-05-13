import questionary
import yaml


def build_config():
    stitched_sections_dir = questionary.path(
        "Path to the stitched sections " "directory:"
    ).ask()

    start_section = int(
        questionary.text(
            "start_section:",
            default="0",
            validate=lambda v: v.isdigit(),
        ).ask()
    )
    end_section = int(
        questionary.text(
            "end_section:",
            default="10",
            validate=lambda v: v.isdigit(),
        ).ask()
    )

    output_dir = questionary.path("Output dir:").ask()
    volume_name = questionary.text("Volume name:").ask()
    bin = int(
        questionary.text(
            "Bin factor:", default="2", validate=lambda v: v.isdigit()
        ).ask()
    )

    config = dict(
        stitched_sections_dir=stitched_sections_dir,
        start_section=start_section,
        end_section=end_section,
        output_dir=output_dir,
        volume_name=volume_name,
        bin=bin,
    )

    with open("coarse-align.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


if __name__ == "__main__":
    build_config()
