import questionary
import yaml


def build_config():
    user_name = questionary.text("User name:").ask()
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

    config = dict(
        user=user_name,
        stitched_sections_dir=stitched_sections_dir,
        start_section=start_section,
        end_section=end_section,
        max_parallel_jobs=10,
    )

    with open("coarse-align.config", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


if __name__ == "__main__":
    build_config()
