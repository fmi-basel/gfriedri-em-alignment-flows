import questionary
import yaml
from parameter_config import FlowFieldEstimationConfig


def build_config():
    user_name = questionary.text("User name:").ask()
    stitched_sections_dir = questionary.path(
        "Path to the stitched sections directory:"
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

    patch_size = int(
        questionary.text(
            "patch_size:", default="160", validate=lambda v: v.isdigit()
        ).ask()
    )
    stride = int(
        questionary.text("stride", default="40", validate=lambda v: v.isdigit()).ask()
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

    config = dict(
        user=user_name,
        stitched_sections_dir=stitched_sections_dir,
        start_section=start_section,
        end_section=end_section,
        ffe_conf=FlowFieldEstimationConfig(
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            min_peak_ratio=min_peak_ratio,
            min_peak_sharpness=min_peak_sharpness,
            max_magnitude=max_magnitude,
            max_deviation=max_deviation,
            max_gradient=max_gradient,
            min_patch_size=min_patch_size,
        ),
        max_parallel_jobs=10,
    )

    with open("fine-align-estimate-flow-fields.config", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


if __name__ == "__main__":
    build_config()
