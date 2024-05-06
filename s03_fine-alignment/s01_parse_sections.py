import argparse
import json
import os
import re
from os.path import basename, join
from pathlib import Path

import yaml


def main(
    stitched_section_dir: str,
):
    section_dirs = list_zarr_sections(root_dir=stitched_section_dir)
    yx_size = get_yx_size(section_dirs, bin=1)

    batch_size = 20

    for batch_number, i in enumerate(range(0, len(section_dirs), batch_size)):
        section_dirs_chunk = section_dirs[i : i + batch_size + 1]
        if len(section_dirs_chunk) > 1:
            with open(f"section_dirs_chunk_{batch_number}.yaml", "w") as f:
                yaml.safe_dump(
                    dict(
                        section_dirs=section_dirs_chunk,
                        yx_size=yx_size,
                    ),
                    f,
                )


def list_zarr_sections(root_dir: str) -> list[str]:
    filename_re = re.compile(r"s[0-9]*_g[0-9]*.zarr")
    files = []
    root, dirs, _ = next(os.walk(root_dir))
    for d in dirs:
        m_filename = filename_re.fullmatch(d)
        if m_filename:
            files.append(str(Path(root).joinpath(d)))

    files.sort(key=lambda v: int(basename(v).split("_")[0][1:]))
    return files


def get_yx_size(section_dirs: list[str], bin: int = 1):

    max_size_y = 0
    max_size_x = 0
    for sec_dir in section_dirs:
        with open(join(sec_dir, "0", ".zarray")) as f:
            shape = json.load(f)["shape"]

        with open(join(sec_dir, "coarse_stack_padding.json")) as f:
            shifts = json.load(f)
            shift_y = shifts["shift_y"]
            shift_x = shifts["shift_x"]

        size_y = shape[0] + shift_y
        size_x = shape[1] + shift_x
        if size_y > max_size_y:
            max_size_y = size_y
        if size_x > max_size_x:
            max_size_x = size_x

    max_size_y = max_size_y - max_size_y % bin
    max_size_x = max_size_x - max_size_x % bin

    assert max_size_y % bin == 0, "yx_size must be divisible by bin."
    assert max_size_x % bin == 0, "yx_size must be divisible by bin."

    return max_size_y, max_size_x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        default="fine_alignment_config.yaml",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(
        stitched_section_dir=config["stitched_sections_dir"],
    )
