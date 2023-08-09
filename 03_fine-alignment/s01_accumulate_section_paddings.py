import json
import os
import re
from os.path import basename, join
from pathlib import Path

import numpy as np
from numpy._typing import ArrayLike


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


def filter(section_dirs: list[str], start_section: int, end_section: int):
    kept = []
    for sec in section_dirs:
        sec_idx = int(basename(sec).split("_")[0][1:])
        if start_section <= sec_idx <= end_section:
            kept.append(sec)

    return kept


def load_shifts(section_dirs: list[str]):
    shifts = []
    for sec in section_dirs[1:]:

        with open(join(sec, "shift_to_previous.json")) as f:
            data = json.load(f)
            shifts.append([data["shift_y"], data["shift_x"]])

    return np.array(shifts)


def get_padding_per_section(shifts) -> ArrayLike:
    cumulated_shifts = np.cumsum(shifts, axis=0)
    cumulated_shifts = np.concatenate([np.array([[0, 0]]), cumulated_shifts], 0)
    cumulated_padding = cumulated_shifts + np.abs(np.min(cumulated_shifts, axis=0))
    return cumulated_padding


def accumulate_paddings(
    stitched_section_dir: str = "",
    end_section: int = 9,
):
    section_dirs = list_zarr_sections(root_dir=stitched_section_dir)

    section_dirs = filter(
        section_dirs=section_dirs,
        start_section=0,
        end_section=end_section,
    )

    shifts = load_shifts(section_dirs=section_dirs)
    section_paddings = get_padding_per_section(shifts=shifts)

    for section, padding in zip(section_dirs, section_paddings):
        with open(join(section, "padding.json"), "w") as f:
            json.dump(padding.tolist(), f, indent=4)


if __name__ == "__main__":
    accumulate_paddings(
        stitched_section_dir="/tungstenfs/temp/generic/stitched-sections/",
        end_section=1088,
    )
