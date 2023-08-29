import json
import os
import re
from os.path import basename, join
from pathlib import Path

import numpy as np
import zarr
from ome_zarr.io import parse_url
from sofima import stitch_rigid
from tqdm import tqdm

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"


def compute_extent(s1, s2):
    y_size = max(s1.shape[0], s2.shape[0])
    x_size = max(s1.shape[1], s2.shape[1])
    return y_size, x_size


def pad_moving(moving, extent):
    padded = np.zeros(extent, dtype=moving.dtype)
    y_offset = (extent[0] - moving.shape[0]) // 2
    x_offset = (extent[1] - moving.shape[1]) // 2
    padded[
        y_offset : y_offset + moving.shape[0], x_offset : x_offset + moving.shape[1]
    ] = moving
    return padded, (y_offset, x_offset)


def pad_target(target, extent):
    padded = np.zeros(extent, dtype=target.dtype)
    padded[: target.shape[0], : target.shape[1]] = target
    return padded


def compute_coarse_alignment(s1, s2):
    bin = max(*s1[0].shape, *s2[0].shape) // 1024

    target = s1[0][::bin, ::bin]
    moving = s2[0][::bin, ::bin]

    extent = compute_extent(target, moving)
    target_padded = pad_target(target, extent=extent)
    moving_padded, moving_offsets = pad_moving(moving, extent=extent)

    [shift_x, shift_y], _ = stitch_rigid._estimate_offset(
        target_padded, moving_padded, 0, 10
    )
    final_shift_y = (shift_y + moving_offsets[0]) * bin
    final_shift_x = (shift_x + moving_offsets[1]) * bin
    return tuple([int(final_shift_y), int(final_shift_x)]), bin


def refine_coarse_alignment(s1, s2, s2_shift, coarse_bin):
    size = 2048
    min_y_size = min(s1[0].shape[0], s2[0].shape[0])
    min_x_size = min(s2[0].shape[1], s2[0].shape[1])

    step_y = min_y_size // 3
    offset_y = max(0, step_y // 2 - size // 2)
    step_x = min_x_size // 3
    offset_x = max(0, step_x // 2 - size // 2)

    shifts_yx = []
    for y in range(offset_y, min_y_size, step_y):
        for x in range(offset_x, min_x_size, step_x):
            target_start_y = min(y, s1[0].shape[0] - size)
            moving_start_y = target_start_y + s2_shift[0]
            target_start_x = min(x, s1[0].shape[1] - size)
            moving_start_x = target_start_x + s2_shift[1]

            target_crop_in_bound = (
                target_start_y >= 0
                and target_start_y + size < s1[0].shape[0]  # noqa: W503
                and target_start_x >= 0  # noqa: W503
                and target_start_x + size < s1[0].shape[1]  # noqa: W503
            )
            moving_crop_in_bound = (
                moving_start_y >= 0
                and moving_start_y + size < s2[0].shape[0]  # noqa: W503
                and moving_start_x >= 0  # noqa: W503
                and moving_start_x + size < s2[0].shape[1]  # noqa: W503
            )

            if target_crop_in_bound and moving_crop_in_bound:
                target_crop = s1[0][
                    target_start_y : target_start_y + size,
                    target_start_x : target_start_x + size,
                ]
                moving_crop = s2[0][
                    moving_start_y : moving_start_y + size,
                    moving_start_x : moving_start_x + size,
                ]

                if target_crop.sum() > 0 and moving_crop.sum() > 0:
                    [shift_x, shift_y], _ = stitch_rigid._estimate_offset(
                        target_crop, moving_crop, 100, 10
                    )
                    if abs(shift_y) <= coarse_bin and abs(shift_x) <= coarse_bin:
                        shifts_yx.append([shift_y, shift_x])

    if len(shifts_yx) == 0:
        # No shift if all patches got rejected.
        shifts_yx.append([0, 0])

    return tuple(np.mean(shifts_yx, axis=0).astype(int))


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


def compute_shift(current_section_dir: str, next_section_dir: str):
    current = zarr.Group(parse_url(current_section_dir).store)
    next = zarr.Group(parse_url(next_section_dir).store)
    coarse_shift, coarse_bin = compute_coarse_alignment(current, next)
    shift_y, shift_x = refine_coarse_alignment(
        current, next, s2_shift=coarse_shift, coarse_bin=coarse_bin
    )

    with open(join(next_section_dir, "shift_to_previous.json"), "w") as f:
        json.dump(dict(shift_y=int(shift_y), shift_x=int(shift_x)), f)


def main(stitched_section_dir: str):
    section_dirs = list_zarr_sections(
        root_dir=stitched_section_dir,
    )

    for i in tqdm(range(len(section_dirs) - 1)):
        compute_shift(
            current_section_dir=section_dirs[i], next_section_dir=section_dirs[i + 1]
        )


if __name__ == "__main__":
    main(
        stitched_section_dir="/tungstenfs/scratch/gmicro_sem/gfriedri/_processing/SOFIMA/user/ganctoma/runs/test/output/stitched-sections",
    )
