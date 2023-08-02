import json
from os.path import basename, join

import numpy as np
import zarr
from numcodecs import Blosc
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
from s01_coarse_align_section_pairs import list_zarr_sections
from skimage.measure import block_reduce
from tqdm import tqdm


def load_shifts(section_dirs: list[str]):
    shifts = []
    for sec in section_dirs[1:]:
        with open(join(sec, "shift_to_previous.json")) as f:
            data = json.load(f)
            shifts.append([data["shift_y"], data["shift_x"]])

    return np.array(shifts)


def get_padding_per_section(shifts):
    cumulated_shifts = np.cumsum(shifts, axis=0)
    cumulated_padding = cumulated_shifts + np.abs(np.min(cumulated_shifts, axis=0))
    return np.concatenate([np.array([[0, 0]]), cumulated_padding], 0)


def write_section(
    section_dir: str,
    padding,
    start_section: int,
    yx_start: tuple[int, int],
    yx_size: tuple[int, int],
    bin: int,
    zarr_root: zarr.Group,
):
    scaler = Scaler(max_layer=4)
    sec_id = int(basename(section_dir).split("_")[0][1:])
    output_index = sec_id - start_section
    current = zarr.Group(parse_url(section_dir).store)
    y_pad, x_pad = padding
    data = block_reduce(
        current[0][
            yx_start[0] : yx_start[0] + yx_size[0] - y_pad,
            yx_start[1] : yx_start[1] + yx_size[1] - x_pad,
        ],
        block_size=bin,
        func=np.mean,
    ).astype(np.uint8)

    for level in range(scaler.max_layer + 1):
        y_start = y_pad // bin // (scaler.downscale**level)
        y_end = y_start + data.shape[0]
        x_start = x_pad // bin // (scaler.downscale**level)
        x_end = x_start + data.shape[1]
        zarr_root[level][
            output_index,
            y_start:y_end,
            x_start:x_end,
        ] = data
        data = scaler.resize_image(data)


def create_zarr(
    output_dir: str,
    volume_name: str,
    start_section: int,
    end_section: int,
    yx_size: tuple[int, int],
    bin: int,
):
    target_dir = join(output_dir, volume_name)
    store = parse_url(target_dir, mode="w").store
    zarr_root = zarr.group(store=store)

    chunks = (1, 2744, 2744)
    write_image(
        image=np.zeros(
            (end_section - start_section, yx_size[0] // bin, yx_size[1] // bin),
            dtype=np.uint8,
        ),
        group=zarr_root,
        axes="zyx",
        scaler=Scaler(max_layer=4),
        storage_options=dict(
            chunks=chunks,
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            overwrite=True,
            write_empty_chunks=False,
        ),
    )

    return target_dir


def main(
    stitched_section_dir: str,
    output_dir: str,
    volume_name: str,
    start_section: int,
    end_section: int,
    yx_start: tuple[int, int],
    yx_size: tuple[int, int],
    bin: int,
):
    assert yx_size[0] % bin == 0, "yx_size must be divisible by bin."
    assert yx_size[1] % bin == 0, "yx_size must be divisible by bin."
    section_dirs = list_zarr_sections(
        root_dir=stitched_section_dir,
    )

    shifts = load_shifts(section_dirs=section_dirs)

    section_paddings = get_padding_per_section(shifts)

    zarr_path = create_zarr(
        output_dir=output_dir,
        volume_name=volume_name,
        start_section=start_section,
        end_section=end_section,
        yx_size=yx_size,
        bin=bin,
    )

    store = parse_url(zarr_path, mode="w").store
    zarr_root = zarr.group(store=store)

    for i in tqdm(range(len(section_dirs) - 1)):
        sec_id = int(basename(section_dirs[i]).split("_")[0][1:])
        if start_section <= sec_id < end_section:
            write_section(
                section_dir=section_dirs[i],
                padding=section_paddings[i],
                start_section=start_section,
                yx_start=yx_start,
                yx_size=yx_size,
                bin=bin,
                zarr_root=zarr_root,
            )


if __name__ == "__main__":
    main(
        stitched_section_dir="/tungstenfs/scratch/gmicro_sem/gfriedri/_processing/SOFIMA/user/ganctoma/runs/test/output/stitched-sections",
        start_section=1074,
        end_section=1098,
        output_dir=".",
        volume_name="test.zarr",
        yx_start=(16000, 0),
        yx_size=(4096 * 4, 4096 * 4),
        bin=4,
    )
