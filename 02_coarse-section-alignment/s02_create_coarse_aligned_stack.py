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

    store = parse_url(join(output_dir, volume_name), mode="w").store
    zarr_root = zarr.group(store=store)

    write_image(
        image=np.zeros(
            (end_section - start_section, yx_size[0] // bin, yx_size[1] // bin),
            dtype=np.uint8,
        ),
        group=zarr_root,
        axes="zyx",
        scaler=Scaler(max_layer=0),
        storage_options=dict(
            chunks=(1, 2744, 2744),
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            overwrite=True,
            write_empty_chunks=False,
        ),
    )

    for i in tqdm(range(len(section_dirs) - 1)):
        sec_id = int(basename(section_dirs[i]).split("_")[0][1:])
        if start_section <= sec_id < end_section:
            output_index = sec_id - start_section
            current = zarr.Group(parse_url(section_dirs[i]).store)
            y_pad, x_pad = section_paddings[i]
            data = block_reduce(
                current[0][
                    yx_start[0] : yx_start[0] + yx_size[0] - y_pad,
                    yx_start[1] : yx_start[1] + yx_size[1] - x_pad,
                ],
                block_size=bin,
                func=np.mean,
            ).astype(np.uint8)
            zarr_root[0][
                output_index,
                y_pad // bin : y_pad // bin + data.shape[0],
                x_pad // bin : x_pad // bin + data.shape[1],
            ] = data


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
