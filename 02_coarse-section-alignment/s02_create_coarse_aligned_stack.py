import json
from os.path import basename, join

import numpy as np
import zarr
from numcodecs import Blosc
from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from s01_coarse_align_section_pairs import list_zarr_sections
from skimage.measure import block_reduce
from tqdm import tqdm


def load_padding(section_dir: str) -> tuple[int, int]:
    with open(join(section_dir, "coarse_stack_padding.json")) as f:
        config = json.load(f)
        shift_y = config["shift_y"]
        shift_x = config["shift_x"]

    return shift_y, shift_x


def write_section(
    section_dir: str,
    start_section: int,
    yx_size: tuple[int, int],
    bin: int,
    zarr_root: zarr.Group,
):
    scaler = Scaler(max_layer=4)
    sec_id = int(basename(section_dir).split("_")[0][1:])
    output_index = sec_id - start_section
    current = zarr.Group(parse_url(section_dir).store)
    y_pad, x_pad = load_padding(section_dir)
    data = block_reduce(
        current[0][
            : yx_size[0] - y_pad,
            : yx_size[1] - x_pad,
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

    datasets = []
    shapes = []
    for path, level in enumerate(range(5)):
        downscale = 2**level
        # Downscale only in YX
        shape = (
            end_section - start_section,
            yx_size[0] // bin // downscale,
            yx_size[1] // bin // downscale,
        )
        zarr_root.create_dataset(
            name=str(path),
            shape=shape,
            chunks=(1, 2744, 2744),
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            overwrite=True,
            write_empty_chunks=False,
            fill_value=0,
            dtype=np.uint8,
            dimension_separator="/",
        )

        datasets.append({"path": str(path)})
        shapes.append(shape)

    fmt = CurrentFormat()
    coordinate_transformations = fmt.generate_coordinate_transformations(shapes)

    fmt.validate_coordinate_transformations(
        ndim=3,
        nlevels=len(shapes),
        coordinate_transformations=coordinate_transformations,
    )
    for dataset, transform in zip(datasets, coordinate_transformations):
        dataset["coordinateTransformations"] = transform

    from ome_zarr import writer

    writer.write_multiscales_metadata(
        group=zarr_root,
        datasets=datasets,
        axes="zyx",
    )

    return target_dir


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


def main(
    stitched_section_dir: str,
    output_dir: str,
    volume_name: str,
    start_section: int,
    end_section: int,
    bin: int,
):
    section_dirs = list_zarr_sections(
        root_dir=stitched_section_dir,
    )

    yx_size = get_yx_size(section_dirs, bin=bin)

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
                start_section=start_section,
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
