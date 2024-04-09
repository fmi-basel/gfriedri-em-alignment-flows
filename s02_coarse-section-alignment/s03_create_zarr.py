import argparse
import json
from os.path import basename, join

import numpy as np
import yaml
import zarr
from numcodecs import Blosc
from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url
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
    cumulated_shifts = np.concatenate([np.array([[0, 0]]), cumulated_shifts], 0)
    cumulated_padding = cumulated_shifts + np.abs(np.min(cumulated_shifts, axis=0))
    return cumulated_padding


def create_zarr(
    output_dir: str,
    volume_name: str,
    n_sections: int,
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
            n_sections,
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
    stitched_section_dirs: list[str],
    output_dir: str,
    volume_name: str,
    bin: int,
):
    shifts = load_shifts(stitched_section_dirs)
    paddings = get_padding_per_section(shifts)
    outputs = []
    for i in tqdm(range(len(stitched_section_dirs))):
        padding_file = join(stitched_section_dirs[i], "coarse_stack_padding.json")
        with open(padding_file, "w") as f:
            json.dump(dict(shift_y=int(paddings[i, 0]), shift_x=int(paddings[i, 1])), f)

        outputs.append((i, stitched_section_dirs[i]))

    empty_zarr_path = create_zarr(
        output_dir=output_dir,
        volume_name=volume_name,
        n_sections=len(outputs),
        yx_size=get_yx_size(stitched_section_dirs, bin=bin),
        bin=bin,
    )

    with open("zarr_dir.yaml", "w") as f:
        yaml.safe_dump([empty_zarr_path], f)

    for idx, i in enumerate(range(0, len(stitched_section_dirs) - 10, 10)):
        with open(f"processed_chunks_{idx}.yaml", "w") as f:
            yaml.safe_dump(outputs[i : i + 10], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, default="coarse-align.config"
    )
    parser.add_argument(
        "--coarse_aligned_sections",
        type=str,
        required=True,
        default="processed_dirs_collection.yaml",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    with open(args.coarse_aligned_sections) as f:
        stitched_section_dirs = list(set(yaml.safe_load(f)))

    stitched_section_dirs = sorted(
        stitched_section_dirs, key=lambda x: int(basename(x).split("_")[0][1:])
    )

    main(
        stitched_section_dirs=stitched_section_dirs,
        output_dir=config["output_dir"],
        volume_name=config["volume_name"],
        bin=config["bin"],
    )
