import argparse
import json
from os.path import join

import numpy as np
import yaml
import zarr
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
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
    out_z: int,
    yx_size: tuple[int, int],
    bin: int,
    zarr_root: zarr.Group,
):
    scaler = Scaler(max_layer=4)
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
            out_z,
            y_start:y_end,
            x_start:x_end,
        ] = data
        data = scaler.resize_image(data)


def main(
    chunk: list[tuple[int, str]],
    zarr_path: str,
    bin: int,
):
    store = parse_url(zarr_path, mode="w").store
    zarr_root = zarr.group(store=store)

    warped_sections = []
    for i in tqdm(range(len(chunk))):
        write_section(
            section_dir=chunk[i][1],
            out_z=chunk[i][0],
            yx_size=zarr_root[0].shape[1:],
            bin=bin,
            zarr_root=zarr_root,
        )
        warped_sections.append(chunk[i][1])

    with open("warped_sections.yaml", "w") as f:
        yaml.safe_dump(warped_sections, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, default="coarse-stack.config"
    )
    parser.add_argument("--zarr_dir", type=str, required=True, default="zarr_dir.yaml")
    parser.add_argument(
        "--chunks", type=str, required=True, default="processed_chunks_0.yaml"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    with open(args.zarr_dir) as f:
        zarr_dir = yaml.safe_load(f)[0]

    with open(args.chunks) as f:
        chunk = yaml.safe_load(f)

    main(
        chunk=chunk,
        zarr_path=zarr_dir,
        bin=config["bin"],
    )
