import argparse
from os.path import basename, join

import numpy as np
import yaml
import zarr
from numcodecs import Blosc
from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url
from s01_parse_sections import get_yx_size, list_zarr_sections


def main(
    stitched_sections_dir: str,
    block_size: int,
    output_dir: str,
    volume_name: str,
    warp_start_section: int,
    warp_end_section: int,
    map_zarr_dir: str,
    flow_stride: int,
):
    section_dirs = list_zarr_sections(root_dir=stitched_sections_dir)

    blocks = []
    for i in range(0, len(section_dirs), block_size):
        blocks.append([i, min(len(section_dirs), i + block_size)])

    yx_size = get_yx_size(section_dirs, bin=1)

    target_dir = create_zarr(
        output_dir=output_dir,
        volume_name=volume_name,
        n_sections=len(section_dirs),
        yx_size=yx_size,
        bin=1,
    )

    n_sections_to_process = 0
    for i in range(len(section_dirs)):
        start_id = int(basename(section_dirs[i]).split("_")[0][1:])
        if warp_start_section <= start_id <= warp_end_section:
            n_sections_to_process += 1

    for chunk, i in enumerate(range(0, len(section_dirs), 20)):
        start_id = int(basename(section_dirs[i]).split("_")[0][1:])
        if warp_start_section - 20 <= start_id <= warp_end_section:
            with open(f"sections_for_warping_{chunk}.yaml", "w") as f:
                yaml.safe_dump(
                    dict(
                        section_dirs=section_dirs[i : i + 20],
                        warp_start_section=warp_start_section,
                        warp_end_section=warp_end_section,
                        target_dir=target_dir,
                        yx_size=yx_size,
                        offset=i,
                        blocks=blocks,
                        map_zarr_dir=map_zarr_dir,
                        flow_stride=flow_stride,
                    ),
                    f,
                )


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
    for path, level in enumerate(range(1)):
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, default="fine_alignment_config.yaml"
    )
    parser.add_argument(
        "--warp_config", type=str, required=True, default="warp_config.yaml"
    )
    parser.add_argument(
        "--map",
        type=str,
        required=True,
        default="map.yaml",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    with open(args.warp_config) as f:
        warp_config = yaml.safe_load(f)

    with open(args.map) as f:
        map_path = yaml.safe_load(f)["map_path"]

    main(
        stitched_sections_dir=warp_config["stitched_sections_dir"],
        block_size=config["mi_conf"]["block_size"],
        output_dir=warp_config["output_dir"],
        volume_name=warp_config["volume_name"],
        warp_start_section=warp_config["warp_start_section"],
        warp_end_section=warp_config["warp_end_section"],
        map_zarr_dir=map_path,
        stride=config["ffe_conf"]["stride"],
    )
