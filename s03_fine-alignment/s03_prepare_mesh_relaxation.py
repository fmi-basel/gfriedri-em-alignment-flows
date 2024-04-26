import argparse
from glob import glob
from os.path import basename, dirname, join

import numpy as np
import yaml
import zarr
from numcodecs import Blosc
from ome_zarr.io import parse_url
from parameter_config import MeshIntegrationConfig


def main(
    stitched_section_dirs: list[str],
    output_dir: str,
    mesh_integration: MeshIntegrationConfig,
):
    dummy_flow = np.load(glob(join(stitched_section_dirs[1], "final_flow_*.npy"))[0])

    path = create_map_storage(
        output_dir=output_dir,
        shape=dummy_flow.shape[2:],
        n_sections=len(stitched_section_dirs) + 1,
        block_size=mesh_integration.block_size,
    )

    for i, chunk_start in enumerate(
        range(0, len(stitched_section_dirs), mesh_integration.block_size)
    ):
        block_relaxation_config = dict(
            map_path=path,
            section_dirs=stitched_section_dirs[
                chunk_start : chunk_start + mesh_integration.block_size + 1
            ],
            section_offset=chunk_start,
            block_index_offset=i,
        )
        with open(f"mesh_relaxation_blocks_{i}.yaml", "w") as f:
            yaml.safe_dump(block_relaxation_config, f, sort_keys=False)


def create_map_storage(
    output_dir: str,
    shape: tuple[int, int],
    n_sections: int,
    block_size: int,
) -> str:
    path = join(output_dir, "maps.zarr")
    store = parse_url(path=path, mode="w").store
    map_zarr: zarr.Group = zarr.group(store=store)

    if "main" not in map_zarr:
        map_zarr.create_dataset(
            name="main",
            shape=(2, n_sections, *shape),
            chunks=(2, 1, *shape),
            dtype="<f4",
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            fill_value=0,
            overwrite=True,
        )
    if "main_inv" not in map_zarr:
        map_zarr.create_dataset(
            name="main_inv",
            shape=(2, n_sections, *shape),
            chunks=(2, 1, *shape),
            dtype="<f4",
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            fill_value=0,
            overwrite=True,
        )
    if "cross_block_flow" not in map_zarr:
        map_zarr.create_dataset(
            name="cross_block_flow",
            shape=(2, n_sections // block_size + 1, *shape),
            chunks=(2, 1, *shape),
            dtype="<f4",
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            fill_value=0,
            overwrite=True,
        )
    if "cross_block" not in map_zarr:
        map_zarr.create_dataset(
            name="cross_block",
            shape=(2, n_sections // block_size + 1, *shape),
            chunks=(2, 1, *shape),
            dtype="<f4",
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            fill_value=0,
            overwrite=True,
        )
    if "cross_block_inv" not in map_zarr:
        map_zarr.create_dataset(
            name="cross_block_inv",
            shape=(2, n_sections // block_size + 1, *shape),
            chunks=(2, 1, *shape),
            dtype="<f4",
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            fill_value=0,
            overwrite=True,
        )
    if "last_inv" not in map_zarr:
        map_zarr.create_dataset(
            name="last_inv",
            shape=(2, n_sections, *shape),
            chunks=(2, 1, *shape),
            dtype="<f4",
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            fill_value=0,
            overwrite=True,
        )

    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, default="fine-align.config"
    )
    parser.add_argument(
        "--flow_paths", type=str, required=True, default="flow_paths.yaml"
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    with open(args.flow_paths) as f:
        flow_paths = sorted(
            list(set(yaml.safe_load(f))),
            key=lambda v: int(basename(dirname(v)).split("_")[0][1:]),
        )

    stitched_section_dirs = [dirname(p) for p in flow_paths]
    section, grid = basename(stitched_section_dirs[0]).split("_")
    first_section = int(section[1:]) - 1
    stitched_section_dirs = [
        join(
            dirname(stitched_section_dirs[0]),
            f"s{str(first_section).zfill(len(section) - 1)}_{grid}",
        ),
    ] + stitched_section_dirs

    main(
        stitched_section_dirs=stitched_section_dirs,
        output_dir=config["output_dir"],
        mesh_integration=MeshIntegrationConfig(**config["mesh_integration"]),
    )
