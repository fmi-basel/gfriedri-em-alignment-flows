import argparse
from os.path import dirname, join

import yaml
import zarr
from numcodecs import Blosc
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
from parameter_config import WarpConfig
from sbem.record.Section import Section
from sbem.tile_stitching.sofima_utils import render_tiles
from tqdm import tqdm


def warp_tiles(output_dir: str, mesh_file: str, stride: int, warp_config: WarpConfig):
    section = Section.load_from_yaml(mesh_file.replace("meshes.npz", "section.yaml"))
    warped_tiles, mask = render_tiles(
        section=section,
        section_dir=dirname(mesh_file),
        stride=stride,
        margin=warp_config.margin,
        parallelism=warp_config.warp_parallelism,
        use_clahe=warp_config.use_clahe,
        clahe_kwargs={
            "kernel_size": warp_config.kernel_size,
            "clip_limit": warp_config.clip_limit,
            "nbins": warp_config.nbins,
        },
    )

    path = join(output_dir, section.get_name() + ".zarr")
    store = parse_url(path, mode="w").store
    write_image(
        image=warped_tiles,
        group=zarr.group(store=store),
        scaler=Scaler(max_layer=0),
        axes="yx",
        storage_options=dict(
            chunks=(2744, 2744),
            compressor=Blosc(
                cname="zstd",
                clevel=3,
                shuffle=Blosc.SHUFFLE,
            ),
            overwrite=True,
            write_empty_chunks=False,
        ),
    )
    return path


def main(mesh_files: list[str], output_dir: str, stride: int, warp_config: WarpConfig):
    from s02_register_tiles import filter_ignore

    mesh_files = filter_ignore(mesh_files, file_name="meshes.npz")

    warped_tiles = []
    for mesh_file in tqdm(mesh_files):
        warped_tiles.append(
            warp_tiles(
                output_dir=output_dir,
                mesh_file=mesh_file,
                stride=stride,
                warp_config=warp_config,
            )
        )

    with open("warped_tiles.yaml", "w") as f:
        yaml.safe_dump(warped_tiles, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="tile-stitching.config")
    parser.add_argument("--meshes", type=str, default="meshes.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    with open(args.meshes) as f:
        mesh_files = yaml.safe_load(f)

    main(
        mesh_files=mesh_files,
        output_dir=join(config["output_dir"], "stitched-sections"),
        stride=config["mesh_integration_config"]["stride"],
        warp_config=WarpConfig(**config["warp_config"]),
    )