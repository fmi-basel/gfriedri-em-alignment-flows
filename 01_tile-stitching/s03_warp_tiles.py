from glob import glob
from os.path import dirname, join

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


def main(output_dir: str, section_dir: str, stride: int, warp_config: WarpConfig):
    from s02_register_tiles import filter_ignore

    mesh_files = glob(join(section_dir, "*", "meshes.npz"))
    mesh_files = filter_ignore(mesh_files, file_name="meshes.npz")

    for mesh_file in tqdm(mesh_files):
        warp_tiles(
            output_dir=output_dir,
            mesh_file=mesh_file,
            stride=stride,
            warp_config=warp_config,
        )


if __name__ == "__main__":
    main(
        output_dir="/home/tibuch/Data/gfriedri/2023-refactor/stitched-sections",
        section_dir="/home/tibuch/Data/gfriedri/2023-refactor/sections",
        stride=20,
        warp_config=WarpConfig(),
    )
