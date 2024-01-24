import argparse
import logging
from glob import glob
from os.path import basename, join, splitext

import jax.numpy as jnp
import numpy as np
import yaml
import zarr
from connectomics.common import bounding_box
from numcodecs import Blosc
from ome_zarr.io import parse_url
from parameter_config import MeshIntegrationConfig
from s01_estimate_flow_fields import list_zarr_sections
from sofima import map_utils, mesh


def section_name(dir: str) -> str:
    return splitext(basename(dir))[0]


def create_map_storage(
    output_dir: str,
    shape: tuple[int, int],
    n_sections: int,
    block_size: int,
) -> None:
    store = parse_url(path=join(output_dir, "maps.zarr"), mode="w").store
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


def mesh_optimization(
    section_dirs: list[str],
    start_section: int,
    block_index: int,
    map_zarr: zarr.Group,
    stride: int,
    integration_config: MeshIntegrationConfig,
    logger: logging.Logger = logging.getLogger("Mesh Optimization"),
):
    config = mesh.IntegrationConfig(
        dt=integration_config.dt,
        gamma=integration_config.gamma,
        k0=integration_config.k0,
        k=integration_config.k,
        stride=stride,
        num_iters=integration_config.num_iters,
        max_iters=integration_config.max_iters,
        stop_v_max=integration_config.stop_v_max,
        dt_max=integration_config.dt_max,
        start_cap=integration_config.start_cap,
        final_cap=integration_config.final_cap,
        prefer_orig_order=integration_config.prefer_orig_order,
    )
    final_flow = []
    for section in section_dirs[1:]:
        ff_path = glob(join(section, "final_flow_*.npy"))
        if len(ff_path) > 0:
            logger.info(f"Loading final flow from {basename(section)}.")
            final_flow.append(np.load(ff_path[0]))

    final_flow = np.concatenate(final_flow, axis=1)
    logger.info(f"Final flow shape: {final_flow.shape}")
    origin = jnp.array([0.0, 0.0])

    solved = map_zarr["main"]
    inv_map = map_zarr["main_inv"]
    cross_block_flow = map_zarr["cross_block_flow"]
    last_inv = map_zarr["last_inv"]
    x = np.zeros_like(solved[:, start_section : start_section + 1, ...])
    for z in range(0, len(section_dirs) - 1):

        logger.info(f"z = {z}")
        prev = map_utils.compose_maps_fast(
            final_flow[:, z : z + 1, ...],
            origin,
            stride,
            x,
            origin,
            stride,
        )
        x = np.zeros_like(solved[:, start_section + z : start_section + z + 1, ...])
        x, e_kin, num_steps = mesh.relax_mesh(x, prev, config)
        x = np.array(x)
        map_box = bounding_box.BoundingBox(start=(0, 0, 0), size=x.shape[1:][::-1])
        if z < len(section_dirs) - 2:
            logger.info(f"Writing to main[{start_section + z + 1}].")
            solved[:, start_section + z + 1 : start_section + z + 2] = x
            inv_map[
                :, start_section + z + 1 : start_section + z + 2
            ] = map_utils.invert_map(x, map_box, map_box, stride)
        else:
            if start_section + z + 1 == solved.shape[1] - 1:
                logger.info(f"Writing to main[{start_section + z + 1}].")
                solved[:, start_section + z + 1 : start_section + z + 2] = x

            logger.info(f"Writing to cross_block[{block_index}].")
            cross_block_flow[:, block_index : block_index + 1] = x
            last_inv[
                :, start_section + z + 1 : start_section + z + 2
            ] = map_utils.invert_map(x, map_box, map_box, stride)


def relax_meshes_in_blocks(
    section_dirs: list[str],
    section_offset: int,
    block_index_offset: int,
    output_dir: str,
    mesh_integration: MeshIntegrationConfig = MeshIntegrationConfig(),
    flow_stride: int = 40,
    logger: logging.Logger = logging.getLogger("Relax Meshes"),
):
    store = parse_url(path=join(output_dir, "maps.zarr"), mode="w").store
    map_zarr: zarr.Group = zarr.group(store=store)
    for block_index, i in enumerate(
        range(0, len(section_dirs) - 1, mesh_integration.block_size)
    ):
        start = i
        end = min(start + mesh_integration.block_size + 1, len(section_dirs))
        start_name = section_name(section_dirs[start])
        end_name = section_name(section_dirs[end - 1])
        logger.info(f"Optimize meshes in block [{start_name}:{end_name}].")
        mesh_optimization(
            section_dirs=section_dirs[start:end],
            start_section=section_offset + start,
            block_index=block_index_offset + block_index,
            map_zarr=map_zarr,
            stride=flow_stride,
            integration_config=mesh_integration,
            logger=logger,
        )


def relax_meshes_cross_blocks(
    output_dir: str,
    mesh_integration: MeshIntegrationConfig,
    flow_stride: int,
    logger: logging.Logger,
):
    store = parse_url(path=join(output_dir, "maps.zarr"), mode="w").store
    map_zarr: zarr.Group = zarr.group(store=store)
    cross_block_flow = map_zarr["cross_block_flow"]
    cross_block_map = map_zarr["cross_block"]
    cross_block_inv_map = map_zarr["cross_block_inv"]
    main_map_size = map_zarr["main"].shape[1:][::-1]
    map_box = bounding_box.BoundingBox(start=(0, 0, 0), size=main_map_size)
    map2x_box = map_box.scale(0.5)
    xblk_stride = flow_stride * 2

    x_block_flow = map_utils.resample_map(
        cross_block_flow, map_box, map2x_box, flow_stride, xblk_stride
    )

    xblk_config = mesh.IntegrationConfig(
        dt=mesh_integration.dt,
        gamma=mesh_integration.gamma,
        k0=0.001,
        k=mesh_integration.k,
        stride=xblk_stride,
        num_iters=mesh_integration.num_iters,
        max_iters=mesh_integration.max_iters,
        stop_v_max=mesh_integration.stop_v_max,
        dt_max=mesh_integration.dt_max,
        start_cap=mesh_integration.start_cap,
        final_cap=mesh_integration.final_cap,
        prefer_orig_order=mesh_integration.prefer_orig_order,
    )
    logger.info(f"{x_block_flow.shape[1]} cross block flows to " f"solve.")
    origin = jnp.array([0.0, 0.0])
    xblk = []
    for z in range(x_block_flow.shape[1]):
        logger.info(f"Solving cross block flow {z}.")
        if z == 0:
            prev = x_block_flow[:, z : z + 1, ...]
        else:
            prev = map_utils.compose_maps_fast(
                x_block_flow[:, z : z + 1, ...],
                origin,
                xblk_stride,
                xblk[-1],
                origin,
                xblk_stride,
            )
        x = np.zeros_like(x_block_flow[:, 0:1, ...])
        x, e_kin, num_steps = mesh.relax_mesh(x, prev, xblk_config)
        x = np.array(x)
        xblk.append(x)

    xblk = np.concatenate(xblk, axis=1)
    logger.info("Resample cross block map.")
    cross_block_map[:, :, ...] = map_utils.resample_map(
        xblk, map2x_box, map_box, flow_stride * 2, flow_stride
    )
    logger.info("Invert cross block map.")
    cross_block_inv_map[:, :, ...] = map_utils.invert_map(
        cross_block_map[:], map_box, map_box, flow_stride
    )


def relax_meshes(
    stitched_section_dir: str = "",
    output_dir: str = "",
    mesh_integration: MeshIntegrationConfig = MeshIntegrationConfig(),
    flow_stride: int = 40,
):
    section_dirs = list_zarr_sections(root_dir=stitched_section_dir)

    dummy_flow = np.load(glob(join(section_dirs[1], "final_flow_*.npy"))[0])
    create_map_storage(
        output_dir=output_dir,
        shape=dummy_flow.shape[2:],
        n_sections=len(section_dirs) + 1,
        block_size=mesh_integration.block_size,
    )

    relax_meshes_in_blocks(
        section_dirs=section_dirs,
        section_offset=0,
        block_index_offset=0,
        output_dir=output_dir,
        mesh_integration=mesh_integration,
        flow_stride=flow_stride,
    )

    relax_meshes_cross_blocks(
        output_dir=output_dir,
        mesh_integration=mesh_integration,
        flow_stride=flow_stride,
        logger=logging.Logger("relax meshes"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="relax_meshes.config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    relax_meshes(
        stitched_section_dir=config["stitched_sections_dir"],
        output_dir=config["output_dir"],
        mesh_integration=MeshIntegrationConfig(**config["mesh_integration"]),
        flow_stride=config["flow_stride"],
    )
