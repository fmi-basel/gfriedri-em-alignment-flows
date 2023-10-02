import logging
from glob import glob
from os.path import join

import jax.numpy as jnp
import numpy as np
import zarr
from connectomics.common import bounding_box
from numcodecs import Blosc
from ome_zarr.io import parse_url
from parameter_config import MeshIntegrationConfig
from s01_accumulate_section_paddings import filter, list_zarr_sections
from sofima import map_utils, mesh


def create_map_storage(
    output_dir: str,
    shape: tuple[int, int],
    n_sections: int,
    block_size: int,
):
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

    return map_zarr


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
    for section in section_dirs:
        ff_path = glob(join(section, "final_flow_*.npy"))[0]
        final_flow.append(np.load(ff_path))

    origin = jnp.array([0.0, 0.0])

    solved = map_zarr["main"]
    inv_map = map_zarr["main_inv"]
    cross_block_flow = map_zarr["cross_block_flow"]
    last_inv = map_zarr["last_inv"]
    for z in range(0, final_flow.shape[1]):
        prev_solved = solved[:, start_section + z : start_section + z + 1, ...]

        prev = map_utils.compose_maps_fast(
            final_flow[:, z : z + 1, ...],
            origin,
            stride,
            prev_solved,
            origin,
            stride,
        )
        x = np.zeros_like(solved[:, start_section + z : start_section + z + 1, ...])
        logger.info(f"Relaxing mesh for section {start_section + z + 1}.")
        x, e_kin, num_steps = mesh.relax_mesh(x, prev, config)
        x = np.array(x)
        map_box = bounding_box.BoundingBox(start=(0, 0, 0), size=x.shape[1:][::-1])
        if z + 1 < final_flow.shape[1]:
            solved[:, start_section + z + 1 : start_section + z + 2] = x
            inv_map[
                :, start_section + z + 1 : start_section + z + 2
            ] = map_utils.invert_map(x, map_box, map_box, stride)
        else:
            if start_section + z + 1 == solved.shape[1] - 1:
                solved[:, start_section + z + 1 : start_section + z + 2] = x
            cross_block_flow[:, block_index : block_index + 1] = x
            last_inv[
                :, start_section + z + 1 : start_section + z + 2
            ] = map_utils.invert_map(x, map_box, map_box, stride)


def relax_meshes_in_blocks(
    section_dirs: list[str],
    map_zarr: zarr.Group,
    integration_config: MeshIntegrationConfig = MeshIntegrationConfig(),
    flow_stride: int = 40,
):
    for block_index, i in enumerate(
        range(0, len(section_dirs), integration_config.block_size)
    ):
        start = i
        end = min(start + integration_config.block_size, len(section_dirs))

        mesh_optimization(
            section_dirs=section_dirs[start:end],
            start_section=start,
            block_index=block_index,
            map_zarr=map_zarr,
            stride=flow_stride,
            integration_config=integration_config,
        )


def relax_meshes_cross_blocks(
    map_zarr: zarr.Group, integration_config: MeshIntegrationConfig, flow_stride: int
):
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
        dt=integration_config.dt,
        gamma=integration_config.gamma,
        k0=0.001,
        k=integration_config.k,
        stride=xblk_stride,
        num_iters=integration_config.num_iters,
        max_iters=integration_config.max_iters,
        stop_v_max=integration_config.stop_v_max,
        dt_max=integration_config.dt_max,
        start_cap=integration_config.start_cap,
        final_cap=integration_config.final_cap,
        prefer_orig_order=integration_config.prefer_orig_order,
    )
    origin = jnp.array([0.0, 0.0])
    xblk = []
    for z in range(x_block_flow.shape[1]):
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
    cross_block_map[:, :, ...] = map_utils.resample_map(
        xblk, map2x_box, map_box, flow_stride * 2, flow_stride
    )
    cross_block_inv_map[:, :, ...] = map_utils.invert_map(
        cross_block_map[:], map_box, map_box, flow_stride
    )


def relax_meshes(
    stitched_section_dir: str = "",
    output_dir: str = "",
    start_section: int = 0,
    end_section: int = 9,
    integration_config: MeshIntegrationConfig = MeshIntegrationConfig(),
    flow_stride: int = 40,
):
    section_dirs = list_zarr_sections(root_dir=stitched_section_dir)
    section_dirs = filter(
        section_dirs=section_dirs, start_section=start_section, end_section=end_section
    )

    dummy_flow = np.load(glob(join(section_dirs[1], "final_flow_*.npy"))[0])
    map_zarr = create_map_storage(
        output_dir=output_dir,
        shape=dummy_flow.shape[2:],
        n_sections=len(section_dirs),
        block_size=integration_config.block_size,
    )

    relax_meshes_in_blocks(
        section_dirs=section_dirs,
        map_zarr=map_zarr,
        integration_config=integration_config,
        flow_stride=flow_stride,
    )

    relax_meshes_cross_blocks(
        map_zarr=map_zarr,
        integration_config=integration_config,
        flow_stride=flow_stride,
    )


if __name__ == "__main__":
    relax_meshes()