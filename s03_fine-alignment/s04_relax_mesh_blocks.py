import argparse
import logging
from datetime import datetime
from glob import glob
from os.path import basename, join, splitext

import jax.numpy as jnp
import numpy as np
import yaml
import zarr
from connectomics.common import bounding_box
from ome_zarr.io import parse_url
from parameter_config import MeshIntegrationConfig
from sofima import map_utils, mesh


def relax_meshes_in_blocks(
    section_dirs: list[str],
    section_offset: int,
    block_index_offset: int,
    map_path: str,
    mesh_integration: MeshIntegrationConfig = MeshIntegrationConfig(),
    flow_stride: int = 40,
):
    logger = create_logger("relax-meshes-in-block")
    store = parse_url(path=map_path, mode="w").store
    map_zarr: zarr.Group = zarr.group(store=store)
    mesh_optimization(
        section_dirs=section_dirs,
        start_section=section_offset,
        block_index=block_index_offset,
        map_zarr=map_zarr,
        stride=flow_stride,
        integration_config=mesh_integration,
        logger=logger,
    )
    with open("relaxed_meshes_in_blocks.yaml", "w") as f:
        yaml.safe_dump(section_dirs[:-1], f, sort_keys=False)


def create_logger(name: str) -> logging.Logger:
    """
    Create logger which logs to <timestamp>-<name>.log inside the current
    working directory.

    Parameters
    ----------
    name
        Name of the logger instance.
    """
    logger = logging.Logger(name.capitalize())
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    handler = logging.FileHandler(f"{now}-{name}.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def section_name(dir: str) -> str:
    return splitext(basename(dir))[0]


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, default="relax-meshes.yaml"
    )
    parser.add_argument(
        "--block_sections",
        type=str,
        required=True,
        default="mesh_relaxation_blocks_0.yaml",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    with open(args.block_sections) as f:
        blocks = yaml.safe_load(f)

    relax_meshes_in_blocks(
        section_dirs=blocks["section_dirs"],
        section_offset=blocks["section_offset"],
        block_index_offset=blocks["block_index_offset"],
        map_path=blocks["map_path"],
        mesh_integration=MeshIntegrationConfig(**config["mesh_integration"]),
        flow_stride=config["flow_stride"],
    )
