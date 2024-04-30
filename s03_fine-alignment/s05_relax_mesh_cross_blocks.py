import argparse
from os.path import join

import jax.numpy as jnp
import numpy as np
import yaml
import zarr
from connectomics.common import bounding_box
from ome_zarr.io import parse_url
from parameter_config import MeshIntegrationConfig
from s04_relax_mesh_blocks import create_logger
from sofima import map_utils, mesh


def main(
    map_path: str,
    mesh_integration: MeshIntegrationConfig,
    flow_stride: int,
):
    logger = create_logger("relax-meshes-cross-blocks")
    store = parse_url(path=map_path, mode="w").store
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
    logger.info(f"{x_block_flow.shape[1]} cross block flows to solve.")
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

    with open("map.yaml", "w") as f:
        yaml.safe_dump(
            dict(
                map_path=map_path,
            ),
            f,
            sort_keys=False,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, default="relax-meshes.yaml"
    )
    parser.add_argument(
        "--relaxed_blocks",
        type=str,
        required=True,
        default="relaxed_blocks.yaml",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    with open(args.relaxed_blocks) as f:
        blocks = yaml.safe_load(f)

    main(
        map_path=join(config["output_dir"], "maps.zarr"),
        mesh_integration=MeshIntegrationConfig(**config["mesh_integration"]),
        flow_stride=config["flow_stride"],
    )
