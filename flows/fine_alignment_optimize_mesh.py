from glob import glob
from os.path import basename, join
from pathlib import Path

import zarr
from cpr.numpy.NumpyTarget import NumpyTarget
from cpr.Serializer import cpr_serializer
from cpr.utilities.utilities import task_input_hash
from cpr.zarr.ZarrSource import ZarrSource
from numcodecs import Blosc
from ome_zarr.io import parse_url
from prefect import flow, get_client, task
from prefect.client.schemas import FlowRun
from prefect.deployments import run_deployment
from prefect.filesystems import LocalFileSystem
from pydantic import BaseModel

from flows.utils.storage_key import RESULT_STORAGE_KEY


class MeshIntegrationConfig(BaseModel):
    dt: float = 0.001
    gamma: float = 0.0
    k0: float = 0.01
    k: float = 0.1
    num_iters: int = 1000
    max_iters: int = 100000
    stop_v_max: float = 0.005
    dt_max: float = 1000
    start_cap: float = 0.01
    final_cap: float = 10
    prefer_orig_order: bool = True
    block_size: int = 500


class FlowComputationConfig(BaseModel):
    patch_size: int = 160
    stride: int = 40
    batch_size: int = 256
    min_peak_ratio: float = 1.6
    min_peak_sharpness: float = 1.6
    max_magnitude: float = 80
    max_deviation: float = 20
    max_gradient: float = 0
    min_patch_size: int = 400
    chunk_size: int = 100
    parallelization: int = 16


@task(cache_key_fn=task_input_hash, result_storage_key=RESULT_STORAGE_KEY)
def start_block_mesh_optimization(parameters):
    run: FlowRun = run_deployment(
        name="Optimize block mesh/default",
        parameters=parameters,
        client=get_client(),
    )
    return run.state.result()


@flow(
    name="Fine alignment - Optimize mesh",
    persist_result=True,
    result_storage=LocalFileSystem.load("gfriedri-em-alignment-flows-storage"),
    result_serializer=cpr_serializer(),
    cache_result_in_memory=False,
)
def fine_alignment_optimize_mesh(
    start_section: int = 0,
    end_section: int = 1000,
    result_dir: Path = "/path/to/flow_field_storage",
    flow_config: FlowComputationConfig = FlowComputationConfig(),
    integration_config: MeshIntegrationConfig = MeshIntegrationConfig(),
):
    flow_dir = join(result_dir, "flow-fields")
    serialized_final_flows = []
    for i in range(start_section, end_section):
        name = basename(glob(join(flow_dir, f"final_flow_{i}_to" f"_{i+1}-*.npy"))[0])
        hash = name.replace(f"final_flow_{i}_to_{i+1}-", "").replace(".npy", "")
        serialized_final_flows.append(
            {
                "ext": ".npy",
                "name": f"final_flow_{i}_to_{i+1}",
                "location": flow_dir,
                "data_hash": hash,
            }
        )

    # Optimize z-alignment in chunks/blocks
    store = parse_url(path=join(result_dir, "maps.zarr"), mode="w").store
    map_zarr: zarr.Group = zarr.group(store=store)

    tmp = NumpyTarget(**serialized_final_flows[0])
    shape = tmp.get_data().shape[2:]
    map_zarr.create_dataset(
        name="main",
        shape=(2, len(serialized_final_flows) + 1, *shape),
        chunks=(2, 1, *shape),
        dtype="<f4",
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
        fill_value=0,
        overwrite=True,
    )
    map_zarr.create_dataset(
        name="main_inv",
        shape=(2, len(serialized_final_flows) + 1, *shape),
        chunks=(2, 1, *shape),
        dtype="<f4",
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
        fill_value=0,
        overwrite=True,
    )
    map_zarr.create_dataset(
        name="cross_block_flow",
        shape=(
            2,
            len(serialized_final_flows) // integration_config.block_size + 1,
            *shape,
        ),
        chunks=(2, 1, *shape),
        dtype="<f4",
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
        fill_value=0,
        overwrite=True,
    )
    map_zarr.create_dataset(
        name="cross_block",
        shape=(
            2,
            len(serialized_final_flows) // integration_config.block_size + 1,
            *shape,
        ),
        chunks=(2, 1, *shape),
        dtype="<f4",
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
        fill_value=0,
        overwrite=True,
    )
    map_zarr.create_dataset(
        name="cross_block_inv",
        shape=(
            2,
            len(serialized_final_flows) // integration_config.block_size + 1,
            *shape,
        ),
        chunks=(2, 1, *shape),
        dtype="<f4",
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
        fill_value=0,
        overwrite=True,
    )
    map_zarr.create_dataset(
        name="last_inv",
        shape=(2, len(serialized_final_flows) + 1, *shape),
        chunks=(2, 1, *shape),
        dtype="<f4",
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
        fill_value=0,
        overwrite=True,
    )
    main_map_zarr = ZarrSource.from_path(
        join(result_dir, "maps.zarr"), group="main", mode="w"
    )
    main_inv_map_zarr = ZarrSource.from_path(
        join(result_dir, "maps.zarr"), group="main_inv", mode="w"
    )
    cross_block_flow_zarr = ZarrSource.from_path(
        join(result_dir, "maps.zarr"), group="cross_block_flow", mode="w"
    )
    cross_block_map_zarr = ZarrSource.from_path(
        join(result_dir, "maps.zarr"), group="cross_block", mode="w"
    )
    cross_block_inv_map_zarr = ZarrSource.from_path(
        join(result_dir, "maps.zarr"), group="cross_block_inv", mode="w"
    )
    last_inv_map_zarr = ZarrSource.from_path(
        join(result_dir, "maps.zarr"), group="last_inv", mode="w"
    )
    runs = []
    blocks = []
    for block_index, i in enumerate(
        range(0, len(serialized_final_flows), integration_config.block_size)
    ):
        start = i
        end = min(start + integration_config.block_size, len(serialized_final_flows))
        blocks.append((start, end))
        opt_mesh_parameters = {
            "final_flow_dicts": serialized_final_flows[start:end],
            "start_section": start,
            "block_index": block_index,
            "main_map_zarr_dict": main_map_zarr.serialize(),
            "main_inv_map_zarr_dict": main_inv_map_zarr.serialize(),
            "cross_block_flow_zarr_dict": cross_block_flow_zarr.serialize(),
            "last_inv_map_zarr_dict": last_inv_map_zarr.serialize(),
            "stride": flow_config.stride,
            "integration_config": integration_config,
        }

        runs.append(
            start_block_mesh_optimization.submit(parameters=opt_mesh_parameters)
        )

    for run in runs:
        run.wait()

    run: FlowRun = run_deployment(
        name="Relax cross block mesh/default",
        parameters={
            "cross_block_flow_zarr_dict": cross_block_flow_zarr.serialize(),
            "cross_block_map_zarr_dict": cross_block_map_zarr.serialize(),
            "cross_block_inv_map_zarr_dict": cross_block_inv_map_zarr.serialize(),
            "main_map_size": main_map_zarr.get_data().shape[1:][::-1],
            "integration_config": integration_config,
            "stride": flow_config.stride,
        },
        client=get_client(),
    )

    if run.state.is_failed():
        raise RuntimeError(run.state.message)


if __name__ == "__main__":
    fine_alignment_optimize_mesh()
