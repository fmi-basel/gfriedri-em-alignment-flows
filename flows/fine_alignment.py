import gc
import json
import os
import threading
from datetime import datetime
from os.path import join
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax.numpy as jnp
import numpy as np
import pkg_resources
import psutil
import zarr
from connectomics.common import bounding_box
from cpr.numpy.NumpyTarget import NumpyTarget
from cpr.Serializer import cpr_serializer
from cpr.utilities.utilities import task_input_hash
from cpr.zarr.ZarrSource import ZarrSource
from faim_prefect.parallelization.utils import wait_for_task_run
from numcodecs import Blosc
from numpy._typing import ArrayLike
from ome_zarr.io import parse_url
from prefect import flow, get_client, get_run_logger, task
from prefect.client.schemas import FlowRun
from prefect.context import FlowRunContext, TaskRunContext
from prefect.deployments import run_deployment
from prefect.filesystems import LocalFileSystem
from pydantic import BaseModel
from skimage.measure import block_reduce
from sofima import flow_field, flow_utils, map_utils, mesh
from tqdm.auto import tqdm

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


class WarpConfig(BaseModel):
    target_volume_name: str = "warped_zyx.zarr"
    start_section: int = 0
    end_section: int = 199
    yx_start: list[int] = list([1000, 2000])
    yx_size: list[int] = list([1000, 1000])
    parallelization: int = 16


@task(cache_key_fn=task_input_hash, result_storage_key=RESULT_STORAGE_KEY)
def load_section(path: str, z: int) -> ZarrSource:
    return ZarrSource.from_path(path, group="0", slices_start=[z], slices_stop=[z + 1])


def clean_flow(
    flow,
    patch_size: int,
    stride: int,
    min_peak_ratio: float = 1.6,
    min_peak_sharpness: float = 1.6,
    max_magnitude: float = 80,
    max_deviation: float = 20,
):
    flow = flow[np.newaxis]
    flow = np.transpose(flow, [1, 0, 2, 3])
    pad = patch_size // 2 // stride
    flow = np.pad(
        flow, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan
    )

    return flow_utils.clean_flow(
        flow,
        min_peak_ratio=min_peak_ratio,
        min_peak_sharpness=min_peak_sharpness,
        max_magnitude=max_magnitude,
        max_deviation=max_deviation,
    )


def exlude_semaphore_task_input_hash(
    context: "TaskRunContext", arguments: Dict[str, Any]
) -> Optional[str]:
    hash_args = {}
    for k, item in arguments.items():
        if (not isinstance(item, threading.Semaphore)) and (
            not isinstance(item, flow_field.JAXMaskedXCorrWithStatsCalculator)
        ):
            hash_args[k] = item

    return task_input_hash(context, hash_args)


@task(
    cache_key_fn=exlude_semaphore_task_input_hash, result_storage_key=RESULT_STORAGE_KEY
)
def compute_flow_field(
    mfc,
    prev,
    curr,
    out_dir,
    z,
    patch_size,
    stride,
    batch_size,
    min_peak_ratio: float,
    min_peak_sharpness: float,
    max_magnitude: float,
    max_deviation: float,
    max_gradient: float,
    min_patch_size: int,
    gpu_sem: threading.Semaphore,
):
    logger = get_run_logger()
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1e9
    logger.info(f"[Start] Process memory usage: {mem_usage} GB")

    prev_data = np.squeeze(prev.get_data())
    curr_data = np.squeeze(curr.get_data())

    logger.info(
        f"prev_data-size: " f"{(prev_data.size * prev_data.itemsize) / 1e+9} GB"
    )
    logger.info(f"curr_data-size: " f"{curr_data.size * curr_data.itemsize / 1e+9} GB")

    try:
        gpu_sem.acquire()
        flows1x = mfc.flow_field(
            prev_data,
            curr_data,
            (patch_size, patch_size),
            (stride, stride),
            batch_size=batch_size,
        )
    except RuntimeError as e:
        raise e
    finally:
        gpu_sem.release()

    prev_data_bin2 = block_reduce(prev_data, func=np.mean).astype(np.float32)
    curr_data_bin2 = block_reduce(curr_data, func=np.mean).astype(np.float32)

    try:
        gpu_sem.acquire()
        flows2x = mfc.flow_field(
            prev_data_bin2,
            curr_data_bin2,
            (patch_size, patch_size),
            (stride, stride),
            batch_size=batch_size,
        )
    except RuntimeError as e:
        raise e
    finally:
        gpu_sem.release()

    del prev_data
    del prev_data_bin2
    del curr_data
    del curr_data_bin2
    prev._data = None
    curr._data = None
    gc.collect()
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1e9
    logger.info(f"[End] Process memory usage: {mem_usage} GB")

    flows1x = clean_flow(
        flows1x,
        patch_size=patch_size,
        stride=stride,
        min_peak_ratio=min_peak_ratio,
        min_peak_sharpness=min_peak_sharpness,
        max_magnitude=max_magnitude,
        max_deviation=max_deviation,
    )

    flows2x = clean_flow(
        flows2x,
        patch_size=patch_size,
        stride=stride,
        min_peak_ratio=min_peak_ratio,
        min_peak_sharpness=min_peak_sharpness,
        max_magnitude=max_magnitude,
        max_deviation=max_deviation,
    )

    final_flow = compute_final_flow(
        flows1x,
        flows2x,
        max_gradient=max_gradient,
        max_deviation=max_deviation,
        min_patch_size=min_patch_size,
    )

    final_flow_output = NumpyTarget.from_path(
        join(out_dir, f"final_flow_{z - 1}_to_{z}.npy")
    )
    final_flow_output.set_data(final_flow)

    return final_flow_output


@flow(
    name="Estimate flow-field",
    persist_result=True,
    result_storage=LocalFileSystem.load("gfriedri-em-alignment-flows-storage"),
    result_serializer=cpr_serializer(),
    cache_result_in_memory=False,
)
def flow_field_estimation(
    path: Path = "/path/to/volume",
    start_section: int = 0,
    end_section: int = 9,
    out_dir: Path = "/path/to/flow_field_storage",
    patch_size: int = 160,
    stride: int = 40,
    batch_size: int = 256,
    min_peak_ratio: float = 1.6,
    min_peak_sharpness: float = 1.6,
    max_magnitude: float = 80,
    max_deviation: float = 20,
    max_gradient: float = 0,
    min_patch_size: int = 400,
):
    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
    gpu_sem = threading.Semaphore(1)
    sections = []
    final_flows = []

    prev = load_section(path, start_section)
    for i in range(start_section + 1, end_section + 1):
        sections.append(load_section.submit(path, i))

    sections = sections[::-1]

    buffer = []
    for z in range(start_section + 1, end_section + 1):
        curr = sections.pop().result()

        buffer.append(
            compute_flow_field.submit(
                mfc=mfc,
                prev=prev,
                curr=curr,
                out_dir=out_dir,
                z=z,
                patch_size=patch_size,
                stride=stride,
                batch_size=batch_size,
                min_peak_ratio=min_peak_ratio,
                min_peak_sharpness=min_peak_sharpness,
                max_magnitude=max_magnitude,
                max_deviation=max_deviation,
                max_gradient=max_gradient,
                min_patch_size=min_patch_size,
                gpu_sem=gpu_sem,
            )
        )
        prev = curr

        wait_for_task_run(
            results=final_flows,
            buffer=buffer,
            max_buffer_length=4,
            result_insert_fn=lambda r: r.result(),
        )

    # Wait for all tasks to finish and collect results
    wait_for_task_run(
        results=final_flows,
        buffer=buffer,
        max_buffer_length=0,
        result_insert_fn=lambda r: r.result(),
    )

    return final_flows


def compute_final_flow(
    f1: ArrayLike,
    f2: ArrayLike,
    max_gradient: float,
    max_deviation: float,
    min_patch_size: int,
):
    f2_hires = np.zeros_like(f1)

    scale = 0.5
    box1x = bounding_box.BoundingBox(
        start=(0, 0, 0), size=(f1.shape[-1], f1.shape[-2], 1)
    )
    box2x = bounding_box.BoundingBox(
        start=(0, 0, 0), size=(f2.shape[-1], f2.shape[-2], 1)
    )

    for z in tqdm(range(f2.shape[1])):
        # Upsample and scale spatial components.
        resampled = map_utils.resample_map(
            f2[:, z : z + 1, ...], box2x, box1x, 1 / scale, 1  #
        )
        f2_hires[:, z : z + 1, ...] = resampled / scale

    return flow_utils.reconcile_flows(
        (f1, f2_hires),
        max_gradient=max_gradient,
        max_deviation=max_deviation,
        min_patch_size=min_patch_size,
    )


@task(
    cache_key_fn=task_input_hash,
    result_storage_key=RESULT_STORAGE_KEY,
    refresh_cache=True,
)
def run_block_mesh_optimization(
    final_flows: list[NumpyTarget],
    start_section: int,
    block_index: int,
    main_map_zarr: ZarrSource,
    main_inv_map_zarr: ZarrSource,
    cross_block_flow_zarr: ZarrSource,
    last_inv_map_zarr: ZarrSource,
    stride: int,
    integration_config: MeshIntegrationConfig = MeshIntegrationConfig(),
):
    logger = get_run_logger()
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
    final_flow = np.concatenate([ff.get_data() for ff in final_flows], axis=1)
    origin = jnp.array([0.0, 0.0])

    solved = main_map_zarr.get_data()
    inv_map = main_inv_map_zarr.get_data()
    cross_block_flow = cross_block_flow_zarr.get_data()
    last_inv = last_inv_map_zarr.get_data()
    for z in tqdm(range(0, final_flow.shape[1])):
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

    return main_map_zarr, main_inv_map_zarr, cross_block_flow_zarr, last_inv_map_zarr


@flow(
    name="Optimize block mesh",
    persist_result=True,
    result_storage=LocalFileSystem.load("gfriedri-em-alignment-flows-storage"),
    result_serializer=cpr_serializer(),
    cache_result_in_memory=False,
)
def optimize_block_mesh(
    final_flow_dicts: List[dict],
    start_section: int,
    block_index: int,
    main_map_zarr_dict: dict,
    main_inv_map_zarr_dict: dict,
    cross_block_flow_zarr_dict: dict,
    last_inv_map_zarr_dict: dict,
    stride: int,
    integration_config: MeshIntegrationConfig = MeshIntegrationConfig(),
):
    final_flows = [NumpyTarget(**d) for d in final_flow_dicts]
    main_map_zarr = ZarrSource(**main_map_zarr_dict)
    main_inv_map_zarr = ZarrSource(**main_inv_map_zarr_dict)
    cross_block_flow_zarr = ZarrSource(**cross_block_flow_zarr_dict)
    last_inv_map_zarr = ZarrSource(**last_inv_map_zarr_dict)

    maps = run_block_mesh_optimization(
        final_flows=final_flows,
        start_section=start_section,
        block_index=block_index,
        main_map_zarr=main_map_zarr,
        main_inv_map_zarr=main_inv_map_zarr,
        cross_block_flow_zarr=cross_block_flow_zarr,
        last_inv_map_zarr=last_inv_map_zarr,
        stride=stride,
        integration_config=integration_config,
    )

    return maps


@task(cache_key_fn=task_input_hash, result_storage_key=RESULT_STORAGE_KEY)
def submit_flows(
    start_section: int,
    end_section: int,
    chunk_size: int,
    path: str,
    out_dir: str,
    patch_size: int,
    stride: int,
    batch_size: int,
    min_peak_ratio: float = 1.6,
    min_peak_sharpness: float = 1.6,
    max_magnitude: float = 80,
    max_deviation: float = 20,
    max_gradient: float = 0,
    min_patch_size: int = 400,
):
    final_flows = []
    for z in range(start_section, end_section, chunk_size):
        run: FlowRun = run_deployment(
            name="Estimate flow-field/default",
            parameters={
                "path": path,
                "start_section": z,
                "end_section": min(z + chunk_size, end_section),
                "out_dir": out_dir,
                "patch_size": patch_size,
                "stride": stride,
                "batch_size": batch_size,
                "min_peak_ratio": min_peak_ratio,
                "min_peak_sharpness": min_peak_sharpness,
                "max_magnitude": max_magnitude,
                "max_deviation": max_deviation,
                "max_gradient": max_gradient,
                "min_patch_size": min_patch_size,
            },
            client=get_client(),
        )

        final_flows.extend(run.state.result())

    return final_flows


def write_alignment_info(
    path: str,
    coarse_volume_path: str,
    start_section: int,
    end_section: int,
    result_dir: str,
    flow_config: FlowComputationConfig,
    integration_config: MeshIntegrationConfig,
    warp_config: WarpConfig,
    context: FlowRunContext,
):
    params = {
        "coarse_volume_path": str(coarse_volume_path),
        "start_section": start_section,
        "end_section": end_section,
        "result_dir": str(result_dir),
        "flow_config": flow_config.dict(),
        "integration_config": integration_config.dict(),
        "warp_config": warp_config.dict(),
    }
    context = {
        "start_time": context.flow_run.start_time.strftime("%Y-%m-%d " "%H:%M:%S"),
        "flow_id": context.flow_run.flow_id,
        "flow_run_id": context.flow_run.id,
        "flow_run_version": context.flow_run.flow_version,
        "flow_run_name": context.flow_run.name,
        "deployment_id": context.flow_run.deployment_id,
    }
    date = datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
    sofima_version = pkg_resources.get_distribution("sofima").version
    content = (
        "# Warp volume\n"
        "Source: [https://github.com/fmi-basel/gfriedri-em-alignment"
        "-flows](https://github.com/fmi-basel/gfriedri-em-alignment"
        "-flows)\n"
        f"Date: {date}\n"
        "\n"
        "## Summary\n"
        "The fine-aligned volume "
        f"{join(result_dir, warp_config.target_volume_name)} was "
        "computed with [SOFIMA]("
        "https://github.com/google-research/sofima).\n"
        "\n"
        "## Parameters\n"
        f"{json.dumps(params, indent=4)}\n"
        "\n"
        "## Packages\n"
        f"* [https://github.com/google-research/sofima]("
        f"https://github.com/google-research/sofima): {sofima_version}\n"
        f"\n"
        f"## Prefect Context\n"
        f"{str(context)}\n"
    )

    with open(path, "w") as f:
        f.write(content)


@task(
    cache_key_fn=task_input_hash,
    result_storage_key=RESULT_STORAGE_KEY,
    refresh_cache=True,
)
def start_block_mesh_optimization(parameters):
    run: FlowRun = run_deployment(
        name="Optimize block mesh/default",
        parameters=parameters,
        client=get_client(),
    )
    return run.state.result()


@task(cache_key_fn=task_input_hash, result_storage_key=RESULT_STORAGE_KEY)
def start_warping(parameters):
    run: FlowRun = run_deployment(
        name="Warp volume/default",
        parameters=parameters,
        client=get_client(),
    )

    return run.state.result()


@flow(
    name="Relax cross block mesh",
    persist_result=True,
    result_storage=LocalFileSystem.load("gfriedri-em-alignment-flows-storage"),
    result_serializer=cpr_serializer(),
    cache_result_in_memory=False,
)
def relax_cross_block_mesh(
    cross_block_flow_zarr_dict: dict,
    cross_block_map_zarr_dict: dict,
    cross_block_inv_map_zarr_dict: dict,
    main_map_size: tuple[int, int, int],
    integration_config: MeshIntegrationConfig,
    stride: int,
):
    cross_block_flow = ZarrSource(**cross_block_flow_zarr_dict).get_data()
    cross_block_map = ZarrSource(**cross_block_map_zarr_dict).get_data()
    cross_block_inv_map = ZarrSource(**cross_block_inv_map_zarr_dict).get_data()
    map_box = bounding_box.BoundingBox(start=(0, 0, 0), size=main_map_size)
    map2x_box = map_box.scale(0.5)
    xblk_stride = stride * 2

    x_block_flow = map_utils.resample_map(
        cross_block_flow, map_box, map2x_box, stride, xblk_stride
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
        xblk, map2x_box, map_box, stride * 2, stride
    )
    cross_block_inv_map[:, :, ...] = map_utils.invert_map(
        cross_block_map[:], map_box, map_box, stride
    )


@flow(
    name="Volume fine-alignment",
    persist_result=True,
    result_storage=LocalFileSystem.load("gfriedri-em-alignment-flows-storage"),
    result_serializer=cpr_serializer(),
    cache_result_in_memory=False,
)
def parallel_flow_field_estimation(
    coarse_volume_path: Path = "/path/to/volume",
    start_section: int = 0,
    end_section: int = 1000,
    result_dir: Path = "/path/to/flow_field_storage",
    flow_config: FlowComputationConfig = FlowComputationConfig(),
    integration_config: MeshIntegrationConfig = MeshIntegrationConfig(),
    warp_config: WarpConfig = WarpConfig(),
):
    n_sections = end_section - start_section
    split = n_sections // flow_config.parallelization
    start = start_section

    flow_dir = join(result_dir, "flow-fields")
    os.makedirs(flow_dir, exist_ok=True)

    runs = []
    for i in range(flow_config.parallelization - 1):
        runs.append(
            submit_flows.submit(
                start_section=start,
                end_section=start + split,
                chunk_size=flow_config.chunk_size,
                path=coarse_volume_path,
                out_dir=flow_dir,
                patch_size=flow_config.patch_size,
                stride=flow_config.stride,
                batch_size=flow_config.batch_size,
                min_peak_ratio=flow_config.min_peak_ratio,
                min_peak_sharpness=flow_config.min_peak_sharpness,
                max_magnitude=flow_config.max_magnitude,
                max_deviation=flow_config.max_deviation,
                max_gradient=flow_config.max_gradient,
                min_patch_size=flow_config.min_patch_size,
            )
        )
        start = start + split

    runs.append(
        submit_flows.submit(
            start_section=start,
            end_section=end_section,
            chunk_size=flow_config.chunk_size,
            path=coarse_volume_path,
            out_dir=flow_dir,
            patch_size=flow_config.patch_size,
            stride=flow_config.stride,
            batch_size=flow_config.batch_size,
            min_peak_ratio=flow_config.min_peak_ratio,
            min_peak_sharpness=flow_config.min_peak_sharpness,
            max_magnitude=flow_config.max_magnitude,
            max_deviation=flow_config.max_deviation,
            max_gradient=flow_config.max_gradient,
            min_patch_size=flow_config.min_patch_size,
        )
    )

    final_flows = []
    for run in runs:
        final_flows.extend(run.result())

    serialized_final_flows = []
    for final_flow in final_flows:
        serialized_final_flows.append(final_flow.serialize())

    # Optimize z-alignment in chunks/blocks
    store = parse_url(path=join(result_dir, "maps.zarr"), mode="w").store
    map_zarr: zarr.Group = zarr.group(store=store)

    shape = final_flows[0].get_data().shape[2:]
    if "main" not in map_zarr:
        map_zarr.create_dataset(
            name="main",
            shape=(2, len(final_flows) + 1, *shape),
            chunks=(2, 1, *shape),
            dtype="<f4",
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            fill_value=0,
            overwrite=True,
        )
    if "main_inv" not in map_zarr:
        map_zarr.create_dataset(
            name="main_inv",
            shape=(2, len(final_flows) + 1, *shape),
            chunks=(2, 1, *shape),
            dtype="<f4",
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            fill_value=0,
            overwrite=True,
        )
    if "cross_block_flow" not in map_zarr:
        map_zarr.create_dataset(
            name="cross_block_flow",
            shape=(2, len(final_flows) // integration_config.block_size + 1, *shape),
            chunks=(2, 1, *shape),
            dtype="<f4",
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            fill_value=0,
            overwrite=True,
        )
    if "cross_block" not in map_zarr:
        map_zarr.create_dataset(
            name="cross_block",
            shape=(2, len(final_flows) // integration_config.block_size + 1, *shape),
            chunks=(2, 1, *shape),
            dtype="<f4",
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            fill_value=0,
            overwrite=True,
        )
    if "cross_block_inv" not in map_zarr:
        map_zarr.create_dataset(
            name="cross_block_inv",
            shape=(2, len(final_flows) // integration_config.block_size + 1, *shape),
            chunks=(2, 1, *shape),
            dtype="<f4",
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            fill_value=0,
            overwrite=True,
        )
    if "last_inv" not in map_zarr:
        map_zarr.create_dataset(
            name="last_inv",
            shape=(2, len(final_flows) + 1, *shape),
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
        range(0, len(final_flows), integration_config.block_size)
    ):
        start = i
        end = min(start + integration_config.block_size, len(final_flows))
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

    # warp_parameters = {
    #     "source_volume": coarse_volume_path,
    #     "target_volume": join(result_dir, warp_config.target_volume_name),
    #     "start_section": warp_config.start_section,
    #     "end_section": warp_config.end_section,
    #     "yx_start": warp_config.yx_start,
    #     "yx_size": warp_config.yx_size,
    #     "blocks": blocks,
    #     "main_map_zarr_dict": main_map_zarr.serialize(),
    #     "main_inv_map_zarr_dict": main_inv_map_zarr.serialize(),
    #     "cross_block_map_zarr_dict": cross_block_map_zarr.serialize(),
    #     "cross_block_inv_map_zarr_dict": cross_block_inv_map_zarr.serialize(),
    #     "last_inv_map_zarr_dict": last_inv_map_zarr.serialize(),
    #     "stride": flow_config.stride,
    #     "parallelization": warp_config.parallelization,
    # }

    # warped_sections = start_warping(
    #     parameters=warp_parameters,
    # )

    # write_alignment_info(
    #     path=join(result_dir, warp_config.target_volume_name, "summary.md"),
    #     coarse_volume_path=coarse_volume_path,
    #     start_section=start_section,
    #     end_section=end_section,
    #     result_dir=result_dir,
    #     flow_config=flow_config,
    #     integration_config=integration_config,
    #     warp_config=warp_config,
    #     context=get_run_context(),
    # )

    # return warped_sections


if __name__ == "__main__":
    parallel_flow_field_estimation()
