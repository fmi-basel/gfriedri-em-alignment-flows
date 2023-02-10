import os
import sys
import threading
from os.path import join
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax.numpy as jnp
import numpy as np
import psutil
from connectomics.common import bounding_box
from cpr.numpy.NumpyTarget import NumpyTarget
from cpr.Serializer import cpr_serializer
from cpr.utilities.utilities import task_input_hash
from cpr.zarr.ZarrSource import ZarrSource
from prefect import flow, get_client, get_run_logger, task
from prefect.client.schemas import FlowRun
from prefect.context import TaskRunContext
from prefect.deployments import run_deployment
from prefect.filesystems import LocalFileSystem
from pydantic import BaseModel
from skimage.measure import block_reduce
from sofima import flow_field, flow_utils, map_utils, mesh
from tqdm.auto import tqdm


@task(cache_key_fn=task_input_hash)
def load_section(path: str, z: int) -> ZarrSource:
    return ZarrSource.from_path(path, group="0", slices_start=[z], slices_stop=[z + 1])


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


@task(cache_key_fn=exlude_semaphore_task_input_hash)
def compute_flow_field(
    mfc,
    prev,
    curr,
    out_dir,
    z,
    patch_size,
    stride,
    batch_size,
    gpu_sem: threading.Semaphore,
):
    logger = get_run_logger()
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1e9
    logger.info(f"Process memory usage: {mem_usage} GB")

    prev_data = np.squeeze(prev.get_data())
    curr_data = np.squeeze(curr.get_data())

    logger.info(f"prev_data-size: {sys.getsizeof(prev_data) / 1e+9} GB")
    logger.info(f"curr_data-size: {sys.getsizeof(curr_data) / 1e+9} GB")

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

    prev_data_bin2 = block_reduce(prev_data, func=np.mean)
    curr_data_bin2 = block_reduce(curr_data, func=np.mean)

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

    os.makedirs(join(out_dir, str(z)), exist_ok=True)
    ff1x = NumpyTarget.from_path(join(out_dir, str(z), f"flows1x_{z - 1}_to_{z}.npy"))
    ff1x.set_data(flows1x)

    ff2x = NumpyTarget.from_path(join(out_dir, str(z), f"flows2x_{z - 1}_to_{z}.npy"))
    ff2x.set_data(flows2x)
    return ff1x, ff2x


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
):
    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
    gpu_sem = threading.Semaphore(1)
    sections = []
    flows1x, flows2x = [], []

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
                gpu_sem=gpu_sem,
            )
        )
        prev = curr

        while len(buffer) >= 4:
            flow1x, flow2x = buffer[0].result()
            flows1x.append(flow1x)
            flows2x.append(flow2x)
            buffer = buffer[1:]

    # Wait for all tasks to finish and collect results
    while len(buffer) > 0:
        flow1x, flow2x = buffer[0].result()
        flows1x.append(flow1x)
        flows2x.append(flow2x)
        buffer = buffer[1:]

    return flows1x, flows2x


@task(cache_key_fn=task_input_hash)
def load_flows(
    flow_list: List[NumpyTarget],
    patch_size: int,
    stride: int,
    out_dir: str,
    min_peak_ratio: float = 1.6,
    min_peak_sharpness: float = 1.6,
    max_magnitude: float = 80,
    max_deviation: float = 20,
):
    flows = []
    for f in flow_list:
        flows.append(f.get_data())

    flows = np.transpose(np.array(flows), [1, 0, 2, 3])
    pad = patch_size // 2 // stride
    flows = np.pad(
        flows, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan
    )

    flow = flow_utils.clean_flow(
        flows,
        min_peak_ratio=min_peak_ratio,
        min_peak_sharpness=min_peak_sharpness,
        max_magnitude=max_magnitude,
        max_deviation=max_deviation,
    )

    os.makedirs(out_dir, exist_ok=True)
    output = NumpyTarget.from_path(join(out_dir, "clean_flows.npy"))
    output.set_data(flow)
    return output


@task(cache_key_fn=task_input_hash)
def compute_final_flow(
    flow1x: NumpyTarget,
    flow2x: NumpyTarget,
    max_gradient: float,
    max_deviation: float,
    min_patch_size: int,
    out_dir: str,
):
    f1 = flow1x.get_data()
    f2 = flow2x.get_data()
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

    final_flow = flow_utils.reconcile_flows(
        (f1, f2_hires),
        max_gradient=max_gradient,
        max_deviation=max_deviation,
        min_patch_size=min_patch_size,
    )

    os.makedirs(out_dir, exist_ok=True)
    output = NumpyTarget.from_path(join(out_dir, "final_flow.npy"))
    output.set_data(final_flow)
    return output


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


@task(cache_key_fn=task_input_hash)
def run_mesh_optimization(
    flow: NumpyTarget,
    out_dir: str,
    stride: int,
    integration_config: MeshIntegrationConfig = MeshIntegrationConfig(),
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

    final_flow = flow.get_data()
    solved = [np.zeros_like(final_flow[:, 0:1, ...])]
    origin = jnp.array([0.0, 0.0])

    for z in tqdm(range(0, final_flow.shape[1])):
        prev = map_utils.compose_maps_fast(
            final_flow[:, z : z + 1, ...], origin, stride, solved[-1], origin, stride
        )
        x = np.zeros_like(solved[0])
        x, e_kin, num_steps = mesh.relax_mesh(x, prev, config)
        x = np.array(x)
        solved.append(x)

    solved = np.concatenate(solved, axis=1)

    # box1x = bounding_box.BoundingBox(
    #     start=(0, 0, 0), size=(final_flow.shape[-1], final_flow.shape[-2], 1)
    # )
    # inv_map = map_utils.invert_map(solved, box1x, box1x, stride)

    outputs = []
    for z in range(1, solved.shape[1]):
        out = NumpyTarget.from_path(join(out_dir, f"map_{z}.npy"))
        out.set_data(solved[:, z : z + 1])
        outputs.append(out)

    return outputs


@flow(
    name="Optimize mesh",
    persist_result=True,
    result_storage=LocalFileSystem.load("gfriedri-em-alignment-flows-storage"),
    result_serializer=cpr_serializer(),
    cache_result_in_memory=False,
)
def optimize_mesh(
    flows_1x_dicts: List[dict],
    flows_2x_dicts: List[dict],
    out_dir: str,
    patch_size: int,
    stride: int,
    min_peak_ratio: float,
    min_peak_sharpness: float,
    max_magnitude: float,
    max_deviation: float,
    max_gradient: float,
    min_patch_size: int,
    integration_config: MeshIntegrationConfig = MeshIntegrationConfig(),
):
    flows_1x = [NumpyTarget(**d) for d in flows_1x_dicts]
    flows_2x = [NumpyTarget(**d) for d in flows_2x_dicts]

    flows1x = load_flows.submit(
        flows_1x,
        patch_size=patch_size,
        stride=stride,
        out_dir=join(out_dir, "clean_flow1x"),
        min_peak_ratio=min_peak_ratio,
        min_peak_sharpness=min_peak_sharpness,
        max_magnitude=max_magnitude,
        max_deviation=max_deviation,
    )
    flows2x = load_flows.submit(
        flows_2x,
        patch_size=patch_size,
        stride=stride,
        out_dir=join(out_dir, "clean_flow2x"),
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
        out_dir=out_dir,
    )

    inv_maps = run_mesh_optimization(
        flow=final_flow,
        out_dir=out_dir,
        stride=stride,
        integration_config=integration_config,
    )

    return inv_maps


@task(cache_key_fn=task_input_hash)
def submit_flows(
    start_section: int,
    end_section: int,
    chunk_size: int,
    path: str,
    out_dir: str,
    patch_size: int,
    stride: int,
    batch_size: int,
):
    flows1x, flows2x = [], []
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
            },
            client=get_client(),
        )

        f1x, f2x = run.state.result()
        flows1x.extend(f1x)
        flows2x.extend(f2x)

    return flows1x, flows2x


@flow(
    name="Estimate flow-field (parallel)",
    persist_result=True,
    result_storage=LocalFileSystem.load("gfriedri-em-alignment-flows-storage"),
    result_serializer=cpr_serializer(),
    cache_result_in_memory=False,
)
def parallel_flow_field_estimation(
    path: Path = "/path/to/volume",
    start_section: int = 0,
    end_section: int = 1000,
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
    integration_config: MeshIntegrationConfig = MeshIntegrationConfig(),
    chunk_size: int = 100,
    parallelization: int = 2,
    refresh_cache: bool = False,
):
    # TODO: Free up memory!
    if refresh_cache:
        # This does not work!
        os.environ["PREFECT_TASKS_REFRESH_CACHE"] = "true"
    n_sections = end_section - start_section
    split = n_sections // parallelization
    start = start_section

    runs = []
    for i in range(parallelization - 1):
        runs.append(
            submit_flows.submit(
                start_section=start,
                end_section=start + split,
                chunk_size=chunk_size,
                path=path,
                out_dir=join(out_dir, "flow-fields"),
                patch_size=patch_size,
                stride=stride,
                batch_size=batch_size,
            )
        )
        start = start + split

    runs.append(
        submit_flows.submit(
            start_section=start,
            end_section=end_section,
            chunk_size=chunk_size,
            path=path,
            out_dir=join(out_dir, "flow-fields"),
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
        )
    )

    flows1x, flows2x = [], []
    for run in runs:
        f1x, f2x = run.result()
        flows1x.extend(f1x)
        flows2x.extend(f2x)

    serialized_f1x = []
    serialized_f2x = []
    for f1, f2 in zip(flows1x, flows2x):
        serialized_f1x.append(f1.serialize())
        serialized_f2x.append(f2.serialize())

    opt_mesh_parameters = {
        "flows_1x_dicts": serialized_f1x,
        "flows_2x_dicts": serialized_f2x,
        "out_dir": join(out_dir, "maps"),
        "patch_size": patch_size,
        "stride": stride,
        "min_peak_ratio": min_peak_ratio,
        "min_peak_sharpness": min_peak_sharpness,
        "max_magnitude": max_magnitude,
        "max_deviation": max_deviation,
        "max_gradient": max_gradient,
        "min_patch_size": min_patch_size,
        "integration_config": integration_config,
    }

    get_run_logger().info(opt_mesh_parameters)

    run: FlowRun = run_deployment(
        name="Optimize mesh/default",
        parameters=opt_mesh_parameters,
        client=get_client(),
    )

    inv_maps = run.state.result()

    return inv_maps


if __name__ == "__main__":
    parallel_flow_field_estimation()
