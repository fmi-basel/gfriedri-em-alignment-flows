import os
import threading
from os.path import join
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from cpr.numpy.NumpyTarget import NumpyTarget
from cpr.Serializer import cpr_serializer
from cpr.utilities.utilities import task_input_hash
from cpr.zarr.ZarrSource import ZarrSource
from prefect import flow, get_client, get_run_logger, task
from prefect.client.schemas import FlowRun
from prefect.context import TaskRunContext
from prefect.deployments import run_deployment
from prefect.filesystems import LocalFileSystem
from skimage.measure import block_reduce
from sofima import flow_field


@task(cache_key_fn=task_input_hash, refresh_cache=True)
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


@task(cache_key_fn=exlude_semaphore_task_input_hash, refresh_cache=True)
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
    prev_data = np.squeeze(prev.get_data())
    curr_data = np.squeeze(curr.get_data())

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


@task(cache_key_fn=task_input_hash, refresh_cache=True)
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
    chunk_size: int = 100,
    parallelization: int = 2,
):
    logger = get_run_logger()

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
                out_dir=out_dir,
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
            out_dir=out_dir,
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

    logger.info(flows1x)
    logger.info(flows2x)
    return flows1x, flows2x


if __name__ == "__main__":
    parallel_flow_field_estimation()
