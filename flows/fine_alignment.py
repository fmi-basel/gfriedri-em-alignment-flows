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
from prefect import flow, task
from prefect.context import TaskRunContext
from prefect.filesystems import LocalFileSystem
from sofima import flow_field


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
    os.makedirs(join(out_dir, str(z)), exist_ok=True)
    flow_field = NumpyTarget.from_path(
        join(out_dir, str(z), f"flow_field_{z - 1}_to_{z}.npy")
    )

    prev_data = np.squeeze(prev.get_data())
    curr_data = np.squeeze(curr.get_data())

    try:
        gpu_sem.acquire()
        ff = mfc.flow_field(
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

    flow_field.set_data(ff)
    return flow_field


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
    flow_fields = []

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

        while len(buffer) > 2:
            flow_fields.append(buffer[0].result())
            buffer = buffer[1:]

    return flow_fields
