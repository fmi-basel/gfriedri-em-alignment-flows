import gc
from pathlib import Path
from typing import List

import numpy as np
import zarr
from connectomics.common import bounding_box
from cpr.numpy.NumpyTarget import NumpyTarget
from cpr.Serializer import cpr_serializer
from cpr.utilities.utilities import task_input_hash
from cpr.zarr.ZarrSource import ZarrSource
from ome_zarr.io import parse_url
from prefect import flow, get_client, task
from prefect.client.schemas import FlowRun
from prefect.deployments import run_deployment
from prefect.filesystems import LocalFileSystem
from sofima import map_utils, warp


@task(cache_key_fn=task_input_hash)
def create_output_volume(source_volume: Path, target_volume: Path):
    source_zarr = zarr.group(store=parse_url(source_volume).store)["0"]

    store = parse_url(path=target_volume, mode="w").store
    target_zarr: zarr.Group = zarr.group(store=store)

    target_zarr.create_dataset(
        name="0",
        shape=source_zarr.shape,
        chunks=source_zarr.chunks,
        dtype=source_zarr.dtype,
        compressor=source_zarr.compressor,
        fill_value=source_zarr.fill_value,
        order=source_zarr.order,
        overwrite=True,
    )

    return ZarrSource.from_path(target_volume, group="0", mode="w")


@task(cache_key_fn=task_input_hash)
def copy_first_section(
    source_volume: ZarrSource, target_volume: ZarrSource, z: int = 0
):
    target_volume.get_data()[z] = source_volume.get_data()[z]
    return ZarrSource.from_path(
        target_volume.get_path(), group="0", slices_start=[z], slices_stop=[z + 1]
    )


@task(cache_key_fn=task_input_hash)
def warp_section(
    source_volume: ZarrSource,
    target_zarr: ZarrSource,
    z: int,
    map: NumpyTarget,
    stride: float,
):
    map_data = map.get_data()
    box = bounding_box.BoundingBox(
        start=(0, 0, 0), size=(map_data.shape[-1], map_data.shape[-2], 1)
    )
    inv_map = map_utils.invert_map(map.get_data(), box, box, stride)

    section_data = source_volume.get_data()

    out_vol = target_zarr.get_data()

    tile_size = 20000
    for y in range(0, section_data.shape[1], tile_size):
        for x in range(0, section_data.shape[2], tile_size):
            src_start_y = max(0, y - 500)
            src_start_x = max(0, x - 500)
            src_end_y = min(y + tile_size + 500, section_data.shape[1])
            src_end_x = min(x + tile_size + 500, section_data.shape[2])
            src_data = section_data[
                z : z + 1, src_start_y:src_end_y, src_start_x:src_end_x
            ][np.newaxis]

            img_box = bounding_box.BoundingBox(
                start=(src_start_x, src_start_y, 0),
                size=(src_end_x - src_start_x, src_end_y - src_start_y, 1),
            )
            end_y = min(y + tile_size, section_data.shape[1])
            end_x = min(x + tile_size, section_data.shape[2])
            out_box = bounding_box.BoundingBox(
                start=(x, y, 0), size=(end_x - x, end_y - y, 1)
            )

            out_vol[z, y:end_y, x:end_x] = warp.warp_subvolume(
                src_data,
                image_box=img_box,
                coord_map=inv_map,
                map_box=box,
                stride=stride,
                out_box=out_box,
                interpolation="lanczos",
                parallelism=1,
            )[0, 0]

            gc.collect()

    result = ZarrSource.from_path(
        path=target_zarr.get_path(),
        group=target_zarr._group,
        slices_start=[z],
        slices_stop=[z + 1],
    )

    return result


@flow(
    name="Warp sections",
    persist_result=True,
    result_storage=LocalFileSystem.load("gfriedri-em-alignment-flows-storage"),
    result_serializer=cpr_serializer(),
    cache_result_in_memory=False,
)
def warp_sections(
    source_volume_dict: dict,
    target_volume_dict: dict,
    start_section: int,
    end_section: int,
    map_dicts: List[dict],
    stride: float,
):
    source_volume = ZarrSource(**source_volume_dict)
    target_volume = ZarrSource(**target_volume_dict)
    maps = [NumpyTarget(**d) for d in map_dicts]

    buffer = []
    if start_section == 0:
        buffer.append(
            copy_first_section.submit(source_volume, target_volume, z=start_section)
        )
        start_section = 1

    warped_sections = []
    for i, z in enumerate(range(start_section, end_section)):
        warp_task = warp_section.submit(
            source_volume,
            target_zarr=target_volume,
            z=z,
            map=maps[i],
            stride=stride,
        )
        buffer.append(warp_task)
        while len(buffer) >= 2:
            warped_sections.append(buffer.pop(0).result())

    while len(buffer) > 0:
        warped_sections.append(buffer.pop(0).result())

    return warped_sections


@task(cache_key_fn=task_input_hash)
def submit_flows(
    source_volume_dict: dict,
    target_volume_dict: dict,
    start_section: int,
    end_section: int,
    map_dicts: List[dict],
    stride: float,
):
    warped_sections = []
    for z in range(start_section, end_section, 10):
        run: FlowRun = run_deployment(
            name="Warp sections/default",
            parameters={
                "source_volume_dict": source_volume_dict,
                "target_volume_dict": target_volume_dict,
                "start_section": z,
                "end_section": min(z + 10, end_section),
                "map_dicts": map_dicts,
                "stride": stride,
            },
            client=get_client(),
        )
        warped_sections.extend(run.state.result())

    return warped_sections


@flow(
    name="Warp volume",
    persist_result=True,
    result_storage=LocalFileSystem.load("gfriedri-em-alignment-flows-storage"),
    result_serializer=cpr_serializer(),
    cache_result_in_memory=False,
)
def warp_volume(
    source_volume: Path,
    target_volume: Path,
    start_section: int,
    end_section: int,
    map_dicts: List[dict],
    stride: float,
    parallelization: int = 5,
):
    src_volume = ZarrSource.from_path(source_volume, group="0")

    warped_zarr = create_output_volume(source_volume, target_volume)

    n_sections = end_section - start_section
    split = n_sections // parallelization
    start = start_section

    runs = []
    for i in range(parallelization - 1):
        if start == 0:
            # The is no flow for the first (zeroth) section
            maps = map_dicts[: start + split - 1]
        else:
            maps = map_dicts[start - 1 : start + split - 1]

        runs.append(
            submit_flows.submit(
                source_volume_dict=src_volume.serialize(),
                target_volume_dict=warped_zarr.serialize(),
                start_section=start,
                end_section=start + split,
                map_dicts=maps,
                stride=stride,
            )
        )
        start = start + split

    runs.append(
        submit_flows.submit(
            source_volume_dict=src_volume.serialize(),
            target_volume_dict=warped_zarr.serialize(),
            start_section=start,
            end_section=end_section + 1,
            map_dicts=map_dicts[start - 1 :],
            stride=stride,
        )
    )

    warped_sections = []
    for run in runs:
        warped_sections.extend(run.result())

    return warped_sections
