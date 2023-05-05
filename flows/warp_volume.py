import gc
import threading
from pathlib import Path
from threading import Semaphore
from typing import Any, Optional

import numpy as np
import zarr
from connectomics.common import bounding_box
from connectomics.volume import subvolume
from cpr.Serializer import cpr_serializer
from cpr.utilities.utilities import task_input_hash
from cpr.zarr.ZarrSource import ZarrSource
from faim_prefect.parallelization.utils import wait_for_task_run
from numpy._typing import ArrayLike
from ome_zarr.io import parse_url
from prefect import flow, get_client, get_run_logger, task
from prefect.client.schemas import FlowRun
from prefect.context import TaskRunContext
from prefect.deployments import run_deployment
from prefect.filesystems import LocalFileSystem
from sofima import map_utils, warp
from sofima.processor import maps

from flows.utils.storage_key import RESULT_STORAGE_KEY


@task(cache_key_fn=task_input_hash, result_storage_key=RESULT_STORAGE_KEY)
def create_output_volume(
    source_volume: Path, target_volume: Path, z_size: int, yx_size: list[int]
):
    source_zarr = zarr.group(store=parse_url(source_volume).store)

    store = parse_url(path=target_volume, mode="w").store
    target_zarr: zarr.Group = zarr.group(store=store)

    target_zarr.create_dataset(
        name="0",
        shape=(z_size, yx_size[0], yx_size[1]),
        chunks=source_zarr["0"].chunks,
        dtype=source_zarr["0"].dtype,
        compressor=source_zarr["0"].compressor,
        fill_value=source_zarr["0"].fill_value,
        order=source_zarr["0"].order,
        overwrite=True,
    )
    target_zarr.attrs.update(source_zarr.attrs)

    return ZarrSource.from_path(target_volume, group="0", mode="w")


def exlude_semaphore_task_input_hash(
    context: "TaskRunContext", arguments: dict[str, Any]
) -> Optional[str]:
    hash_args = {}
    for k, item in arguments.items():
        if not isinstance(item, threading.Semaphore):
            hash_args[k] = item

    return task_input_hash(context, hash_args)


@task(
    cache_key_fn=exlude_semaphore_task_input_hash, result_storage_key=RESULT_STORAGE_KEY
)
def warp_section(
    source_volume: ZarrSource,
    target_zarr: ZarrSource,
    yx_start: list[int],
    yx_size: list[int],
    x: int,
    y: int,
    z: int,
    tile_size: int,
    z_offset: int,
    inv_map: ArrayLike,
    box: bounding_box.BoundingBox,
    stride: float,
    memory_lock: Semaphore,
):
    # TODO: Compute this only once per section.
    # box = bounding_box.BoundingBox(
    #     start=(0, 0, 0), size=(map_data.shape[-1], map_data.shape[-2], 1)
    # )
    # inv_map = map_utils.invert_map(map_data, box, box, stride)

    section_data = source_volume.get_data()

    out_vol = target_zarr.get_data()

    # tile_size = 20000
    # for y in range(yx_start[0], yx_start[0] + yx_size[0], tile_size):
    #     for x in range(yx_start[1], yx_start[1] + yx_size[1], tile_size):
    src_start_y = max(0, y - 500)
    src_start_x = max(0, x - 500)
    src_end_y = min(
        min(y + tile_size + 500, y + yx_size[0] + 500),
        yx_start[0] + out_vol.shape[1],
    )
    src_end_x = min(
        min(x + tile_size + 500, x + yx_size[1] + 500),
        yx_start[1] + out_vol.shape[2],
    )

    try:
        memory_lock.acquire()
        src_data = section_data[
            z : z + 1, src_start_y:src_end_y, src_start_x:src_end_x
        ][np.newaxis]

        img_box = bounding_box.BoundingBox(
            start=(src_start_x, src_start_y, 0),
            size=(src_end_x - src_start_x, src_end_y - src_start_y, 1),
        )

        end_y = min(min(y + tile_size, y + yx_size[0]), yx_start[0] + out_vol.shape[1])
        end_x = min(min(x + tile_size, x + yx_size[1]), yx_start[1] + out_vol.shape[2])
        out_box = bounding_box.BoundingBox(
            start=(x, y, 0), size=(end_x - x, end_y - y, 1)
        )

        out_start_y = min(y, y - yx_start[0])
        out_start_x = min(x, x - yx_start[1])
        out_end_y = min(end_y, end_y - yx_start[0])
        out_end_x = min(end_x, end_x - yx_start[1])
        logger = get_run_logger()
        logger.debug(f"y = {y}, x = {x}")
        logger.debug(f"img_box = {img_box}")
        logger.debug(f"out_box = {out_box}")
        logger.debug(f"out_start_y = {out_start_y}, out_end_y = {out_end_y}")
        logger.debug(f"out_start_x = {out_start_x}, out_end_x = {out_end_x}")
        out_vol[
            z - z_offset, out_start_y:out_end_y, out_start_x:out_end_x
        ] = warp.warp_subvolume(
            src_data,
            image_box=img_box,
            coord_map=inv_map,
            map_box=box,
            stride=stride,
            out_box=out_box,
            interpolation="lanczos",
            parallelism=1,
        )[
            0, 0
        ]

        gc.collect()
    except Exception as e:
        raise e
    finally:
        memory_lock.release()

    result = ZarrSource.from_path(
        path=target_zarr.get_path(),
        group=target_zarr._group,
        slices_start=[z - z_offset, y, x],
        slices_stop=[z - z_offset + 1, y + tile_size, x + tile_size],
    )

    return result


@task(
    persist_result=False,
    cache_result_in_memory=True,
    refresh_cache=True,
)
def reconcile_flow(
    blocks: list[tuple[int, int]],
    main_map_zarr_dict: dict,
    main_inv_map_zarr_dict: dict,
    cross_block_map_zarr_dict: dict,
    cross_block_inv_map_zarr_dict: dict,
    last_inv_map_zarr_dict: dict,
    stride: int,
    start_section: int,
    end_section: int,
):
    main_map = ZarrSource(**main_map_zarr_dict).get_data()
    main_inv_map = ZarrSource(**main_inv_map_zarr_dict).get_data()
    cross_block_map = ZarrSource(**cross_block_map_zarr_dict).get_data()
    cross_block_inv_map = ZarrSource(**cross_block_inv_map_zarr_dict).get_data()
    last_inv_map = ZarrSource(**last_inv_map_zarr_dict).get_data()

    class ReconcileCrossBlockMaps(maps.ReconcileCrossBlockMaps):
        def _open_volume(self, path: str):
            if path == "main_inv":
                return main_inv_map
            elif path == "last_inv":
                return last_inv_map
            elif path == "xblk":
                return cross_block_map
            elif path == "xblk_inv":
                return cross_block_inv_map
            else:
                raise ValueError(f"Unknown volume {path}")

    reconcile = ReconcileCrossBlockMaps(
        "xblk",
        "xblk_inv",
        "last_inv",
        "main_inv",
        {str(e): i for i, (s, e) in enumerate(blocks)},
        stride,
        0,
    )
    reconcile.set_effective_subvol_and_overlap(main_map.shape[1:][::-1], (0, 0, 0))
    size = (*main_map.shape[2:][::-1], end_section - start_section)
    main_box = bounding_box.BoundingBox(start=(0, 0, start_section), size=size)
    return reconcile.process(
        subvolume.Subvolume(main_map[:, start_section:end_section], main_box)
    ).data


@task(
    persist_result=False,
    cache_result_in_memory=True,
    refresh_cache=True,
)
def compute_inv_map(map_data, stride):
    box = bounding_box.BoundingBox(
        start=(0, 0, 0), size=(map_data.shape[-1], map_data.shape[-2], 1)
    )
    inv_map = map_utils.invert_map(map_data, box, box, stride)
    return inv_map, box


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
    z_offset: int,
    yx_start: list[int],
    yx_size: list[int],
    blocks: list[tuple[int, int]],
    main_map_zarr_dict: dict,
    main_inv_map_zarr_dict: dict,
    cross_block_map_zarr_dict: dict,
    cross_block_inv_map_zarr_dict: dict,
    last_inv_map_zarr_dict: dict,
    stride: int,
):
    source_volume = ZarrSource(**source_volume_dict)
    target_volume = ZarrSource(**target_volume_dict)

    memory_lock = Semaphore(19)
    warped_sections = []
    buffer = []
    tile_size = 2744 * 2
    for i, z in enumerate(range(start_section, end_section)):
        main_map = reconcile_flow.submit(
            blocks=blocks,
            main_map_zarr_dict=main_map_zarr_dict,
            main_inv_map_zarr_dict=main_inv_map_zarr_dict,
            cross_block_map_zarr_dict=cross_block_map_zarr_dict,
            cross_block_inv_map_zarr_dict=cross_block_inv_map_zarr_dict,
            last_inv_map_zarr_dict=last_inv_map_zarr_dict,
            stride=stride,
            start_section=z,
            end_section=z + 1,
        )
        task_run = compute_inv_map.submit(map_data=main_map.result(), stride=stride)

        wait_for_task_run(
            results=warped_sections,
            buffer=buffer,
            max_buffer_length=0,
        )
        gc.collect()

        inv_map, box = task_run.result()
        for y in range(yx_start[0], yx_start[0] + yx_size[0], tile_size):
            for x in range(yx_start[1], yx_start[1] + yx_size[1], tile_size):
                buffer.append(
                    warp_section.submit(
                        source_volume,
                        target_zarr=target_volume,
                        yx_start=yx_start,
                        yx_size=yx_size,
                        x=x,
                        y=y,
                        z=z,
                        tile_size=tile_size,
                        z_offset=z_offset,
                        inv_map=inv_map,
                        box=box,
                        stride=stride,
                        memory_lock=memory_lock,
                    )
                )

    wait_for_task_run(
        results=warped_sections,
        buffer=buffer,
        max_buffer_length=0,
    )
    return warped_sections


@task(cache_key_fn=task_input_hash, result_storage_key=RESULT_STORAGE_KEY)
def submit_flows(
    source_volume_dict: dict,
    target_volume_dict: dict,
    start_section: int,
    end_section: int,
    z_offset: int,
    yx_start: list[int],
    yx_size: list[int],
    blocks: list[tuple[int, int]],
    main_map_zarr_dict: dict,
    main_inv_map_zarr_dict: dict,
    cross_block_map_zarr_dict: dict,
    cross_block_inv_map_zarr_dict: dict,
    last_inv_map_zarr_dict: dict,
    stride: float,
):
    n_sections_per_job = 25
    warped_sections = []
    for i, z in enumerate(range(start_section, end_section, n_sections_per_job)):

        run: FlowRun = run_deployment(
            name="Warp sections/default",
            parameters={
                "source_volume_dict": source_volume_dict,
                "target_volume_dict": target_volume_dict,
                "start_section": z,
                "end_section": min(z + n_sections_per_job, end_section),
                "z_offset": z_offset,
                "yx_start": yx_start,
                "yx_size": yx_size,
                "blocks": blocks,
                "main_map_zarr_dict": main_map_zarr_dict,
                "main_inv_map_zarr_dict": main_inv_map_zarr_dict,
                "cross_block_map_zarr_dict": cross_block_map_zarr_dict,
                "cross_block_inv_map_zarr_dict": cross_block_inv_map_zarr_dict,
                "last_inv_map_zarr_dict": last_inv_map_zarr_dict,
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
    yx_start: list[int],
    yx_size: list[int],
    blocks: list[tuple[int, int]],
    main_map_zarr_dict: dict,
    main_inv_map_zarr_dict: dict,
    cross_block_map_zarr_dict: dict,
    cross_block_inv_map_zarr_dict: dict,
    last_inv_map_zarr_dict: dict,
    stride: float,
    parallelization: int = 5,
):
    src_volume = ZarrSource.from_path(source_volume, group="0")

    warped_zarr = create_output_volume(
        source_volume,
        target_volume,
        z_size=end_section - start_section + 1,
        yx_size=yx_size,
    )

    n_sections = end_section - start_section
    split = n_sections // parallelization
    start = start_section
    z_offset = start

    runs = []
    for i in range(parallelization - 1):
        runs.append(
            submit_flows.submit(
                source_volume_dict=src_volume.serialize(),
                target_volume_dict=warped_zarr.serialize(),
                start_section=start,
                end_section=start + split,
                z_offset=z_offset,
                yx_start=yx_start,
                yx_size=yx_size,
                blocks=blocks,
                main_map_zarr_dict=main_map_zarr_dict,
                main_inv_map_zarr_dict=main_inv_map_zarr_dict,
                cross_block_map_zarr_dict=cross_block_map_zarr_dict,
                cross_block_inv_map_zarr_dict=cross_block_inv_map_zarr_dict,
                last_inv_map_zarr_dict=last_inv_map_zarr_dict,
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
            z_offset=z_offset,
            yx_start=yx_start,
            yx_size=yx_size,
            blocks=blocks,
            main_map_zarr_dict=main_map_zarr_dict,
            main_inv_map_zarr_dict=main_inv_map_zarr_dict,
            cross_block_map_zarr_dict=cross_block_map_zarr_dict,
            cross_block_inv_map_zarr_dict=cross_block_inv_map_zarr_dict,
            last_inv_map_zarr_dict=last_inv_map_zarr_dict,
            stride=stride,
        )
    )

    warped_sections = []
    for run in runs:
        warped_sections.extend(run.result())

    return warped_sections
