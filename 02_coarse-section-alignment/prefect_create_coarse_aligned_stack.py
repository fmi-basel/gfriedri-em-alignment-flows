from os.path import basename

import zarr
from ome_zarr.io import parse_url
from prefect import flow, get_run_logger, task
from prefect.states import Completed, Failed
from prefect.task_runners import SequentialTaskRunner
from prefect.tasks import task_input_hash
from prefect_coarse_alignment import list_zarr_sections_task, submit_flowrun
from s02_create_coarse_aligned_stack import (
    create_zarr,
    get_padding_per_section,
    load_shifts,
    write_section,
)

RESULT_STORAGE_KEY = "{flow_run.name}/{task_run.task_name}/{task_run.name}.json"


@task(
    name="load-shifts",
    refresh_cache=True,
    persist_result=True,
    result_storage_key=RESULT_STORAGE_KEY,
    cache_result_in_memory=False,
    cache_key_fn=task_input_hash,
)
def compute_padding_task(section_dirs: list[str]):
    shifts = load_shifts(section_dirs=section_dirs)
    return get_padding_per_section(shifts=shifts)


@task(
    name="create-volume",
    refresh_cache=True,
    persist_result=True,
    result_storage_key=RESULT_STORAGE_KEY,
    cache_result_in_memory=False,
    cache_key_fn=task_input_hash,
)
def create_zarr_task(
    output_dir: str,
    volume_name: str,
    start_section: int,
    end_section: int,
    yx_size: tuple[int, int],
    bin: int,
):
    return create_zarr(
        output_dir=output_dir,
        volume_name=volume_name,
        start_section=start_section,
        end_section=end_section,
        yx_size=yx_size,
        bin=bin,
    )


@task(
    name="write-section",
    task_run_name="write-section-{section_name}",
    refresh_cache=True,
    persist_result=True,
    result_storage_key=RESULT_STORAGE_KEY,
    cache_result_in_memory=False,
    cache_key_fn=task_input_hash,
)
def write_section_task(
    section_dir: str,
    padding,
    start_section: int,
    yx_start: tuple[int, int],
    yx_size: tuple[int, int],
    bin: int,
    zarr_root: zarr.Group,
    section_name: str,
):
    write_section(
        section_dir=section_dir,
        padding=padding,
        start_section=start_section,
        yx_start=yx_start,
        yx_size=yx_size,
        bin=bin,
        zarr_root=zarr_root,
    )


@flow(
    name="[SOFIMA] Write Coarse Aligned Sections",
    persist_result=True,
    task_runner=SequentialTaskRunner(),
    cache_result_in_memory=False,
)
def write_coarse_aligned_sections(
    section_dirs: list[str],
    index: int,
    batch_size: int,
    zarr_path: str,
    start_section: int,
    yx_start: tuple[int, int],
    yx_size: tuple[int, int],
    bin: int,
):
    store = parse_url(zarr_path, mode="w").store
    zarr_root = zarr.group(store=store)

    section_paddings = compute_padding_task(
        section_dirs=section_dirs,
    )

    for i in range(index, min(index + batch_size, len(section_dirs))):
        write_section_task(
            section_dir=section_dirs[i],
            padding=section_paddings[i],
            start_section=start_section,
            yx_start=yx_start,
            yx_size=yx_size,
            bin=bin,
            zarr_root=zarr_root,
            section_name=basename(section_dirs[i]),
        )


@flow(
    name="[SOFIMA] Create Coarse Stack",
    persist_result=True,
    cache_result_in_memory=False,
    retries=1,
)
def create_coarse_stack(
    user: str,
    stitched_sections_dir: str,
    start_section: int = 0,
    end_section: int = 10,
    output_dir: str = "",
    volume_name: str = "coarse_volume.zarr",
    yx_start: tuple[int, int] = tuple([0, 0]),
    yx_size: tuple[int, int] = tuple([4096, 4096]),
    bin: int = 2,
    max_parallel_jobs: int = 10,
):
    assert (
        yx_size[0] % bin == 0
    ), f"yx_size[1]={yx_size[0]} is not divisible by bin={bin}."
    assert (
        yx_size[1] % bin == 0
    ), f"yx_size[1]={yx_size[1]} is not divisible by bin={bin}."

    logger = get_run_logger()

    section_dirs = list_zarr_sections_task(
        root_dir=stitched_sections_dir,
    )
    logger.info(len(section_dirs))

    filtered_section_dirs = []
    for sec in section_dirs:
        sec_id = int(basename(sec).split("_")[0][1:])
        if start_section <= sec_id < end_section:
            filtered_section_dirs.append(sec)

    logger.info(len(filtered_section_dirs))
    empty_zarr_path = create_zarr_task(
        output_dir=output_dir,
        volume_name=volume_name,
        start_section=start_section,
        end_section=end_section,
        yx_size=yx_size,
        bin=bin,
    )

    batch_size = int(max(10, min(len(filtered_section_dirs) // max_parallel_jobs, 500)))
    n_jobs = len(filtered_section_dirs) // batch_size + 1
    batch_size = len(filtered_section_dirs) // n_jobs

    runs = []
    for batch_number, i in enumerate(range(0, len(filtered_section_dirs), batch_size)):
        runs.append(
            submit_flowrun.submit(
                flow_name=f"[SOFIMA] Write Coarse Aligned Sections/{user}",
                parameters=dict(
                    section_dirs=filtered_section_dirs[i : i + batch_size],
                    index=i,
                    batch_size=batch_size,
                    zarr_path=empty_zarr_path,
                    start_section=start_section,
                    yx_start=yx_start,
                    yx_size=yx_size,
                    bin=bin,
                ),
                batch=batch_number,
            )
        )

    some_failed = False
    results = []
    for run in runs:
        if run.result(raise_on_failure=False).is_failed():
            some_failed = True
        results.extend(
            run.result(raise_on_failure=False).result(raise_on_failure=False)
        )

    if some_failed:
        return Failed()
    else:
        return Completed()


if __name__ == "__main__":
    create_coarse_stack()
