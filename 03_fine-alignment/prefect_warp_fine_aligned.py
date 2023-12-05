from os.path import basename
from time import sleep

from prefect import flow, get_run_logger, task
from prefect.client.schemas import FlowRun
from prefect.deployments import run_deployment
from prefect.states import Completed, Failed
from prefect.tasks import task_input_hash
from s01_estimate_flow_fields import get_yx_size, list_zarr_sections
from s03_warp_fine_aligned_sections import create_zarr, warp_sections

RESULT_STORAGE_KEY = "{flow_run.name}/{task_run.task_name}/{task_run.name}.json"


@flow(name="[SOFIMA] Warp Sections", persist_result=True, cache_result_in_memory=False)
def warp_sections_flow(
    section_dirs: list[str],
    warp_start_section: int,
    warp_end_section: int,
    target_dir: str,
    yx_size: tuple[int, int],
    offset: int,
    blocks: list[tuple[int, int]],
    map_zarr_dir: str,
    flow_stride: int,
):
    warp_sections(
        section_dirs=section_dirs,
        warp_start_section=warp_start_section,
        warp_end_section=warp_end_section,
        target_dir=target_dir,
        yx_size=yx_size,
        offset=offset,
        blocks=blocks,
        map_zarr_dir=map_zarr_dir,
        flow_stride=flow_stride,
        logger=get_run_logger(),
    )


@task(
    task_run_name="submit-flow-run-{flow_name}-{batch}",
    persist_result=True,
    result_storage_key=RESULT_STORAGE_KEY,
    cache_result_in_memory=False,
    cache_key_fn=task_input_hash,
    refresh_cache=True,
    retries=2,
    retry_delay_seconds=10,
)
def submit_flowrun(flow_name: str, parameters: dict, batch: int):
    run: FlowRun = run_deployment(
        name=flow_name,
        parameters=parameters,
    )
    return run.state


@flow(
    name="[SOFIMA] Warp Fine Alignment",
    persist_result=True,
    cache_result_in_memory=False,
)
def warp_fine_alignment(
    user: str = "",
    stitched_sections_dir: str = "",
    warp_start_section: int = 0,
    warp_end_section: int = 9,
    output_dir: str = "",
    volume_name: str = "",
    block_size: int = 50,
    map_zarr_dir: str = "",
    flow_stride: int = 40,
    max_parallel_jobs: int = 25,
):
    section_dirs = list_zarr_sections(root_dir=stitched_sections_dir)

    blocks = []
    for i in range(0, len(section_dirs), block_size):
        blocks.append([i, min(len(section_dirs), i + block_size)])

    yx_size = get_yx_size(section_dirs, bin=1)

    target_dir = create_zarr(
        output_dir=output_dir,
        volume_name=volume_name,
        n_sections=len(section_dirs),
        yx_size=yx_size,
        bin=1,
    )

    n_sections_to_process = 0
    for i in range(len(section_dirs)):
        start_id = int(basename(section_dirs[i]).split("_")[0][1:])
        if warp_start_section <= start_id <= warp_end_section:
            n_sections_to_process += 1

    n_sections = len(section_dirs)
    batch_size = int(max(10, min(n_sections_to_process // max_parallel_jobs, 250)))
    n_jobs = n_sections_to_process // batch_size + 1
    batch_size = n_sections_to_process // n_jobs

    runs = []
    for batch_idx, i in enumerate(range(0, n_sections, batch_size)):
        start_id = int(basename(section_dirs[batch_idx * batch_size]).split("_")[0][1:])
        if warp_start_section <= start_id <= warp_end_section:
            sleep(5)
            runs.append(
                submit_flowrun.submit(
                    flow_name=f"[SOFIMA] Warp Sections/{user}",
                    parameters=dict(
                        section_dirs=section_dirs[
                            batch_idx * batch_size : (batch_idx + 1) * batch_size
                        ],
                        warp_start_section=warp_start_section,
                        warp_end_section=warp_end_section,
                        target_dir=target_dir,
                        yx_size=yx_size,
                        offset=i,
                        blocks=blocks,
                        map_zarr_dir=map_zarr_dir,
                        flow_stride=flow_stride,
                    ),
                    batch=batch_idx,
                    return_state=False,
                )
            )

    some_failed = False
    for run in runs:
        if run.result(raise_on_failure=False).is_failed():
            some_failed = True

    if some_failed:
        return Failed()
    else:
        return Completed()
