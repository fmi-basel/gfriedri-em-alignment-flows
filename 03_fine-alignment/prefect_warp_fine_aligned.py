from prefect import flow, get_run_logger, task
from prefect.client.schemas import FlowRun
from prefect.deployments import run_deployment
from prefect.states import Completed, Failed
from prefect.tasks import task_input_hash
from s01_estimate_flow_fields import filter_sections, get_yx_size, list_zarr_sections
from s03_warp_fine_aligned_sections import create_zarr, warp_sections

RESULT_STORAGE_KEY = "{flow_run.name}/{task_run.task_name}/{task_run.name}.json"


@flow(name="[SOFIMA] Warp Sections", persist_result=True, cache_result_in_memory=False)
def warp_sections_flow(
    section_dirs: list[str],
    target_dir: str,
    yx_size: tuple[int, int],
    offset: int,
    start_section: int,
    end_section: int,
    blocks: list[tuple[int, int]],
    map_zarr_dir: str,
    flow_stride: int,
):
    warp_sections(
        section_dirs=section_dirs,
        target_dir=target_dir,
        yx_size=yx_size,
        offset=offset,
        start_section=start_section,
        end_section=end_section,
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
    output_dir: str = "",
    volume_name: str = "",
    start_section: int = 0,
    end_section: int = 9,
    block_size: int = 50,
    map_zarr_dir: str = "",
    flow_stride: int = 40,
    max_parallel_jobs: int = 10,
):
    section_dirs = list_zarr_sections(root_dir=stitched_sections_dir)
    section_dirs = filter_sections(
        section_dirs=section_dirs, start_section=start_section, end_section=end_section
    )

    blocks = []
    for i in range(0, len(section_dirs), block_size):
        blocks.append([i, min(len(section_dirs), i + block_size)])

    yx_size = get_yx_size(section_dirs, bin=1)

    target_dir = create_zarr(
        output_dir=output_dir,
        volume_name=volume_name,
        start_section=start_section,
        end_section=end_section,
        yx_size=yx_size,
        bin=1,
    )

    n_sections = end_section - start_section
    batch_size = int(max(10, min(n_sections // max_parallel_jobs, 250)))
    n_jobs = n_sections // batch_size + 1
    batch_size = n_sections // n_jobs

    runs = []
    for batch_idx, i in enumerate(range(start_section, end_section, batch_size)):
        runs.append(
            submit_flowrun.submit(
                flow_name=f"[SOFIMA] Warp Sections/{user}",
                parameters=dict(
                    section_dirs=section_dirs[
                        batch_idx * batch_size : (batch_idx + 1) * batch_size
                    ],
                    target_dir=target_dir,
                    yx_size=yx_size,
                    offset=start_section,
                    start_section=i,
                    end_section=i + batch_size,
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
