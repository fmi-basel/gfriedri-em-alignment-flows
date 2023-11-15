import json
from os.path import basename, join

from prefect import flow, task
from prefect.client.schemas import FlowRun
from prefect.deployments import run_deployment
from prefect.task_runners import SequentialTaskRunner
from prefect.tasks import task_input_hash
from s01_coarse_align_section_pairs import (
    compute_shift,
    filter_sections,
    get_padding_per_section,
    list_zarr_sections,
    load_shifts,
)

RESULT_STORAGE_KEY = "{flow_run.name}/{task_run.task_name}/{task_run.name}.json"


@task(
    task_run_name="submit flow-run-{flow_name}-{batch}",
    persist_result=True,
    result_storage_key="{flow_run.name}/submit flow-run/{task_run.id}.json",
    cache_result_in_memory=False,
    cache_key_fn=task_input_hash,
)
def submit_flowrun(
    flow_name: str,
    parameters: dict,
    batch: int,
):
    run: FlowRun = run_deployment(
        name=flow_name,
        parameters=parameters,
    )
    return run.state


@task(
    name="list-zarr-sections",
    refresh_cache=True,
    persist_result=True,
    result_storage_key=RESULT_STORAGE_KEY,
    cache_result_in_memory=False,
    cache_key_fn=task_input_hash,
)
def list_zarr_sections_task(
    root_dir: str,
):
    return list_zarr_sections(root_dir=root_dir)


@task(
    name="compute-shift",
    task_run_name="coarse-align-{next_name}-to-{current_name}",
    refresh_cache=True,
    persist_result=True,
    result_storage_key=RESULT_STORAGE_KEY,
    cache_result_in_memory=False,
    cache_key_fn=task_input_hash,
)
def compute_shift_task(
    current_section_dir: str, next_section_dir: str, current_name: str, next_name: str
):
    compute_shift(
        current_section_dir=current_section_dir,
        next_section_dir=next_section_dir,
    )


@task(
    name="compute-padding",
    refresh_cache=True,
    persist_result=True,
    result_storage_key=RESULT_STORAGE_KEY,
    cache_result_in_memory=False,
    cache_key_fn=task_input_hash,
)
def compute_padding(section_dirs):
    shifts = load_shifts(section_dirs)
    paddings = get_padding_per_section(shifts)
    for i in range(len(section_dirs)):
        with open(join(section_dirs[i], "coarse_stack_padding.json"), "w") as f:
            json.dump(dict(shift_y=int(paddings[i, 0]), shift_x=int(paddings[i, 1])), f)


@flow(
    name="[SOFIMA] Pair-wise Coarse Align",
    persist_result=True,
    task_runner=SequentialTaskRunner(),
    cache_result_in_memory=False,
)
def coarse_align_pairs(section_dirs: list[str]):
    for i in range(len(section_dirs) - 1):
        compute_shift_task(
            current_section_dir=section_dirs[i],
            next_section_dir=section_dirs[i + 1],
            current_name=basename(section_dirs[i]),
            next_name=basename(section_dirs[i + 1]),
        )


@flow(
    name="[SOFIMA] Coarse Align",
    persist_result=True,
    cache_result_in_memory=False,
    retries=1,
)
def coarse_alignment(
    user: str,
    stitched_sections_dir: str,
    start_section: int = 0,
    end_section: int = 10,
    max_parallel_jobs: int = 10,
):
    section_dirs = list_zarr_sections_task(
        root_dir=stitched_sections_dir,
    )

    section_dirs = filter_sections(
        section_dirs=section_dirs,
        start_section=start_section,
        end_section=end_section,
    )

    batch_size = int(max(10, min(len(section_dirs) // max_parallel_jobs, 500)))
    n_jobs = len(section_dirs) // batch_size + 1
    batch_size = len(section_dirs) // n_jobs

    runs = []
    for batch_number, i in enumerate(range(0, len(section_dirs), batch_size)):
        start = max(0, i - 1)
        end = i + batch_size
        runs.append(
            submit_flowrun.submit(
                flow_name=f"[SOFIMA] Pair-wise Coarse Align/{user}",
                parameters=dict(section_dirs=section_dirs[start:end]),
                batch=batch_number,
                return_state=False,
            )
        )

    for run in runs:
        run.result(raise_on_failure=True)

    compute_padding(section_dirs)


if __name__ == "__main__":
    coarse_alignment()
