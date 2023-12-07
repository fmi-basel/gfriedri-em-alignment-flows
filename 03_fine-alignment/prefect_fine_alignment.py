from os.path import basename, join, splitext
from time import sleep

import numpy as np
from parameter_config import FlowFieldEstimationConfig
from prefect import flow, get_run_logger
from prefect.client.schemas import FlowRun
from prefect.deployments import run_deployment
from prefect.states import Completed, Failed
from prefect.tasks import task, task_input_hash
from s01_estimate_flow_fields import (
    clean_flow,
    compute_final_flow,
    get_yx_size,
    list_zarr_sections,
    load_section_data,
)
from skimage.measure import block_reduce
from sofima import flow_field

RESULT_STORAGE_KEY = "{flow_run.name}/{task_run.task_name}/{task_run.name}.json"


def section_name(dir: str) -> str:
    return splitext(basename(dir))[0]


@flow(
    name="[SOFIMA] Estimate Z Flow-Fields",
    persist_result=True,
    cache_result_in_memory=False,
)
def estimate_z_flow_fields(
    section_dirs: str = "",
    yx_size: tuple[int, int] = (10, 10),
    ffe_conf: FlowFieldEstimationConfig = FlowFieldEstimationConfig(),
):
    logger = get_run_logger()

    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()

    previous_name = section_name(section_dirs[0])
    logger.info(f"Load section {previous_name}.")
    previous_section = load_section_data(section_dir=section_dirs[0], yx_size=yx_size)

    for i in range(1, len(section_dirs)):
        current_name = section_name(section_dirs[i])
        logger.info(f"Load section {current_name}.")
        current_section = load_section_data(
            section_dir=section_dirs[i], yx_size=yx_size
        )

        logger.info(
            f"Compute flow-field between {previous_name} and " f"{current_name}."
        )
        flows1x = mfc.flow_field(
            previous_section,
            current_section,
            (ffe_conf.patch_size, ffe_conf.patch_size),
            (ffe_conf.stride, ffe_conf.stride),
            batch_size=ffe_conf.batch_size,
        )

        logger.info("Bin data 2x.")
        prev_data_bin2 = block_reduce(previous_section, func=np.mean).astype(np.float32)
        curr_data_bin2 = block_reduce(current_section, func=np.mean).astype(np.float32)

        logger.info(
            f"Compute bin2 flow-field between {previous_name} and " f"{current_name}."
        )
        flows2x = mfc.flow_field(
            prev_data_bin2,
            curr_data_bin2,
            (ffe_conf.patch_size, ffe_conf.patch_size),
            (ffe_conf.stride, ffe_conf.stride),
            batch_size=ffe_conf.batch_size,
        )

        logger.info("Clean flow-field.")
        flows1x = clean_flow(
            flows1x,
            patch_size=ffe_conf.patch_size,
            stride=ffe_conf.stride,
            min_peak_ratio=ffe_conf.min_peak_ratio,
            min_peak_sharpness=ffe_conf.min_peak_sharpness,
            max_magnitude=ffe_conf.max_magnitude,
            max_deviation=ffe_conf.max_deviation,
        )

        logger.info("Clean bin2 flow-field.")
        flows2x = clean_flow(
            flows2x,
            patch_size=ffe_conf.patch_size,
            stride=ffe_conf.stride,
            min_peak_ratio=ffe_conf.min_peak_ratio,
            min_peak_sharpness=ffe_conf.min_peak_sharpness,
            max_magnitude=ffe_conf.max_magnitude,
            max_deviation=ffe_conf.max_deviation,
        )

        logger.info("Compute final flow.")
        final_flow = compute_final_flow(
            flows1x,
            flows2x,
            max_gradient=ffe_conf.max_gradient,
            max_deviation=ffe_conf.max_deviation,
            min_patch_size=ffe_conf.min_patch_size,
        )

        name = f"final_flow_{previous_name}_to_{current_name}.npy"
        logger.info(f"Save {name}.")
        np.save(join(section_dirs[i], name), final_flow)

        previous_section = current_section
        previous_name = current_name


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
    name="[SOFIMA] Estimate Z Flow-Fields (parallel)",
    persist_result=True,
    cache_result_in_memory=False,
)
def estimate_z_flow_fields_parallel(
    user: str = "",
    stitched_sections_dir: str = "",
    ffe_conf: FlowFieldEstimationConfig = FlowFieldEstimationConfig(),
    max_parallel_jobs: int = 10,
):
    section_dirs = list_zarr_sections(root_dir=stitched_sections_dir)

    yx_size = get_yx_size(section_dirs, bin=1)
    get_run_logger().info(f"Computed yx_size = ({yx_size[0]}, {yx_size[1]}).")

    n_sections = len(section_dirs)
    batch_size = int(max(10, min(n_sections // max_parallel_jobs, 250)))
    n_jobs = n_sections // batch_size + 1
    batch_size = n_sections // n_jobs

    runs = []
    for batch_number, i in enumerate(range(0, len(section_dirs), batch_size)):
        sleep(15)
        runs.append(
            submit_flowrun.submit(
                flow_name=f"[SOFIMA] Estimate Z Flow-Fields/{user}",
                parameters=dict(
                    section_dirs=section_dirs[i : i + batch_size + 1],
                    yx_size=yx_size,
                    ffe_conf=ffe_conf,
                ),
                batch=batch_number,
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
