import os
from os.path import dirname, exists, join

from prefect import State, flow, get_run_logger, task
from prefect.client.schemas import FlowRun
from prefect.deployments import run_deployment
from prefect.states import Completed, Failed
from prefect.task_runners import SequentialTaskRunner
from prefect.tasks import task_input_hash
from s01_parse_data import AcquisitionConfig, parse_data
from s02_register_tiles import MeshIntegrationConfig, RegistrationConfig, filter_ignore
from s03_warp_tiles import WarpConfig, warp_tiles
from sbem.record.Section import Section
from sbem.tile_stitching.sofima_utils import register_tiles
from sofima import mesh

try:
    from cvx2 import latest as cvx2
except ImportError:
    import cv2 as cvx2  # pytype:disable=import-error

RESULT_STORAGE_KEY = "{flow_run.name}/{task_run.task_name}/{task_run.name}.json"


@task(
    name="parse-sbem-data",
    refresh_cache=True,
    persist_result=True,
    result_storage_key=RESULT_STORAGE_KEY,
    cache_result_in_memory=False,
    cache_key_fn=task_input_hash,
)
def parse_data_task(
    output_dir: str,
    acquisition_conf: AcquisitionConfig,
    start_section: int,
    end_section: int,
):
    return parse_data(
        output_dir=output_dir,
        sbem_root_dir=acquisition_conf.sbem_root_dir,
        acquisition=acquisition_conf.acquisition,
        tile_grid=acquisition_conf.tile_grid,
        thickness=acquisition_conf.thickness,
        resolution_xy=acquisition_conf.resolution_xy,
        start_section=start_section,
        end_section=end_section,
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
    # if run.state.is_completed():
    #     return run.state.result()
    # elif run.state.is_failed():
    #     return run.state.result(raise_on_failure=False)
    # else:
    #     return run.state.result()


@task(
    name="register-tiles",
    task_run_name="register-tiles-{section_name}",
    persist_result=True,
    result_storage_key=RESULT_STORAGE_KEY,
    cache_result_in_memory=False,
    cache_key_fn=task_input_hash,
)
def register_tiles_task(
    section_yaml_file: str,
    section_name: str,
    mesh_integration_config: MeshIntegrationConfig,
    registration_config: RegistrationConfig,
):
    try:
        return register_tiles(
            section=Section.load_from_yaml(path=section_yaml_file),
            section_dir=dirname(section_yaml_file),
            stride=mesh_integration_config.stride,
            overlaps_x=tuple(registration_config.overlaps_x),
            overlaps_y=tuple(registration_config.overlaps_y),
            min_overlap=registration_config.min_overlap,
            min_range=tuple(registration_config.min_range),
            patch_size=tuple(registration_config.patch_size),
            batch_size=registration_config.batch_size,
            min_peak_ratio=registration_config.min_peak_ratio,
            min_peak_sharpness=registration_config.min_peak_sharpness,
            max_deviation=registration_config.max_deviation,
            max_magnitude=registration_config.max_magnitude,
            min_patch_size=registration_config.min_patch_size,
            max_gradient=registration_config.max_gradient,
            reconcile_flow_max_deviation=registration_config.reconcile_flow_max_deviation,
            integration_config=mesh.IntegrationConfig(
                dt=mesh_integration_config.dt,
                gamma=mesh_integration_config.gamma,
                k0=mesh_integration_config.k0,
                k=mesh_integration_config.k,
                stride=mesh_integration_config.stride,
                num_iters=mesh_integration_config.num_iters,
                max_iters=mesh_integration_config.max_iters,
                stop_v_max=mesh_integration_config.stop_v_max,
                dt_max=mesh_integration_config.dt_max,
                prefer_orig_order=mesh_integration_config.prefer_orig_order,
                start_cap=mesh_integration_config.start_cap,
                final_cap=mesh_integration_config.final_cap,
                remove_drift=mesh_integration_config.remove_drift,
            ),
            logger=get_run_logger(),
        )
    except StopIteration as e:
        raise RuntimeError(e)


@flow(
    name="[SOFIMA] Register Tiles",
    persist_result=True,
    task_runner=SequentialTaskRunner(),
    cache_result_in_memory=False,
    retries=0,
)
def register_tiles_flow(
    section_yaml_files: list[str] = [""],
    mesh_integration_config: MeshIntegrationConfig = MeshIntegrationConfig(),
    registration_config: RegistrationConfig = RegistrationConfig(),
    error_log_dir: str = "",
):
    cvx2.setNumThreads(1)

    section_yaml_files = filter_ignore(section_yaml_files, file_name="section.yaml")

    states: list[tuple[str, State]] = []
    for section_yaml_file in section_yaml_files:
        section_name = Section.load_from_yaml(path=section_yaml_file).get_name()
        states.append(
            tuple(
                [
                    section_name,
                    register_tiles_task(
                        section_yaml_file=section_yaml_file,
                        section_name=section_name,
                        mesh_integration_config=mesh_integration_config,
                        registration_config=registration_config,
                        return_state=True,
                    ),
                ]
            )
        )

    meshes = []
    failed_meshes = []
    for section_name, state in states:
        err_log_file = join(error_log_dir, f"{section_name}.err")
        if state.is_completed():
            if exists(err_log_file):
                os.rename(err_log_file, f"{err_log_file}.solved")

            meshes.append(state.result())
        else:
            with open(join(error_log_dir, f"{section_name}.err"), "w") as f:
                f.writelines(state.message)
            failed_meshes.append(section_name)

        get_run_logger().info(f"From Register Tiles: " f"{section_name} - {meshes[-1]}")

    if len(failed_meshes) > 0:
        return Failed(
            message=f"Tile registration failed for sections: {failed_meshes}",
            data=meshes,
        )
    else:
        return Completed(data=meshes)


@task(
    name="warp-tiles",
    task_run_name="warp-tiles-{section_name}",
    persist_result=True,
    result_storage_key=RESULT_STORAGE_KEY,
    cache_result_in_memory=False,
    cache_key_fn=task_input_hash,
    retries=0,
    retry_delay_seconds=1,
)
def warp_tiles_task(
    section_name: str,
    output_dir: str,
    mesh_file: str,
    stride: int,
    warp_config: WarpConfig,
):
    return warp_tiles(
        output_dir=output_dir,
        mesh_file=mesh_file,
        stride=stride,
        warp_config=warp_config,
    )


@flow(
    name="[SOFIMA] Warp Tiles",
    persist_result=True,
    task_runner=SequentialTaskRunner(),
    cache_result_in_memory=False,
    retries=0,
)
def warp_tiles_flow(
    output_dir: str,
    mesh_files: list[str],
    stride: int,
    warp_config: WarpConfig,
):
    cvx2.setNumThreads(1)
    from s02_register_tiles import filter_ignore

    mesh_files = filter_ignore(mesh_files, file_name="meshes.npz")

    for mesh_file in mesh_files:
        section = Section.load_from_yaml(
            mesh_file.replace("meshes.npz", "section.yaml")
        )
        warp_tiles_task(
            section_name=section.get_name(),
            output_dir=output_dir,
            mesh_file=mesh_file,
            stride=stride,
            warp_config=warp_config,
        )


@flow(
    name="[SOFIMA] Tile Stitching",
    persist_result=True,
    cache_result_in_memory=False,
    retries=0,
)
def tile_stitching(
    user: str,
    output_dir: str,
    acquisition_config: AcquisitionConfig = AcquisitionConfig(),
    start_section: int = 0,
    end_section: int = 10,
    mesh_integration_config: MeshIntegrationConfig = MeshIntegrationConfig(),
    registration_config: RegistrationConfig = RegistrationConfig(),
    warp_config: WarpConfig = WarpConfig(),
    max_parallel_jobs: int = 10,
):
    sections = parse_data_task(
        output_dir=join(output_dir, "sections"),
        acquisition_conf=acquisition_config,
        start_section=start_section,
        end_section=end_section,
    )

    batch_size = int(max(10, min(len(sections) // max_parallel_jobs, 250)))
    n_jobs = len(sections) // batch_size + 1
    batch_size = len(sections) // n_jobs

    tile_reg_err_log_dir = join(output_dir, "tile_registration_errors")
    os.makedirs(tile_reg_err_log_dir, exist_ok=True)
    runs = []
    for batch_number, i in enumerate(range(0, len(sections), batch_size)):
        runs.append(
            submit_flowrun.submit(
                flow_name=f"[SOFIMA] Register Tiles/{user}",
                parameters=dict(
                    section_yaml_files=sections[i : i + batch_size],
                    mesh_integration_config=mesh_integration_config,
                    registration_config=registration_config,
                    error_log_dir=tile_reg_err_log_dir,
                ),
                batch=batch_number,
                return_state=False,
            )
        )

    some_failed = False
    meshes = []
    for run in runs:
        if run.result(raise_on_failure=False).is_failed():
            some_failed = True
        meshes.extend(run.result(raise_on_failure=False).result(raise_on_failure=False))

    runs = []
    for batch_number, i in enumerate(range(0, len(meshes), batch_size)):
        runs.append(
            submit_flowrun.submit(
                flow_name=f"[SOFIMA] Warp Tiles/{user}",
                parameters=dict(
                    output_dir=join(output_dir, "stitched-sections"),
                    mesh_files=meshes[i : i + batch_size],
                    stride=mesh_integration_config.stride,
                    warp_config=warp_config,
                ),
                batch=batch_number,
            )
        )

    registered_tiles = []
    for run in runs:
        registered_tiles.extend(run.result())

    if some_failed:
        return Failed()
    else:
        return Completed()


if __name__ == "__main__":
    tile_stitching()
