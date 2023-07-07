from os.path import dirname, join

from prefect import flow, task
from prefect.client.schemas import FlowRun
from prefect.deployments import run_deployment
from prefect.task_runners import SequentialTaskRunner
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
)
def parse_data_task(
    output_dir: str,
    acquisition_conf: AcquisitionConfig,
):
    return parse_data(
        output_dir=output_dir,
        sbem_root_dir=acquisition_conf.sbem_root_dir,
        acquisition=acquisition_conf.acquisition,
        tile_grid=acquisition_conf.tile_grid,
        thickness=acquisition_conf.thickness,
        resolution_xy=acquisition_conf.resolution_xy,
        tile_width=acquisition_conf.tile_width,
        tile_height=acquisition_conf.tile_height,
        tile_overlap=acquisition_conf.tile_overlap,
    )


@task(task_run_name="submit flow-run: {flow_name}")
def submit_flowrun(
    flow_name: str,
    parameters: dict,
):
    run: FlowRun = run_deployment(
        name=flow_name,
        parameters=parameters,
    )
    return run.state.result()


@task(
    name="register-tiles",
    persist_result=True,
    result_storage_key=RESULT_STORAGE_KEY,
)
def register_tiles_task(
    section_yaml_file: str,
    mesh_integration_config: MeshIntegrationConfig,
    registration_config: RegistrationConfig,
):
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
    )


@flow(
    name="[SOFIMA] Register Tiles",
    persist_result=True,
    task_runner=SequentialTaskRunner(),
)
def register_tiles_flow(
    section_yaml_files: list[str],
    mesh_integration_config: MeshIntegrationConfig,
    registration_config: RegistrationConfig,
):
    cvx2.setNumThreads(1)

    section_yaml_files = filter_ignore(section_yaml_files, file_name="section.yaml")

    meshes = []
    for section_yaml_file in section_yaml_files:
        meshes.append(
            register_tiles_task(
                section_yaml_file=section_yaml_file,
                mesh_integration_config=mesh_integration_config,
                registration_config=registration_config,
            )
        )

    return meshes


@task(
    name="warp-tiles",
    persist_result=True,
    result_storage_key=RESULT_STORAGE_KEY,
)
def warp_tiles_task(
    output_dir: str, mesh_file: str, stride: int, warp_config: WarpConfig
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
        warp_tiles_task(
            output_dir=output_dir,
            mesh_file=mesh_file,
            stride=stride,
            warp_config=warp_config,
        )


@flow(
    name="[SOFIMA] Tile Stitching",
    persist_result=True,
)
def tile_stitching(
    user: str,
    output_dir: str,
    acquisition_conf: AcquisitionConfig = AcquisitionConfig(),
    mesh_integration_config: MeshIntegrationConfig = MeshIntegrationConfig(),
    registration_config: RegistrationConfig = RegistrationConfig(),
    warp_config: WarpConfig = WarpConfig(),
    max_parallel_jobs: int = 10,
):
    sections = parse_data_task(
        output_dir=join(output_dir, "sections"), acquisition_conf=acquisition_conf
    )

    batch_size = int(max(10, min(len(sections) // max_parallel_jobs, 500)))
    n_jobs = len(sections) // batch_size + 1
    batch_size = len(sections) // n_jobs

    runs = []
    for i in range(0, len(sections), batch_size):
        runs.append(
            submit_flowrun.submit(
                flow_name=f"[SOFIMA] Register Tiles/{user}",
                parameters=dict(
                    section_yaml_files=sections[i : i + batch_size],
                    mesh_integration_config=mesh_integration_config,
                    registration_config=registration_config,
                ),
            )
        )

    meshes = []
    for run in runs:
        meshes.extend(run.result())

    runs = []
    for i in range(0, len(meshes), batch_size):
        runs.append(
            submit_flowrun.submit(
                flow_name=f"[SOFIMA] Warp Tiles/{user}",
                parameters=dict(
                    output_dir=join(output_dir, "stitched-sections"),
                    mesh_files=meshes[i : i + batch_size],
                    stride=mesh_integration_config.stride,
                    warp_config=warp_config,
                ),
            )
        )

    registered_tiles = []
    for run in runs:
        registered_tiles.extend(run.result())