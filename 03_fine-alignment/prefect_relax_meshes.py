from glob import glob
from os.path import join

import numpy as np
from parameter_config import MeshIntegrationConfig
from prefect import flow, get_run_logger, task
from prefect.client.schemas import FlowRun
from prefect.deployments import run_deployment
from prefect.tasks import task_input_hash
from s01_estimate_flow_fields import list_zarr_sections
from s02_relax_meshes import (
    create_map_storage,
    relax_meshes_cross_blocks,
    relax_meshes_in_blocks,
)

RESULT_STORAGE_KEY = "{flow_run.name}/{task_run.task_name}/{task_run.name}.json"


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
    name="[SOFIMA] Relax Meshes in Blocks",
    persist_result=True,
    cache_result_in_memory=False,
)
def relax_meshes_in_blocks_flow(
    section_dirs: list[str],
    section_offset: int,
    block_index_offset: int,
    output_dir: str,
    mesh_integration: MeshIntegrationConfig,
    flow_stride: int,
):
    relax_meshes_in_blocks(
        section_dirs=section_dirs,
        section_offset=section_offset,
        block_index_offset=block_index_offset,
        output_dir=output_dir,
        mesh_integration=mesh_integration,
        flow_stride=flow_stride,
        logger=get_run_logger(),
    )


@flow(
    name="[SOFIMA] Relax Meshes Cross Blocks",
    persist_result=True,
    cache_result_in_memory=False,
)
def relax_mehses_cross_blocks_flow(
    output_dir: str, mesh_integration: MeshIntegrationConfig, flow_stride: int
):
    relax_meshes_cross_blocks(
        output_dir=output_dir,
        mesh_integration=mesh_integration,
        flow_stride=flow_stride,
        logger=get_run_logger(),
    )


@flow(name="[SOFIMA] Relax Meshes", persist_result=True, cache_result_in_memory=False)
def relax_meshes_flow(
    user: str = "",
    stitched_sections_dir: str = "",
    output_dir: str = "",
    mesh_integration: MeshIntegrationConfig = MeshIntegrationConfig(),
    flow_stride: int = 40,
):
    section_dirs = list_zarr_sections(root_dir=stitched_sections_dir)

    dummy_flow = np.load(glob(join(section_dirs[1], "final_flow_*.npy"))[0])
    create_map_storage(
        output_dir=output_dir,
        shape=dummy_flow.shape[2:],
        n_sections=len(section_dirs) + 1,
        block_size=mesh_integration.block_size,
    )

    chunk_factor = 100 // mesh_integration.block_size
    chunk_size = chunk_factor * mesh_integration.block_size
    runs = []
    for i, chunk_start in enumerate(range(0, len(section_dirs), chunk_size)):
        runs.append(
            submit_flowrun.submit(
                flow_name=f"[SOFIMA] Relax Meshes in Blocks/{user}",
                parameters=dict(
                    section_dirs=section_dirs[chunk_start : chunk_start + chunk_size],
                    section_offset=chunk_start,
                    block_index_offset=i,
                    output_dir=output_dir,
                    mesh_integration=mesh_integration,
                    flow_stride=flow_stride,
                ),
                batch=i,
                return_state=False,
            )
        )

    for run in runs:
        run.result()

    submit_flowrun.submit(
        flow_name=f"[SOFIMA] Relax Meshes Cross Blocks/{user}",
        parameters=dict(
            output_dir=output_dir,
            mesh_integration=mesh_integration,
            flow_stride=flow_stride,
        ),
        batch=0,
        return_state=False,
    ).result()
