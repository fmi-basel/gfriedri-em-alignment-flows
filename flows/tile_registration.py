import json
from os.path import join
from typing import Dict

import git
from prefect import flow, get_run_logger, task, unmapped
from prefect_dask import DaskTaskRunner
from sbem.experiment.Experiment import Experiment
from utils.system import save_system_information

from flows.flow_parameter_types.ExperimentConfig import ExperimentConfig
from flows.flow_parameter_types.TileRegistrationConfig import (
    MeshIntegrationConfig,
    RegistrationConfig,
)
from flows.sofima_tasks.sofima_tasks import build_integration_config, run_sofima


@task(persist_result=False)
def load_experiment(path: str):
    return Experiment.load(path=path)


@task(persist_result=False)
def get_sections(
    exp: Experiment,
    sample_name: str,
    acquisition: str,
    tile_grid_num: int,
    start_section_num: int,
    end_section_num: int,
):
    logger = get_run_logger()
    logger.info(f"Sample: {sample_name} of acquisition {acquisition}")
    logger.info(f"Retrieving sections {start_section_num} to " f"{end_section_num}.")
    if start_section_num is not None and end_section_num is not None:
        sections = exp.get_sample(sample_name).get_section_range(
            start_section_num=start_section_num,
            end_section_num=end_section_num,
            tile_grid_num=tile_grid_num,
            include_skipped=False,
        )
    else:
        sections = exp.get_sample(sample_name).get_sections_of_acquisition(
            acquisition=acquisition, tile_grid_num=tile_grid_num, include_skipped=False
        )

    section_paths = []
    path = join(
        sections[0].get_sample().get_experiment().get_root_dir(),
        sections[0].get_sample().get_experiment().get_name(),
        sections[0].get_sample().get_name(),
    )
    for s in sections:
        section_paths.append(
            {
                "name": s.get_name(),
                "stitched": s.is_stitched(),
                "skip": s.skip(),
                "acquisition": s.get_acquisition(),
                "section_num": s.get_section_num(),
                "tile_grid_num": s.get_tile_grid_num(),
                "details": "",
                "path": path,
            }
        )

    return section_paths


@task()
def save_params(output_dir: str, params: Dict):
    """
    Dump prefect context into prefect-context.json.
    :param output_dir:
    :param context_dict:
    :return:
    """
    logger = get_run_logger()

    outpath = join(output_dir, "args_tile-registration.json")

    with open(outpath, "w") as f:
        json.dump(params, f, indent=4)

    logger.info(f"Saved flow parameters to {outpath}.")


@task()
def commit_changes(exp: Experiment):
    with git.Repo(join(exp.get_root_dir(), exp.get_name())) as repo:
        repo.index.add(repo.untracked_files)
        repo.index.add([item.a_path for item in repo.index.diff(None)])
        repo.index.commit("Compute tile-registration meshes.", author=exp._git_author)


@flow(
    name="Register Tiles",
    task_runner=DaskTaskRunner(
        cluster_class="dask_jobqueue.SLURMCluster",
        cluster_kwargs={
            "account": "dlthings",
            "queue": "main",
            "cores": 4,
            "processes": 1,
            "memory": "12 GB",
            "walltime": "06:00:00",
            "job_extra_directives": [
                "--gpus-per-node=1",
                "--ntasks=1",
                "--output=/tungstenfs/scratch/gmicro_share/_prefect/slurm/gfriedri-em-alignment-flows/output/%j.out",
            ],
            "worker_extra_args": [
                "--lifetime",
                "345m",
                "--lifetime-stagger",
                "15m",
            ],
            "job_script_prologue": [
                "conda run -p /tungstenfs/scratch/gmicro_share/_prefect/miniconda3/envs/airtable python /tungstenfs/scratch/gmicro_share/_prefect/airtable/log-slurm-job.py --config /tungstenfs/scratch/gmicro/_prefect/airtable/slurm-job-log.ini"
            ],
        },
        adapt_kwargs={
            "minimum": 1,
            "maximum": 12,
        },
    ),
    result_storage="local-file-system/gfriedri-em-alignment-flows-storage",
    persist_result=False,
)
def tile_registration_flow(
    exp_config: ExperimentConfig = ExperimentConfig(),
    mesh_integration_config: MeshIntegrationConfig = MeshIntegrationConfig(),
    registration_config: RegistrationConfig = RegistrationConfig(),
):
    params = {
        "exp_config": exp_config.dict(),
        "mesh_integration_config": mesh_integration_config.dict(),
        "registration_config": registration_config.dict(),
    }
    logger = get_run_logger()
    exp = load_experiment.submit(path=exp_config.exp_path).result()

    # TODO: Move to micromamba setup.
    # save_env = save_conda_env.submit(
    #     output_dir=join(exp.get_root_dir(), exp.get_name(), "processing")
    # )

    save_sys = save_system_information.submit(
        output_dir=join(exp.get_root_dir(), exp.get_name(), "processing")
    )

    run_context = save_params.submit(
        output_dir=join(exp.get_root_dir(), exp.get_name(), "processing"), params=params
    )

    section_paths = get_sections.submit(
        exp=exp,
        sample_name=exp_config.sample_name,
        acquisition=exp_config.acquisition,
        tile_grid_num=exp_config.tile_grid_num,
        start_section_num=exp_config.start_section_num,
        end_section_num=exp_config.end_section_num,
    ).result()

    logger.info(f"Found {len(section_paths)} sections.")

    integration_config = build_integration_config.submit(
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
    )

    meshes = run_sofima.map(
        section_dict=section_paths,
        stride=unmapped(mesh_integration_config.stride),
        overlaps_x=unmapped(tuple(registration_config.overlaps_x)),
        overlaps_y=unmapped(tuple(registration_config.overlaps_y)),
        min_overlap=unmapped(registration_config.min_overlap),
        min_range=unmapped(tuple(registration_config.min_range)),
        patch_size=unmapped(tuple(registration_config.patch_size)),
        batch_size=unmapped(registration_config.batch_size),
        min_peak_ratio=unmapped(registration_config.min_peak_ratio),
        min_peak_sharpness=unmapped(registration_config.min_peak_sharpness),
        max_deviation=unmapped(registration_config.max_deviation),
        max_magnitude=unmapped(registration_config.max_magnitude),
        min_patch_size=unmapped(registration_config.min_patch_size),
        max_gradient=unmapped(registration_config.max_gradient),
        reconcile_flow_max_deviation=unmapped(
            registration_config.reconcile_flow_max_deviation
        ),
        integration_config=unmapped(integration_config),
    )

    for m in meshes:
        m.wait()

    commit_changes.submit(
        exp=exp,
        wait_for=[exp, save_sys, run_context],
    )
