import json
from os import makedirs
from os.path import join
from typing import Dict, List, Tuple

import git
from prefect import flow, get_run_logger, task, unmapped
from prefect_dask import DaskTaskRunner
from sbem.experiment.Experiment import Experiment
from sbem.record.Section import Section
from utils.env import save_conda_env
from utils.system import save_system_information

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
    if start_section_num is not None and end_section_num is not None:
        return exp.get_sample(sample_name).get_section_range(
            start_section_num=start_section_num,
            end_section_num=end_section_num,
            tile_grid_num=tile_grid_num,
            include_skipped=False,
        )
    else:
        return exp.get_sample(sample_name).get_sections_of_acquisition(
            acquisition=acquisition, tile_grid_num=tile_grid_num, include_skipped=False
        )


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


@task()
def log_sections_without_mesh(sections: List[Section]):
    names = []
    for sec in sections:
        if sec.get_alignment_mesh() is None:
            names.append(sec.get_name())

    sname = sections[0].get_sample().get_name()
    file_name = f"{sname}_failed_sections.json"
    path = join(
        sections[0].get_sample().get_experiment().get_root_dir(),
        sections[0].get_sample().get_experiment().get_name(),
        "tile-registration-errors",
    )
    makedirs(path, exist_ok=True)
    with open(join(path, file_name), "w") as f:
        json.dump(names, f, indent=4)


@flow(
    name="Register Tiles",
    task_runner=DaskTaskRunner(
        cluster_class="dask_jobqueue.SLURMCluster",
        cluster_kwargs={
            "account": "dlthings",
            "queue": "gpu_long",
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
            "maximum": 32,
        },
    ),
    result_storage="local-file-system/gfriedri-em-alignment-flows-storage",
    persist_result=False,
)
def tile_registration_flow(
    exp_path: str = "/path/to/experiment.yaml",
    sample_name: str = "Sample",
    acquisition: str = "run_0",
    start_section_num: int = None,
    end_section_num: int = None,
    tile_grid_num: int = 1,
    dt: float = 0.001,
    gamma: float = 0.0,
    k0: float = 0.01,
    k: float = 0.1,
    stride: int = 20,
    num_iters: int = 1000,
    max_iters: int = 20000,
    stop_v_max: float = 0.001,
    dt_max: float = 100.0,
    prefer_orig_order: bool = True,
    start_cap: float = 1.0,
    final_cap: float = 10.0,
    remove_drift: bool = True,
    overlaps_x: Tuple[int, int] = tuple([200, 300]),
    overlaps_y: Tuple[int, int] = tuple([200, 300]),
    min_overlap: int = 20,
    patch_size: Tuple[int, int] = tuple([120, 120]),
    batch_size: int = 8000,
    min_peak_ratio: float = 1.4,
    min_peak_sharpness: float = 1.4,
    max_deviation: float = 5.0,
    max_magnitude: float = 0.0,
    min_patch_size: int = 10,
    max_gradient: float = -1.0,
    reconcile_flow_max_deviation: float = -1.0,
    persist_result=False,
):
    params = dict(locals())
    logger = get_run_logger()
    exp = load_experiment.submit(path=exp_path).result()

    save_env = save_conda_env.submit(
        output_dir=join(exp.get_root_dir(), exp.get_name(), "processing")
    )

    save_sys = save_system_information.submit(
        output_dir=join(exp.get_root_dir(), exp.get_name(), "processing")
    )

    run_context = save_params.submit(
        output_dir=join(exp.get_root_dir(), exp.get_name(), "processing"), params=params
    )

    sections = get_sections.submit(
        exp=exp,
        sample_name=sample_name,
        acquisition=acquisition,
        tile_grid_num=tile_grid_num,
        start_section_num=start_section_num,
        end_section_num=end_section_num,
    ).result()

    logger.info(f"Found {len(sections)} sections.")

    integration_config = build_integration_config.submit(
        dt=dt,
        gamma=gamma,
        k0=k0,
        k=k,
        stride=stride,
        num_iters=num_iters,
        max_iters=max_iters,
        stop_v_max=stop_v_max,
        dt_max=dt_max,
        prefer_orig_order=prefer_orig_order,
        start_cap=start_cap,
        final_cap=final_cap,
        remove_drift=remove_drift,
    )

    meshes = run_sofima.map(
        sections,
        stride=unmapped(stride),
        overlaps_x=unmapped(overlaps_x),
        overlaps_y=unmapped(overlaps_y),
        min_overlap=unmapped(min_overlap),
        patch_size=unmapped(patch_size),
        batch_size=unmapped(batch_size),
        min_peak_ratio=unmapped(min_peak_ratio),
        min_peak_sharpness=unmapped(min_peak_sharpness),
        max_deviation=unmapped(max_deviation),
        max_magnitude=unmapped(max_magnitude),
        min_patch_size=unmapped(min_patch_size),
        max_gradient=unmapped(max_gradient),
        reconcile_flow_max_deviation=unmapped(reconcile_flow_max_deviation),
        integration_config=unmapped(integration_config),
    )

    failed_ones = log_sections_without_mesh.submit(sections=meshes)

    commit_changes.submit(
        exp=exp,
        wait_for=[exp, failed_ones, save_env, save_sys, run_context],
    )

    return meshes
