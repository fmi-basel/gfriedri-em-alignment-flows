import json
from os.path import join
from typing import Dict

import git
from prefect import flow, task
from prefect.logging.loggers import get_run_logger
from prefect_dask import DaskTaskRunner
from sbem.experiment import Experiment
from sbem.storage.Volume import Volume
from utils.env import save_conda_env
from utils.system import save_system_information


@task()
def load_experiment(path: str):
    return Experiment.load(path)


@task()
def create_volume(
    volume_name: str,
    description: str,
    exp: Experiment,
    sample_name: str,
):
    vol = Volume(
        name=volume_name,
        description=description,
        documentation="",
        authors=exp._authors,
        root_dir=join(exp.get_root_dir(), exp.get_name()),
        exist_ok=False,
        license=exp.get_license(),
        cite=exp._cite,
    )

    exp.get_sample(sample_name)._aligned_data = join(vol.get_dir(), "volume.yaml")
    exp.save(overwrite=True)

    return vol


@task()
def save_params(output_dir: str, params: Dict):
    """
    Dump prefect context into prefect-context.json.
    :param output_dir:
    :param context_dict:
    :return:
    """
    logger = get_run_logger()

    outpath = join(output_dir, "args_add-volume.json")
    with open(outpath, "w") as f:
        json.dump(params, f, indent=4)

    logger.info(f"Saved flow parameters to {outpath}.")


@task()
def commit_changes(exp: Experiment, volume_name: str, name: str):
    with git.Repo(join(exp.get_root_dir(), exp.get_name())) as repo:
        repo.index.add(repo.untracked_files)
        repo.index.add([item.a_path for item in repo.index.diff(None)])
        repo.index.commit(
            f"Add volume {volume_name} to sample '{name}'.", author=exp._git_author
        )


@task
def add_gitignore(output_dir: str):
    with open(join(output_dir, ".gitignore"), "a") as f:
        f.writelines(["ngff_volume.zarr/\n"])


@flow(
    name="Add Volume",
    task_runner=DaskTaskRunner(
        cluster_class="dask_jobqueue.SLURMCluster",
        cluster_kwargs={
            "account": "dlthings",
            "cores": 2,
            "processes": 1,
            "memory": "12 GB",
            "walltime": "00:30:00",
            "job_extra_directives": [
                "--ntasks=1",
                "--output=/tungstenfs/scratch/gmicro_share/_prefect/slurm/output/%j.out",
            ],
            "worker_extra_args": ["--lifetime", "30m", "--lifetime-stagger", "5m"],
            "job_script_prologue": [
                "conda run -p /tungstenfs/scratch/gmicro_share/_prefect/miniconda3/envs/airtable python /tungstenfs/scratch/gmicro_share/_prefect/airtable/log-slurm-job.py --config /tungstenfs/scratch/gmicro/_prefect/airtable/slurm-job-log.ini"
            ],
        },
        adapt_kwargs={
            "minimum": 1,
            "maximum": 1,
        },
    ),
    persist_result=False,
)
def add_volume_flow(
    exp_path: str = "/path/to/experiment.yaml",
    sample_name: str = "Sample",
    volume_name: str = "Volume",
    description: str = "An aligned volume.",
):
    params = dict(locals())
    exp: Experiment = load_experiment.submit(path=exp_path).result()

    volume = create_volume(
        volume_name=volume_name,
        description=description,
        exp=exp,
        sample_name=sample_name,
    )

    save_env = save_conda_env.submit(
        output_dir=join(exp.get_root_dir(), exp.get_name(), "processing")
    )

    save_sys = save_system_information.submit(
        output_dir=join(exp.get_root_dir(), exp.get_name(), "processing")
    )

    run_context = save_params.submit(
        output_dir=join(exp.get_root_dir(), exp.get_name(), "processing"), params=params
    )

    gitignore = add_gitignore.submit(
        output_dir=join(exp.get_root_dir(), exp.get_name())
    )

    commit_changes.submit(
        exp=exp,
        volume_name=volume_name,
        name=sample_name,
        wait_for=[exp, volume, save_env, save_sys, run_context, gitignore],
    )
