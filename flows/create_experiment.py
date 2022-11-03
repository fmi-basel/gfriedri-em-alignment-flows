import argparse
import json
from os import mkdir
from os.path import exists, join
from typing import Dict

import git
from prefect.flows import flow
from prefect.logging import get_run_logger
from prefect.tasks import task
from prefect_dask import DaskTaskRunner
from sbem.experiment.Experiment import Experiment
from sbem.record.Author import Author
from sbem.record.Citation import Citation
from utils.env import save_conda_env
from utils.system import save_system_information


@task()
def create_experiment(name: str, description: str, root_dir: str) -> Experiment:
    assert " " not in name, "Name contains spaces."
    exp = Experiment(
        name=name,
        description=description,
        documentation="",
        authors=[Author(name="", affiliation="")],
        root_dir=root_dir,
        exist_ok=True,
        cite=[Citation(doi="", text="", url="")],
    )
    exp.save()

    if not exists(join(exp.get_root_dir(), exp.get_name(), "processing")):
        mkdir(join(exp.get_root_dir(), exp.get_name(), "processing"))

    return exp


@task()
def save_params(output_dir: str, params: Dict):
    """
    Dump prefect context into prefect-context.json.
    :param output_dir:
    :param context_dict:
    :return:
    """
    logger = get_run_logger()

    outpath = join(output_dir, "args_create-experiment.json")
    with open(outpath, "w") as f:
        json.dump(params, f, indent=4)

    logger.info(f"Saved flow parameters to {outpath}.")


@task()
def commit_changes(exp: Experiment):
    with git.Repo(join(exp.get_root_dir(), exp.get_name())) as repo:
        repo.index.add(repo.untracked_files)
        repo.index.add([item.a_path for item in repo.index.diff(None)])
        repo.index.commit("Create experiment.", author=exp._git_author)


@flow(
    name="Create Experiment",
    task_runner=DaskTaskRunner(
        cluster_class="dask_jobqueue.SLURMCluster",
        cluster_kwargs={
            "account": "dlthings",
            "cores": 2,
            "memory": "2 GB",
            "walltime": "00:10:00",
            "worker_extra_args": ["--lifetime", "8m", "--lifetime-stagger", "2m"],
            "job_script_prologue": [
                "conda run -p /tungstenfs/scratch/gmicro_share/_prefect/miniconda3/envs/airtable python /tungstenfs/scratch/gmicro_share/_prefect/airtable/log-slurm-job.py --config /tungstenfs/scratch/gmicro/_prefect/airtable/slurm-job-log.ini"
            ],
        },
        adapt_kwargs={
            "minimum": 1,
            "maximum": 1,
        },
    ),
)
def create_experiment_flow(
    name: str = "experiment",
    description: str = "Experiment to answer questions.",
    root_dir: str = "/tungstenfs/scratch/gmicro_sem",
    persist_result=False,
):
    params = dict(locals())
    exp = create_experiment.submit(
        name=name, description=description, root_dir=root_dir
    ).result()

    save_env = save_conda_env.submit(
        output_dir=join(exp.get_root_dir(), exp.get_name(), "processing")
    )

    save_sys = save_system_information.submit(
        output_dir=join(exp.get_root_dir(), exp.get_name(), "processing")
    )

    run_context = save_params.submit(
        output_dir=join(exp.get_root_dir(), exp.get_name(), "processing"), params=params
    )

    commit_changes.submit(exp=exp, wait_for=[exp, save_env, save_sys, run_context])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--description")
    parser.add_argument("--root_dir")
    args = parser.parse_args()

    create_experiment_flow(
        name=args.name, description=args.description, root_dir=args.root_dir
    )
