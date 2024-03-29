import argparse
import json
from os.path import join
from typing import Dict

import git
from prefect import flow, get_run_logger, task
from prefect_dask import DaskTaskRunner
from sbem.experiment.Experiment import Experiment
from sbem.record.Sample import Sample
from utils.system import save_system_information


@task()
def load_experiment(path: str):
    return Experiment.load(path=path)


@task()
def add_sample(exp: Experiment, name: str, description: str):
    assert " " not in name, "Name contains spaces."
    assert exp.get_sample(name) is None, "Sample exists already."
    Sample(
        experiment=exp,
        name=name,
        description=description,
        documentation="",
        aligned_data="",
    )

    exp.save(overwrite=True)


@task()
def save_params(output_dir: str, params: Dict):
    """
    Dump prefect context into prefect-context.json.
    :param output_dir:
    :param context_dict:
    :return:
    """
    logger = get_run_logger()

    outpath = join(output_dir, "args_add-sample.json")
    with open(outpath, "w") as f:
        json.dump(params, f, indent=4)

    logger.info(f"Saved flow parameters to {outpath}.")


@task()
def commit_changes(exp: Experiment, name: str):
    with git.Repo(join(exp.get_root_dir(), exp.get_name())) as repo:
        repo.index.add(repo.untracked_files)
        repo.index.add([item.a_path for item in repo.index.diff(None)])
        repo.index.commit(f"Add sample '{name}'.", author=exp._git_author)


@flow(
    name="Add Sample",
    task_runner=DaskTaskRunner(
        cluster_class="dask_jobqueue.SLURMCluster",
        cluster_kwargs={
            "account": "dlthings",
            "cores": 2,
            "processes": 1,
            "memory": "512 MB",
            "walltime": "00:10:00",
            "job_extra_directives": [
                "--ntasks=1",
                "--output=/tungstenfs/scratch/gmicro_share/_prefect/slurm/gfriedri-em-alignment-flows/output/%j.out",
            ],
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
    persist_result=False,
)
def add_sample_to_experiment_flow(
    exp_path: str = "/path/to/experiment.yaml",
    name: str = "Sample",
    description: str = "My best sample.",
):
    params = dict(locals())
    exp = load_experiment.submit(path=exp_path).result()
    cs = add_sample.submit(exp=exp, name=name, description=description)

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

    commit_changes.submit(exp=exp, name=name, wait_for=[exp, cs, save_sys, run_context])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path")
    parser.add_argument("--name")
    parser.add_argument("--description")
    args = parser.parse_args()

    add_sample_to_experiment_flow(
        exp_path=args.exp_path, name=args.name, description=args.description
    )
