from typing import List, Tuple

from prefect import flow, get_run_logger, task
from prefect_dask import DaskTaskRunner


@task()
def a_task():
    logger = get_run_logger()

    logger.info("I am logging from a task.")


runner = DaskTaskRunner(
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
)

# runner = DaskTaskRunner()


@flow(
    name="Logger Test",
    # task_runner=runner,
)
def flow(a_tuple: Tuple[int, int], a_list: List[int]):
    logger = get_run_logger()
    logger.info("Flow starts!")

    a_task.submit()

    logger.info("Flow ends!")


if __name__ == "__main__":
    flow()
