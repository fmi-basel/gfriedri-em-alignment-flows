import time

from prefect import flow, get_run_logger, task
from prefect_dask import DaskTaskRunner


@task
async def my_task(id: int):
    logger = get_run_logger()
    logger.info(f"Hello from my_task #{id}! Sleeping for 30 seconds...")
    time.sleep(30)
    logger.info(f"my_task #{id} done!")


@flow(
    name="task run already finished",
    task_runner=DaskTaskRunner(
        cluster_class="dask_jobqueue.SLURMCluster",
        cluster_kwargs={
            "account": "dlthings",
            "cores": 1,
            "processes": 1,
            "memory": "512 MB",
            "walltime": "00:10:00",
            "job_extra_directives": [
                "--ntasks=1",
                "--output=/tungstenfs/scratch/gmicro_share/_prefect/slurm/task_already_exists_output/%j.out",
            ],
            "worker_extra_args": ["--lifetime", "8m", "--lifetime-stagger", "2m"],
        },
        adapt_kwargs={
            "minimum": 1,
            "maximum": 32,
        },
    ),
)
def my_flow():
    logger = get_run_logger()
    logger.info("Starting")
    # Spam tasks here!
    for i in range(1, 50):
        logger.debug("Starting a task run #%d", i)
        my_task.submit(i + 1)
        logger.debug("Sleeping after task run submission for 5 sec")
        time.sleep(5)
