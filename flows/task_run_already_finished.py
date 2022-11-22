import time

from prefect import flow, get_run_logger, task
from prefect_dask import DaskTaskRunner


@task()
def get_list(n=500):
    return range(1, n)


@task(cache_result_in_memory=False)
def my_task(id: int):
    logger = get_run_logger()
    logger.info(f"Hello from my_task #{id}! Sleeping for 30 seconds...")
    time.sleep(3)
    logger.info(f"my_task #{id} done!")
    return id


@task()
def done():
    get_run_logger().info("Done")


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
    persist_result=False,
)
def my_flow():
    logger = get_run_logger()
    logger.info("Starting")

    ids = get_list()
    tasks = my_task.map(ids)

    for t in tasks:
        t.wait()

    logger.info("Done submitting")

    done()
