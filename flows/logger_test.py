if False:  # TYPE_CHECKING
    from typing import Literal

import numpy as np
from prefect import flow, get_run_logger, task
from prefect_dask import DaskTaskRunner
from pydantic import BaseModel

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


# runner = DaskTaskRunner(client_kwargs={})


@task()
def a_task(i):
    logger = get_run_logger()
    logger.info(f"I am logging from a task. {i}")
    img = np.random.rand(1000, 1000)
    for y in range(1000):
        for x in range(1000):
            img[y, x] = img[x, y]

    logger.info("Done.")
    # sleep(5)
    # os.system("sleep 5")


class FMI_User(BaseModel):
    group: Literal["gmicro", "gfriedri", "gchao"]
    user: str


@flow(
    name="Logger Test",
    task_runner=runner,
)
def flow(group_user: FMI_User):
    logger = get_run_logger()
    logger.info("Flow starts!")

    a_task.map(range(3))

    a_task.submit(4)

    logger.info("Flow ends!")


if __name__ == "__main__":
    flow()
