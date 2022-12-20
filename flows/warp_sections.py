import json
from os.path import exists, join
from typing import Dict

import git
import numpy as np
import prefect
from prefect import flow, get_run_logger, task
from prefect_dask import DaskTaskRunner
from sbem.experiment import Experiment
from sbem.record.Sample import Sample
from sbem.record.Section import Section
from sbem.storage.Volume import Volume
from sbem.tile_stitching.sofima_utils import render_tiles
from utils.env import save_conda_env
from utils.system import save_system_information

from flows.flow_parameter_types.ExperimentConfig import ExperimentConfig
from flows.flow_parameter_types.WarpConfig import WarpConfig


@task(cache_result_in_memory=False)
def load_experiment(path: str):
    return Experiment.load(path)


@task(cache_result_in_memory=False)
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

    outpath = join(output_dir, "args_warp-sections.json")

    with open(outpath, "w") as f:
        json.dump(params, f, indent=4)

    logger.info(f"Saved flow parameters to {outpath}.")


@task()
def commit_changes(exp: Experiment):
    with git.Repo(join(exp.get_root_dir(), exp.get_name())) as repo:
        repo.index.add(repo.untracked_files)
        repo.index.add([item.a_path for item in repo.index.diff(None)])
        repo.index.commit("Warp sections and save to volume.", author=exp._git_author)


def get_origin_tile_id(tile_id_map):
    for x in range(tile_id_map.shape[1]):
        for y in range(tile_id_map.shape[0]):
            if tile_id_map[y, x] != -1:
                return tile_id_map[y, x]


def get_top_most_tile_id(tile_id_map):
    for y in range(tile_id_map.shape[0]):
        for x in range(tile_id_map.shape[1]):
            if tile_id_map[y, x] != -1:
                return tile_id_map[y, x]


def get_left_most_tile_id(tile_id_map):
    for x in range(tile_id_map.shape[1]):
        for y in range(tile_id_map.shape[0]):
            if tile_id_map[y, x] != -1:
                return tile_id_map[y, x]


def get_section_stage_coords(section):
    top_most_tile_id = get_top_most_tile_id(section.get_tile_id_map())
    left_most_tile_id = get_left_most_tile_id(section.get_tile_id_map())

    top_most_tile = section.get_tile(top_most_tile_id)
    left_most_tile = section.get_tile(left_most_tile_id)

    return np.array([top_most_tile.y, left_most_tile.x])


def get_volume_z_pos(volume: Volume, section: Section):
    for z, section_num in enumerate(volume._section_list):
        if section_num > section.get_section_num():
            return z

    return len(volume._section_list)


@task(cache_result_in_memory=False, retries=1, retry_delay_seconds=30)
def warp_and_save(
    sample: Sample,
    tile_grid_num: int,
    section,
    volume,
    stride: int,
    margin: int,
    use_clahe: bool,
    clahe_kwargs: Dict,
):
    logger = get_run_logger()
    if not section.is_stitched():
        section.load_from_yaml()
        if not exists(
            join(
                section.get_sample().get_experiment().get_root_dir(),
                section.get_sample().get_experiment().get_name(),
                section.get_sample().get_name(),
                section.get_name(),
                "meshes.npz",
            )
        ):
            logger.info("Mesh not found. Please run tile-registration first.")
            return None

        logger.info(f"Warp section {section.get_name()}.")
        warped_tiles, mask = render_tiles(
            section=section,
            stride=stride,
            margin=margin,
            parallelism=32,
            use_clahe=use_clahe,
            clahe_kwargs=clahe_kwargs,
        )

        if warped_tiles is not None:
            if len(volume._section_list) == 0:
                logger.info("First section insert at [0, 0, 0].")
                volume.write_section(
                    section_num=section.get_section_num(),
                    data=warped_tiles[np.newaxis],
                    offsets=tuple([0, 0, 0]),
                )
            else:
                zeroth_section_num = volume._section_list[0]
                zeroth_section = sample.get_section(
                    f"s{zeroth_section_num}_" f"g{tile_grid_num}"
                )
                zeroth_section.load_from_yaml()

                zeroth_section_stage_coords = get_section_stage_coords(zeroth_section)
                current_section_stage_coords = get_section_stage_coords(section)

                offset_yx = current_section_stage_coords - zeroth_section_stage_coords

                z_index = get_volume_z_pos(volume, section)
                offset = tuple([z_index, int(offset_yx[0]), int(offset_yx[1])])

                logger.info(f"Insert section at {offset}.")

                volume.write_section(
                    section_num=section.get_section_num(),
                    data=warped_tiles[np.newaxis],
                    offsets=offset,
                )

            logger.info(
                f"Section {section.get_name()} successfully stitched and saved."
            )
            section._stitched = True

    return volume


@task()
def get_volume(exp: Experiment, sample_name: str):
    return Volume.load(exp.get_sample(sample_name).get_aligned_data())


@flow(
    name="Warp Sections",
    task_runner=DaskTaskRunner(
        cluster_class="dask_jobqueue.SLURMCluster",
        cluster_kwargs={
            "account": "dlthings",
            "queue": "cpu_long",
            "cores": 32,
            "processes": 1,
            "memory": "32 GB",
            "walltime": "24:00:00",
            "job_extra_directives": [
                "--ntasks=1",
                "--output=/tungstenfs/scratch/gmicro_share/_prefect/slurm/gfriedri-em-alignment-flows/output/%j.out",
            ],
            "worker_extra_args": [
                "--lifetime",
                "1440m",
                "--lifetime-stagger",
                "10m",
            ],
            "job_script_prologue": [
                "conda run -p /tungstenfs/scratch/gmicro_share/_prefect/miniconda3/envs/airtable python /tungstenfs/scratch/gmicro_share/_prefect/airtable/log-slurm-job.py --config /tungstenfs/scratch/gmicro/_prefect/airtable/slurm-job-log.ini"
            ],
        },
        adapt_kwargs={
            "minimum": 1,
            "maximum": 1,
        },
    ),
    result_storage="local-file-system/gfriedri-em-alignment-flows-storage",
    persist_result=True,
)
def warp_sections_flow(
    exp_config: ExperimentConfig = ExperimentConfig(),
    warp_config: WarpConfig = WarpConfig(),
):
    params = {
        "exp_config": exp_config.dict(),
        "warp_config": warp_config.dict(),
    }
    exp: Experiment = load_experiment.submit(path=exp_config.exp_path).result()

    assert exists(exp.get_sample(exp_config.sample_name).get_aligned_data())

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
        sample_name=exp_config.sample_name,
        acquisition=exp_config.acquisition,
        tile_grid_num=exp_config.tile_grid_num,
        start_section_num=exp_config.start_section_num,
        end_section_num=exp_config.end_section_num,
    ).result()

    volume = get_volume.submit(exp=exp, sample_name=exp_config.sample_name).result()

    for section in sections:
        try:
            volume = warp_and_save.submit(
                sample=exp.get_sample(exp_config.sample_name),
                tile_grid_num=exp_config.tile_grid_num,
                section=section,
                volume=volume,
                stride=warp_config.stride,
                margin=warp_config.margin,
                use_clahe=warp_config.use_clahe,
                clahe_kwargs={
                    "kernel_size": warp_config.kernel_size,
                    "clip_limit": warp_config.clip_limit,
                    "nbins": warp_config.nbins,
                },
            ).result()
        except prefect.exceptions.CrashedRun:
            # Re-submit if slurm node changes.
            volume = warp_and_save.submit(
                sample=exp.get_sample(exp_config.sample_name),
                tile_grid_num=exp_config.tile_grid_num,
                section=section,
                volume=volume,
                stride=warp_config.stride,
                margin=warp_config.margin,
                use_clahe=warp_config.use_clahe,
                clahe_kwargs={
                    "kernel_size": warp_config.kernel_size,
                    "clip_limit": warp_config.clip_limit,
                    "nbins": warp_config.nbins,
                },
            ).result()

        exp.get_sample(exp_config.sample_name).get_section(
            section.get_name()
        )._stitched = (volume is not None)
        exp.save(overwrite=True)
        volume.save()

    commit_changes.submit(
        exp=exp,
        wait_for=[exp, save_env, save_sys, run_context],
    )
