import gc
from os.path import join

import prefect
from prefect import get_run_logger, task
from sbem.experiment.Experiment import Experiment
from sbem.record.Section import Section
from sbem.tile_stitching.sofima_utils import (
    default_mesh_integration_config,
    load_sections,
)
from sofima import mesh


@task(cache_result_in_memory=False)
def run_sofima(
    section: Section,
    stride: int,
    overlaps_x: tuple,
    overlaps_y: tuple,
    min_overlap: int,
    patch_size: tuple = (120, 120),
    batch_size: int = 8000,
    min_peak_ratio: float = 1.4,
    min_peak_sharpness: float = 1.4,
    max_deviation: int = 5,
    max_magnitude: int = 0,
    min_patch_size: int = 10,
    max_gradient: float = -1,
    reconcile_flow_max_deviation: float = -1,
    integration_config: mesh.IntegrationConfig = default_mesh_integration_config(),
):
    logger = get_run_logger()
    sec_long_name = join(
        section.get_sample().get_experiment().get_name(),
        section.get_sample().get_name(),
        section.get_name(),
    )
    logger.info(f"Compute mesh for section {sec_long_name}.")

    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    from sbem.tile_stitching.sofima_utils import register_tiles

    section.load_from_yaml()
    if section.get_alignment_mesh() is None:
        try:
            register_tiles(
                section,
                stride=stride,
                overlaps_x=overlaps_x,
                overlaps_y=overlaps_y,
                min_overlap=min_overlap,
                patch_size=patch_size,
                batch_size=batch_size,
                min_peak_ratio=min_peak_ratio,
                min_peak_sharpness=min_peak_sharpness,
                max_deviation=max_deviation,
                max_magnitude=max_magnitude,
                min_patch_size=min_patch_size,
                max_gradient=max_gradient,
                reconcile_flow_max_deviation=reconcile_flow_max_deviation,
                integration_config=integration_config,
                logger=logger,
            )
            path = join(
                section.get_sample().get_experiment().get_root_dir(),
                section.get_sample().get_experiment().get_name(),
                section.get_sample().get_name(),
            )
            section.save(path, overwrite=True)
            return clear_memory(section)
        except Exception as e:
            logger.error(f"Encounter error in section " f"{section.get_section_dir()}.")
            raise e
    else:
        return clear_memory(section)


def clear_memory(section: Section):
    sec = Section.lazy_loading(
        name=section.get_name(),
        section_num=section.get_section_num(),
        tile_grid_num=section.get_tile_grid_num(),
        stitched=section.is_stitched(),
        skip=section.skip(),
        acquisition=section.get_acquisition(),
        details="",
    )
    sec.set_sample(section.get_sample())
    del section
    gc.collect()
    return sec


@task()
def run_warp_and_save(
    section: Section,
    stride: int,
    margin: int = 50,
    use_clahe: bool = False,
    clahe_kwargs: ... = None,
    parallelism: int = 1,
):
    logger = prefect.context.get("logger")
    logger.info(f"Warp and save section {section.save_dir}.")

    from sbem.tile_stitching.sofima_utils import render_tiles

    stitched, mask = render_tiles(
        section,
        stride=stride,
        margin=margin,
        parallelism=parallelism,
        use_clahe=use_clahe,
        clahe_kwargs=clahe_kwargs,
    )

    if stitched is not None and mask is not None:
        section.write_stitched(stitched=stitched, mask=mask)

    return section


@task()
def load_experiment_task(exp_path: str):
    return Experiment.load(path=exp_path)


@task()
def load_sections_task(exp: Experiment, sample_name: str, tile_grid_num: int):
    return load_sections(exp=exp, sample_name=sample_name, tile_grid_num=tile_grid_num)


@task()
def build_integration_config(
    dt,
    gamma,
    k0,
    k,
    stride,
    num_iters,
    max_iters,
    stop_v_max,
    dt_max,
    prefer_orig_order,
    start_cap,
    final_cap,
    remove_drift,
):
    logger = get_run_logger()
    logger.info(f"Integration config parameters: {locals()}")
    return mesh.IntegrationConfig(
        dt=float(dt),
        gamma=float(gamma),
        k0=float(k0),
        k=float(k),
        stride=int(stride),
        num_iters=int(num_iters),
        max_iters=int(max_iters),
        stop_v_max=float(stop_v_max),
        dt_max=float(dt_max),
        prefer_orig_order=bool(prefer_orig_order),
        start_cap=float(start_cap),
        final_cap=float(final_cap),
        remove_drift=bool(remove_drift),
    )
