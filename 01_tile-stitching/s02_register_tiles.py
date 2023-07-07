from glob import glob
from os.path import dirname, exists, join

from pydantic import BaseModel
from sbem.record.Section import Section
from sbem.tile_stitching.sofima_utils import register_tiles
from sofima import mesh
from tqdm import tqdm


class MeshIntegrationConfig(BaseModel):
    dt: float = 0.001
    gamma: float = 0.0
    k0: float = 0.01
    k: float = 0.1
    stride: int = 20
    num_iters: int = 1000
    max_iters: int = 20000
    stop_v_max: float = 0.001
    dt_max: float = 100.0
    prefer_orig_order: bool = True
    start_cap: float = 1.0
    final_cap: float = 10.0
    remove_drift: bool = True


class RegistrationConfig(BaseModel):
    overlaps_x: list[int] = [200, 300, 400]
    overlaps_y: list[int] = [200, 300, 400]
    min_overlap: int = 20
    min_range: list[int] = [10, 100, 0]
    patch_size: list[int] = [80, 80]
    batch_size: int = 8000
    min_peak_ratio: float = 1.4
    min_peak_sharpness: float = 1.4
    max_deviation: int = 5
    max_magnitude: int = 0
    min_patch_size: int = 10
    max_gradient: float = -1.0
    reconcile_flow_max_deviation: float = -1.0


def run_tile_registration(
    section_yaml_file: str,
    mesh_integration_config: MeshIntegrationConfig = MeshIntegrationConfig(),
    registration_config: RegistrationConfig = RegistrationConfig(),
):
    return register_tiles(
        section=Section.load_from_yaml(path=section_yaml_file),
        section_dir=dirname(section_yaml_file),
        stride=mesh_integration_config.stride,
        overlaps_x=tuple(registration_config.overlaps_x),
        overlaps_y=tuple(registration_config.overlaps_y),
        min_overlap=registration_config.min_overlap,
        min_range=tuple(registration_config.min_range),
        patch_size=tuple(registration_config.patch_size),
        batch_size=registration_config.batch_size,
        min_peak_ratio=registration_config.min_peak_ratio,
        min_peak_sharpness=registration_config.min_peak_sharpness,
        max_deviation=registration_config.max_deviation,
        max_magnitude=registration_config.max_magnitude,
        min_patch_size=registration_config.min_patch_size,
        max_gradient=registration_config.max_gradient,
        reconcile_flow_max_deviation=registration_config.reconcile_flow_max_deviation,
        integration_config=mesh.IntegrationConfig(
            dt=mesh_integration_config.dt,
            gamma=mesh_integration_config.gamma,
            k0=mesh_integration_config.k0,
            k=mesh_integration_config.k,
            stride=mesh_integration_config.stride,
            num_iters=mesh_integration_config.num_iters,
            max_iters=mesh_integration_config.max_iters,
            stop_v_max=mesh_integration_config.stop_v_max,
            dt_max=mesh_integration_config.dt_max,
            prefer_orig_order=mesh_integration_config.prefer_orig_order,
            start_cap=mesh_integration_config.start_cap,
            final_cap=mesh_integration_config.final_cap,
            remove_drift=mesh_integration_config.remove_drift,
        ),
    )


def filter_ignore(files, file_name="section.yaml"):
    keep = []
    for file in files:
        if not exists(file.replace(file_name, "IGNORE")):
            keep.append(file)

    return keep


def main(
    section_dir: str,
    mesh_integration_config: MeshIntegrationConfig = MeshIntegrationConfig(),
    registration_config: RegistrationConfig = RegistrationConfig(),
):
    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    section_yaml_files = glob(join(section_dir, "*", "section.yaml"))
    section_yaml_files = filter_ignore(section_yaml_files)

    for section_yaml_file in tqdm(section_yaml_files):
        run_tile_registration(
            section_yaml_file=section_yaml_file,
            mesh_integration_config=mesh_integration_config,
            registration_config=registration_config,
        )


if __name__ == "__main__":
    main(
        section_dir="/home/tibuch/Data/gfriedri/2023-refactor/sections",
    )
