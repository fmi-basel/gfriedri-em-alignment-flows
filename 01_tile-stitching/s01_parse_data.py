from glob import glob
from os.path import exists, join

from parameter_config import AcquisitionConfig
from sbem.experiment.parse_utils import get_tile_metadata
from sbem.record.Section import Section
from sbem.record.Tile import Tile


def section_in_range(name: str, start: int, end: int) -> bool:
    s, _ = name.split("_")
    z = int(s[1:])
    return start <= z <= end


def parse_data(
    output_dir: str,
    sbem_root_dir: str,
    acquisition: str,
    tile_grid: str,
    thickness: float,
    resolution_xy: float,
    tile_width: int,
    tile_height: int,
    tile_overlap: int,
    start_section: int,
    end_section: int,
):
    tile_grid_num = int(tile_grid[1:])

    metadata_files = sorted(glob(join(sbem_root_dir, "meta", "logs", "metadata_*")))

    tile_specs = get_tile_metadata(
        sbem_root_dir, metadata_files, tile_grid_num, resolution_xy
    )

    sections = {}
    existing_sections = glob(join(output_dir, "*", "section.yaml"))
    for es in existing_sections:
        section = Section.load_from_yaml(es)
        if section_in_range(section.get_name(), start_section, end_section):
            sections[section.get_name()] = section

    for tile_spec in tile_specs:
        section_name = f"s{tile_spec['z']}_g{tile_grid_num}"
        if section_in_range(section_name, start_section, end_section):
            if section_name in sections.keys():
                section = sections[section_name]
            else:
                section = Section(
                    acquisition=acquisition,
                    section_num=tile_spec["z"],
                    tile_grid_num=tile_grid_num,
                    thickness=thickness,
                    tile_height=tile_height,
                    tile_width=tile_width,
                    tile_overlap=tile_overlap,
                )
                sections[section_name] = section

            Tile(
                section,
                tile_id=tile_spec["tile_id"],
                path=tile_spec["tile_file"],
                stage_x=tile_spec["x"],
                stage_y=tile_spec["y"],
                resolution_xy=resolution_xy,
                unit="nm",
            )

    section_paths = []
    for section in sections.values():
        section.save(path=output_dir, overwrite=True)
        tile_id_map_path = join(output_dir, section.get_name(), "tile_id_map.json")
        if not exists(tile_id_map_path):
            section.get_tile_id_map(path=tile_id_map_path)

        section_paths.append(join(output_dir, section.get_name(), "section.yaml"))

    return section_paths


def main(
    section_dir: str,
    acquisition_conf: AcquisitionConfig = AcquisitionConfig(),
    start_section: int = 0,
    end_section: int = 10,
):
    _ = parse_data(
        output_dir=section_dir,
        sbem_root_dir=acquisition_conf.sbem_root_dir,
        acquisition=acquisition_conf.acquisition,
        tile_grid=acquisition_conf.tile_grid,
        thickness=acquisition_conf.thickness,
        resolution_xy=acquisition_conf.resolution_xy,
        tile_width=acquisition_conf.tile_width,
        tile_height=acquisition_conf.tile_height,
        tile_overlap=acquisition_conf.tile_overlap,
        start_section=start_section,
        end_section=end_section,
    )


if __name__ == "__main__":
    main(
        section_dir="/home/tibuch/Data/gfriedri/2023-refactor/sections",
        sbem_root_dir="/tungstenfs/scratch/gmicro/prefect-test/gfriedri-em"
        "-alignment-flows/test-data/",
        acquisition="run_0",
        tile_grid="g0001",
        thickness=25,
        resolution_xy=11,
        tile_width=3072,
        tile_height=2304,
        tile_overlap=220,
        start_section=1074,
        end_section=1098,
    )
