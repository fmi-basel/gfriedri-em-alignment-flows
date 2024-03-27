import argparse
from glob import glob
from os.path import exists, join

import yaml
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
                    tile_height=tile_spec["tile_height"],
                    tile_width=tile_spec["tile_width"],
                    tile_overlap=tile_spec["overlap"],
                )
                sections[section_name] = section

            assert (
                tile_spec["tile_height"] == section.get_tile_height()
            ), f"Tile height is off in section {section.get_name()}."
            assert (
                tile_spec["tile_width"] == section.get_tile_width()
            ), f"Tile width is off in section {section.get_name()}."
            assert (
                tile_spec["overlap"] == section.get_tile_overlap()
            ), f"Tile overlap is off in section {section.get_name()}."
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
    output_dir: str,
    acquisition_conf: AcquisitionConfig = AcquisitionConfig(),
    start_section: int = 0,
    end_section: int = 10,
):
    section_yaml_files = parse_data(
        output_dir=output_dir,
        sbem_root_dir=acquisition_conf.sbem_root_dir,
        acquisition=acquisition_conf.acquisition,
        tile_grid=acquisition_conf.tile_grid,
        thickness=acquisition_conf.thickness,
        resolution_xy=acquisition_conf.resolution_xy,
        start_section=start_section,
        end_section=end_section,
    )
    chunk_size = 2
    for chunk, i in enumerate(range(0, len(section_yaml_files), chunk_size)):
        section_yaml_files_chunk = section_yaml_files[i : i + chunk_size]
        with open(f"section_yaml_files_chunk_{chunk}.yaml", "w") as f:
            yaml.safe_dump(section_yaml_files_chunk, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="tile-stitching.config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(
        output_dir=join(config["output_dir"], "sections"),
        acquisition_conf=AcquisitionConfig(**config["acquisition_config"]),
        start_section=config["start_section"],
        end_section=config["end_section"],
    )
