import argparse
import os
import re
from os.path import basename
from pathlib import Path

import yaml


def list_zarr_sections(root_dir: str) -> list[str]:
    filename_re = re.compile(r"s[0-9]*_g[0-9]*.zarr")
    files = []
    root, dirs, _ = next(os.walk(root_dir))
    for d in dirs:
        m_filename = filename_re.fullmatch(d)
        if m_filename:
            files.append(str(Path(root).joinpath(d)))

    files.sort(key=lambda v: int(basename(v).split("_")[0][1:]))
    return files


def filter_sections(section_dirs: list[str], start_section: int, end_section: int):
    kept = []
    for sec in section_dirs:
        sec_idx = int(basename(sec).split("_")[0][1:])
        if start_section <= sec_idx <= end_section:
            kept.append(sec)

    return kept


def main(
    stitched_sections_dir: str,
    start_section: int,
    end_section: int,
):
    section_dirs = list_zarr_sections(
        root_dir=stitched_sections_dir,
    )
    section_dirs = filter_sections(
        section_dirs=section_dirs,
        start_section=start_section,
        end_section=end_section,
    )

    chunk_size = 2
    for chunk, i in enumerate(range(0, len(section_dirs), chunk_size)):
        start = max(0, i - 1)
        end = min(len(section_dirs), i + chunk_size)
        section_dirs_chunk = section_dirs[start:end]
        with open(f"section_dirs_chunk_{chunk}.yaml", "w") as f:
            yaml.safe_dump(section_dirs_chunk, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, default="coarse-align.config"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(
        stitched_sections_dir=config["stitched_sections_dir"],
        start_section=config["start_section"],
        end_section=config["end_section"],
    )
