import argparse
import json
import os
import re
from os.path import basename, join, splitext
from pathlib import Path

import numpy as np
import yaml
import zarr
from connectomics.common import bounding_box
from numpy._typing import ArrayLike
from ome_zarr.io import parse_url
from parameter_config import FlowFieldEstimationConfig
from skimage.measure import block_reduce
from sofima import flow_field, flow_utils, map_utils


def get_yx_size(section_dirs: list[str], bin: int = 1):

    max_size_y = 0
    max_size_x = 0
    for sec_dir in section_dirs:
        with open(join(sec_dir, "0", ".zarray")) as f:
            shape = json.load(f)["shape"]

        with open(join(sec_dir, "coarse_stack_padding.json")) as f:
            shifts = json.load(f)
            shift_y = shifts["shift_y"]
            shift_x = shifts["shift_x"]

        size_y = shape[0] + shift_y
        size_x = shape[1] + shift_x
        if size_y > max_size_y:
            max_size_y = size_y
        if size_x > max_size_x:
            max_size_x = size_x

    max_size_y = max_size_y - max_size_y % bin
    max_size_x = max_size_x - max_size_x % bin

    assert max_size_y % bin == 0, "yx_size must be divisible by bin."
    assert max_size_x % bin == 0, "yx_size must be divisible by bin."

    return max_size_y, max_size_x


def filter_sections(section_dirs: list[str], start_section: int, end_section: int):
    kept = []
    for sec in section_dirs:
        sec_idx = int(basename(sec).split("_")[0][1:])
        if start_section <= sec_idx <= end_section:
            kept.append(sec)

    return kept


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


def load_section_data(section_dir: str, yx_size: tuple[int, int]) -> ArrayLike:
    with open(join(section_dir, "coarse_stack_padding.json")) as f:
        config = json.load(f)
        pad_y = config["shift_y"]
        pad_x = config["shift_x"]

    zarray = zarr.Group(parse_url(section_dir).store)[0]

    data = np.zeros(yx_size, dtype=zarray.dtype)
    data[pad_y : pad_y + zarray.shape[0], pad_x : pad_x + zarray.shape[1]] = zarray[:]

    return data


def clean_flow(
    flow,
    patch_size: int,
    stride: int,
    min_peak_ratio: float = 1.6,
    min_peak_sharpness: float = 1.6,
    max_magnitude: float = 80,
    max_deviation: float = 20,
):
    flow = flow[np.newaxis]
    flow = np.transpose(flow, [1, 0, 2, 3])
    pad = patch_size // 2 // stride
    flow = np.pad(
        flow, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan
    )

    return flow_utils.clean_flow(
        flow,
        min_peak_ratio=min_peak_ratio,
        min_peak_sharpness=min_peak_sharpness,
        max_magnitude=max_magnitude,
        max_deviation=max_deviation,
    )


def compute_final_flow(
    f1: ArrayLike,
    f2: ArrayLike,
    max_gradient: float,
    max_deviation: float,
    min_patch_size: int,
):
    f2_hires = np.zeros_like(f1)

    scale = 0.5
    box1x = bounding_box.BoundingBox(
        start=(0, 0, 0), size=(f1.shape[-1], f1.shape[-2], 1)
    )
    box2x = bounding_box.BoundingBox(
        start=(0, 0, 0), size=(f2.shape[-1], f2.shape[-2], 1)
    )

    for z in range(f2.shape[1]):
        # Upsample and scale spatial components.
        resampled = map_utils.resample_map(
            f2[:, z : z + 1, ...], box2x, box1x, 1 / scale, 1  #
        )
        f2_hires[:, z : z + 1, ...] = resampled / scale

    return flow_utils.reconcile_flows(
        (f1, f2_hires),
        max_gradient=max_gradient,
        max_deviation=max_deviation,
        min_patch_size=min_patch_size,
    )


def estimate_flow_fields(
    stitched_section_dir: str = "",
    start_section: int = 0,
    end_section: int = 9,
    ffe_conf: FlowFieldEstimationConfig = FlowFieldEstimationConfig(),
):
    section_dirs = list_zarr_sections(root_dir=stitched_section_dir)
    section_dirs = filter_sections(
        section_dirs=section_dirs, start_section=start_section, end_section=end_section
    )

    yx_size = get_yx_size(section_dirs, bin=1)

    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()

    previous_section = load_section_data(
        section_dir=section_dirs[0],
        yx_size=yx_size,
    )
    previous_name = splitext(basename(section_dirs[0]))[0]

    for i in range(1, len(section_dirs)):
        current_section = load_section_data(
            section_dir=section_dirs[i],
            yx_size=yx_size,
        )
        current_name = splitext(basename(section_dirs[i]))[0]

        flows1x = mfc.flow_field(
            previous_section,
            current_section,
            (ffe_conf.patch_size, ffe_conf.patch_size),
            (ffe_conf.stride, ffe_conf.stride),
            batch_size=ffe_conf.batch_size,
        )

        prev_data_bin2 = block_reduce(previous_section, func=np.mean).astype(np.float32)
        curr_data_bin2 = block_reduce(current_section, func=np.mean).astype(np.float32)

        flows2x = mfc.flow_field(
            prev_data_bin2,
            curr_data_bin2,
            (ffe_conf.patch_size, ffe_conf.patch_size),
            (ffe_conf.stride, ffe_conf.stride),
            batch_size=ffe_conf.batch_size,
        )

        flows1x = clean_flow(
            flows1x,
            patch_size=ffe_conf.patch_size,
            stride=ffe_conf.stride,
            min_peak_ratio=ffe_conf.min_peak_ratio,
            min_peak_sharpness=ffe_conf.min_peak_sharpness,
            max_magnitude=ffe_conf.max_magnitude,
            max_deviation=ffe_conf.max_deviation,
        )

        flows2x = clean_flow(
            flows2x,
            patch_size=ffe_conf.patch_size,
            stride=ffe_conf.stride,
            min_peak_ratio=ffe_conf.min_peak_ratio,
            min_peak_sharpness=ffe_conf.min_peak_sharpness,
            max_magnitude=ffe_conf.max_magnitude,
            max_deviation=ffe_conf.max_deviation,
        )

        final_flow = compute_final_flow(
            flows1x,
            flows2x,
            max_gradient=ffe_conf.max_gradient,
            max_deviation=ffe_conf.max_deviation,
            min_patch_size=ffe_conf.min_patch_size,
        )

        name = f"final_flow_{previous_name}_to_{current_name}.npy"
        np.save(join(section_dirs[i], name), final_flow)

        previous_section = current_section
        previous_name = current_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="fine_align_estimate_flow_fields.config"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    estimate_flow_fields(
        stitched_section_dir=config["stitched_sections_dir"],
        start_section=config["start_section"],
        end_section=config["end_section"],
        ffe_conf=FlowFieldEstimationConfig(**config["ffe_conf"]),
    )
