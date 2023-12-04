import argparse
import json
import logging
import os
import re
from os.path import basename, join
from pathlib import Path

import numpy as np
import yaml
import zarr
from connectomics.common import bounding_box
from connectomics.volume import subvolume
from numcodecs import Blosc
from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url
from s01_estimate_flow_fields import filter_sections, get_yx_size
from scipy.interpolate import RegularGridInterpolator
from sofima import map_utils
from sofima.processor import maps

try:
    from cvx2 import latest as cvx2
except ImportError:
    import cv2 as cvx2  # pytype:disable=import-error


def _cvx2_interpolation(inter_scheme: str):
    inter_map = {
        "nearest": cvx2.INTER_NEAREST,
        "linear": cvx2.INTER_LINEAR,
        "cubic": cvx2.INTER_CUBIC,
        "lanczos": cvx2.INTER_LANCZOS4,
    }
    return inter_map[inter_scheme]


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


def create_zarr(
    output_dir: str,
    volume_name: str,
    n_sections: int,
    yx_size: tuple[int, int],
    bin: int,
    n_levels: int = 1,
):
    target_dir = join(output_dir, volume_name)
    store = parse_url(target_dir, mode="w").store
    zarr_root = zarr.group(store=store)

    datasets = []
    shapes = []
    for path, level in enumerate(range(n_levels)):
        downscale = 2**level
        # Downscale only in YX
        shape = (
            n_sections,
            yx_size[0] // bin // downscale,
            yx_size[1] // bin // downscale,
        )
        zarr_root.create_dataset(
            name=str(path),
            shape=shape,
            chunks=(1, 2744, 2744),
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            overwrite=True,
            write_empty_chunks=False,
            fill_value=0,
            dtype=np.uint8,
            dimension_separator="/",
        )

        datasets.append({"path": str(path)})
        shapes.append(shape)

    fmt = CurrentFormat()
    coordinate_transformations = fmt.generate_coordinate_transformations(shapes)

    fmt.validate_coordinate_transformations(
        ndim=3,
        nlevels=len(shapes),
        coordinate_transformations=coordinate_transformations,
    )
    for dataset, transform in zip(datasets, coordinate_transformations):
        dataset["coordinateTransformations"] = transform

    from ome_zarr import writer

    writer.write_multiscales_metadata(
        group=zarr_root,
        datasets=datasets,
        axes="zyx",
    )

    return target_dir


def compute_inv_map(map_data, stride):
    box = bounding_box.BoundingBox(
        start=(0, 0, 0), size=(map_data.shape[-1], map_data.shape[-2], 1)
    )
    inv_map = map_utils.invert_map(map_data, box, box, stride)
    return inv_map, box


def reconcile_flow(
    blocks: list[tuple[int, int]],
    main_map: zarr.Group,
    main_inv_map: zarr.Group,
    cross_block_map: zarr.Group,
    cross_block_inv_map: zarr.Group,
    last_inv_map: zarr.Group,
    stride: int,
    start_section: int,
    end_section: int,
    logger: logging.Logger,
):
    class ReconcileCrossBlockMaps(maps.ReconcileCrossBlockMaps):
        def _open_volume(self, path: str):
            if path == "main_inv":
                return main_inv_map
            elif path == "last_inv":
                return last_inv_map
            elif path == "xblk":
                return cross_block_map
            elif path == "xblk_inv":
                return cross_block_inv_map
            else:
                raise ValueError(f"Unknown volume {path}")

    reconcile = ReconcileCrossBlockMaps(
        "xblk",
        "xblk_inv",
        "last_inv",
        "main_inv",
        {str(e): i for i, (s, e) in enumerate(blocks)},
        stride,
        0,
    )
    reconcile.set_effective_subvol_and_overlap(main_map.shape[1:][::-1], (0, 0, 0))
    size = (*main_map.shape[2:][::-1], end_section - start_section)
    main_box = bounding_box.BoundingBox(start=(0, 0, start_section), size=size)
    reconciled_flow = reconcile.process(
        subvolume.Subvolume(main_map[:, start_section:end_section], main_box)
    ).data

    return compute_inv_map(
        map_data=reconciled_flow,
        stride=stride,
    )


def load_section_data(
    section_dir: str,
    yx_start: tuple[int, int],
    yx_end: tuple[int, int],
    logger: logging.Logger,
):
    with open(join(section_dir, "coarse_stack_padding.json")) as f:
        config = json.load(f)
        pad_y = config["shift_y"]
        pad_x = config["shift_x"]

    zarray = zarr.Group(parse_url(section_dir).store)[0]
    sec_start_y = max(0, yx_start[0] - pad_y)
    sec_start_x = max(0, yx_start[1] - pad_x)
    sec_end_y = min(max(0, yx_end[0] - pad_y), zarray.shape[0])
    sec_end_x = min(max(0, yx_end[1] - pad_x), zarray.shape[1])

    data = np.zeros(
        (yx_end[0] - yx_start[0], yx_end[1] - yx_start[1]), dtype=zarray.dtype
    )

    if (
        sec_start_y > zarray.shape[0]  # noqa: W503
        or sec_start_x > zarray.shape[1]  # noqa: W503
        or sec_end_y == 0  # noqa: W503
        or sec_end_x == 0  # noqa: W503
    ):
        # no data in this section
        return data
    else:
        chunk_shape = data.shape
        chunk_start_y = max(0, pad_y - yx_start[0])
        chunk_end_y = min(chunk_shape[0], chunk_start_y + sec_end_y - sec_start_y)
        chunk_start_x = max(0, pad_x - yx_start[1])
        chunk_end_x = min(chunk_shape[1], chunk_start_x + sec_end_x - sec_start_x)
        data[chunk_start_y:chunk_end_y, chunk_start_x:chunk_end_x] = zarray[
            sec_start_y:sec_end_y, sec_start_x:sec_end_x
        ]
        return data


def warp_subvolume(
    image: np.ndarray,
    image_box: bounding_box.BoundingBoxBase,
    coord_map: np.ndarray,
    map_box: bounding_box.BoundingBoxBase,
    stride: float,
    out_box: bounding_box.BoundingBoxBase,
    offset: float = 0.0,
    target_volume: zarr.Group = None,
    z: int = 0,
    out_start_y: int = 0,
    out_end_y: int = 10,
    out_start_x: int = 0,
    out_end_x: int = 10,
    logger: logging.Logger = logging.getLogger("warp"),
) -> np.ndarray:
    """Warps a subvolume of data according to a coordinate map.

    Args:
      image: [n, z, y, x] data to warp; valid data types are those supported by
        OpenCV's `remap` as well as uint64, which is treated as segmentation data
      image_box: bounding box identifying the part of the volume from which the
        image data was extracted
      coord_map: [2, z, y, x] xy 'inverse' coordinate map in relative format (each
        entry in the map specifies the source coordinate in 'image' from which to
        read data)
      map_box: bounding box identifying the part of the volume from which the
        coordinate map was extracted
      stride: length in pixels of the image corresponding to a single unit (pixel)
        of the coordinate map
      out_box: bounding box for the warped data
      interpolation: interpolation scheme to use; defaults to nearest neighbor for
        uint64 data, and Lanczos for other types
      offset: (deprecated, do not use) non-zero values necessary to reproduce some
        old renders
      parallelism: number of threads to use for warping sections

    Returns:
      warped image covering 'out_box'
    """
    interpolation = _cvx2_interpolation("lanczos")

    orig_dtype = image.dtype

    # Convert values within the coordinate map so that they are
    # within the local coordinate system of 'image'.
    abs_map = map_utils.to_absolute(coord_map, stride)
    abs_map += (map_box.start[:2] * stride - image_box.start[:2] + offset).reshape(
        2, 1, 1, 1
    )

    # Coordinates of the map nodes within the local coordinate
    # system of 'out_box'.
    map_y, map_x = np.ogrid[: coord_map.shape[2], : coord_map.shape[3]]
    map_y = (map_y + map_box.start[1]) * stride - out_box.start[1] + offset
    map_x = (map_x + map_box.start[0]) * stride - out_box.start[0] + offset
    map_points = (map_y.ravel(), map_x.ravel())

    out_y, out_x = np.mgrid[: out_box.size[1], : out_box.size[0]]

    try:
        maptype = cvx2.CVX_16SC2
    except AttributeError:
        maptype = cvx2.CV_16SC2

    dense_x = RegularGridInterpolator(
        map_points, abs_map[0, 0, ...], bounds_error=False, fill_value=None
    )
    dense_y = RegularGridInterpolator(
        map_points, abs_map[1, 0, ...], bounds_error=False, fill_value=None
    )

    # dxy: [0 .. out_box.size] -> [coord within image]
    dx = dense_x((out_y, out_x)).astype(np.float32)
    dy = dense_y((out_y, out_x)).astype(np.float32)

    dx, dy = cvx2.convertMaps(
        dx,
        dy,
        dstmap1type=maptype,
        nninterpolation=(interpolation == cvx2.INTER_NEAREST),
    )

    warped = cvx2.remap(image[0, 0, ...], dx, dy, interpolation=interpolation).astype(
        orig_dtype
    )

    if warped.sum() > 0:
        target_volume[z, out_start_y:out_end_y, out_start_x:out_end_x] = warped


def warp_sections(
    section_dirs: list[str],
    target_dir: str,
    yx_size: tuple[int, int],
    offset: int,
    blocks: list[tuple[int, int]],
    map_zarr_dir: str,
    flow_stride: int,
    logger: logging.Logger,
):
    store = parse_url(path=map_zarr_dir, mode="w").store
    map_zarr: zarr.Group = zarr.group(store=store)

    target_volume = zarr.group(store=parse_url(path=target_dir, mode="w").store)[0]
    logger.info(f"Processing {len(section_dirs)} sections.")
    for i, section_dir in enumerate(section_dirs):
        sec_id = int(basename(section_dir).split("_")[0][1:])
        logger.info(f"Warp section {sec_id}.")
        inv_map, box = reconcile_flow(
            blocks=blocks,
            main_map=map_zarr["main"],
            main_inv_map=map_zarr["main_inv"],
            cross_block_map=map_zarr["cross_block"],
            cross_block_inv_map=map_zarr["cross_block_inv"],
            last_inv_map=map_zarr["last_inv"],
            stride=flow_stride,
            start_section=i + offset,
            end_section=i + offset + 1,
            logger=logger,
        )

        chunk_shape = target_volume.chunks
        tile_size_y = chunk_shape[1] * 2
        tile_size_x = chunk_shape[2] * 2
        overlap = chunk_shape[1]
        for y in range(0, yx_size[0], tile_size_y):
            for x in range(0, yx_size[1], tile_size_x):
                src_start_y = max(0, y - overlap)
                src_start_x = max(0, x - overlap)
                src_end_y = min(
                    min(y + tile_size_y + overlap, y + yx_size[0] + overlap),
                    target_volume.shape[1],
                )
                src_end_x = min(
                    min(x + tile_size_x + overlap, x + yx_size[1] + overlap),
                    target_volume.shape[2],
                )
                src_data = load_section_data(
                    section_dir=section_dir,
                    yx_start=(src_start_y, src_start_x),
                    yx_end=(src_end_y, src_end_x),
                    logger=logger,
                )[np.newaxis, np.newaxis]

                sum = src_data.sum()
                if sum > 0:
                    img_box = bounding_box.BoundingBox(
                        start=(src_start_x, src_start_y, 0),
                        size=(src_end_x - src_start_x, src_end_y - src_start_y, 1),
                    )

                    out_end_y = min(
                        y + tile_size_y,
                        target_volume.shape[1],
                    )
                    out_end_x = min(
                        x + tile_size_x,
                        target_volume.shape[2],
                    )
                    out_box = bounding_box.BoundingBox(
                        start=(x, y, 0), size=(out_end_x - x, out_end_y - y, 1)
                    )

                    warp_subvolume(
                        image=src_data,
                        image_box=img_box,
                        coord_map=inv_map,
                        map_box=box,
                        stride=flow_stride,
                        out_box=out_box,
                        target_volume=target_volume,
                        z=offset + i,
                        out_start_y=y,
                        out_end_y=out_end_y,
                        out_start_x=x,
                        out_end_x=out_end_x,
                        logger=logger,
                    )


def warp_fine_aligned_sections(
    stitched_sections_dir: str,
    output_dir: str,
    volume_name: str,
    start_section: int,
    end_section: int,
    block_size: int,
    map_zarr_dir: str,
    flow_stride: int,
):
    cvx2.setNumThreads(1)
    zarr.blosc.use_threads = False

    section_dirs = list_zarr_sections(root_dir=stitched_sections_dir)
    section_dirs = filter_sections(
        section_dirs=section_dirs, start_section=start_section, end_section=end_section
    )

    blocks = []
    for i in range(0, len(section_dirs), block_size):
        blocks.append((i, min(len(section_dirs), i + block_size)))

    yx_size = get_yx_size(section_dirs, bin=1)

    target_dir = create_zarr(
        output_dir=output_dir,
        volume_name=volume_name,
        n_sections=len(section_dirs),
        yx_size=yx_size,
        bin=1,
    )

    warp_sections(
        section_dirs=section_dirs,
        target_dir=target_dir,
        yx_size=yx_size,
        offset=start_section,
        blocks=blocks,
        map_zarr_dir=map_zarr_dir,
        flow_stride=flow_stride,
        logger=logging.getLogger("Warp Sections"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="warp_fine_aligned_sections.config"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    warp_fine_aligned_sections(
        stitched_sections_dir=config["stitched_sections_dir"],
        output_dir=config["output_dir"],
        volume_name=config["volume_name"],
        start_section=config["start_section"],
        end_section=config["end_section"],
        block_size=config["block_size"],
        map_zarr_dir=config["map_zarr_dir"],
        flow_stride=config["flow_stride"],
    )
