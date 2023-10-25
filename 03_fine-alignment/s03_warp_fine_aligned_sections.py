import json
import os
import re
from multiprocessing.pool import ThreadPool
from os.path import basename, join
from pathlib import Path

import numpy as np
import zarr
from connectomics.common import bounding_box
from connectomics.volume import subvolume
from numcodecs import Blosc
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
from scipy.interpolate import interpolate
from sofima import map_utils
from sofima.processor import maps
from sofima.warp import _cvx2_interpolation

from flows.warp_volume import compute_inv_map

try:
    from cvx2 import latest as cvx2
except ImportError:
    import cv2 as cvx2  # pytype:disable=import-error


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
    start_section: int,
    end_section: int,
    yx_size: tuple[int, int],
):
    target_dir = join(output_dir, volume_name)
    store = parse_url(target_dir, mode="w").store
    zarr_root = zarr.group(store=store)

    chunks = (1, 2744, 2744)
    write_image(
        image=np.zeros(
            (end_section - start_section, yx_size[0], yx_size[1]),
            dtype=np.uint8,
        ),
        group=zarr_root,
        axes="zyx",
        scaler=Scaler(max_layer=4),
        storage_options=dict(
            chunks=chunks,
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            overwrite=True,
            write_empty_chunks=False,
        ),
    )

    return target_dir


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
    section_dir: str, yx_start: tuple[int, int], yx_size: tuple[int, int]
):
    with open(join(section_dir, "padding.json")) as f:
        padding = np.array(json.load(f))

    source = zarr.Group(parse_url(section_dir).store)[0]

    data = np.zeros(yx_size, dtype=np.uint8)

    source_start_y = max(0, yx_start[0] - padding[0])
    source_end_y = min(yx_start[0] - padding[0] + yx_size[0], source.shape[0])
    source_start_x = max(0, yx_start[1] - padding[1])
    source_end_x = min(yx_start[1] - padding[1] + yx_size[1], source.shape[1])
    src_y = slice(source_start_y, source_end_y)
    src_x = slice(source_start_x, source_end_x)

    target_start_y = max(0, padding[0] - yx_start[0])
    target_end_y = data.shape[0] - target_start_y
    target_start_x = max(0, padding[1] - yx_start[1])
    target_end_x = data.shape[1] - target_start_x
    trg_y = slice(target_start_y, target_end_y)
    trg_x = slice(target_start_x, target_end_x)
    data[trg_y, trg_x] = source[src_y, src_x]
    return data


def write_section(
    section_dir: str,
    padding,
    start_section: int,
    yx_start: tuple[int, int],
    yx_size: tuple[int, int],
    zarr_root: zarr.Group,
):
    scaler = Scaler(max_layer=4)
    sec_id = int(basename(section_dir).split("_")[0][1:])
    output_index = sec_id - start_section
    data = load_section_data(section_dir=section_dir)

    for level in range(scaler.max_layer + 1):
        y_start = 0
        y_end = y_start + data.shape[0]
        x_start = 0
        x_end = x_start + data.shape[1]
        zarr_root[level][
            output_index,
            y_start:y_end,
            x_start:x_end,
        ] = data
        data = scaler.resize_image(data)


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

    dense_x = interpolate.RegularGridInterpolator(
        map_points, abs_map[0, 0, ...], bounds_error=False, fill_value=None
    )
    dense_y = interpolate.RegularGridInterpolator(
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
    yx_start: tuple[int, int],
    yx_size: tuple[int, int],
    start_section: int,
    end_section: int,
    blocks: list[tuple[int, int]],
    map_zarr: zarr.Group,
    flow_stride: int,
):
    pool = ThreadPool(processes=1)
    next_flow = pool.apply(
        reconcile_flow,
        kwds={
            "blocks": blocks,
            "main_map": map_zarr["main"],
            "main_inv_map": map_zarr["main_inv"],
            "cross_block_map": map_zarr["cross_block"],
            "cross_block_inv_map": map_zarr["cross_block_inv"],
            "last_inv_map": map_zarr["last_inv"],
            "stride": flow_stride,
            "start_section": start_section,
            "end_section": start_section + 1,
        },
    )

    target_volume = zarr.group(store=(parse_url(path=target_dir, mode="w").store))

    for section_dir in section_dirs:
        sec_id = int(basename(section_dir).split("_")[0][1:])
        if start_section <= sec_id <= end_section:
            output_index = sec_id - start_section

            inv_map, box = next_flow.get()

            if sec_id < end_section - 1:
                next_flow = pool.apply_async(
                    reconcile_flow,
                    kwds={
                        "blocks": blocks,
                        "main_map": map_zarr["main"],
                        "main_inv_map": map_zarr["main_inv"],
                        "cross_block_map": map_zarr["cross_block"],
                        "cross_block_inv_map": map_zarr["cross_block_inv"],
                        "last_inv_map": map_zarr["last_inv"],
                        "stride": flow_stride,
                        "start_section": sec_id + 1,
                        "end_section": sec_id + 2,
                    },
                )

            chunk_shape = target_volume.chunks
            tile_size_y = chunk_shape[1] * 1
            tile_size_x = chunk_shape[2] * 1
            y_start = yx_start[0]
            x_start = yx_start[1]
            overlap = tile_size_y

            for y in range(y_start, y_start + yx_size[0], tile_size_y):
                for x in range(x_start, x_start + yx_size[1], tile_size_x):
                    src_start_y = max(0, y - overlap)
                    src_start_x = max(0, x - overlap)
                    src_end_y = min(
                        min(y + tile_size_y + overlap, y + yx_size[0] + overlap),
                        yx_start[0] + target_volume.shape[1],
                    )
                    src_end_x = min(
                        min(x + tile_size_x + overlap, x + yx_size[1] + overlap),
                        yx_start[1] + target_volume.shape[2],
                    )

                    src_data = load_section_data(
                        section_dir=section_dir,
                        yx_start=(src_start_y, src_start_x),
                        yx_size=(src_end_y - src_start_y, src_end_x - src_start_x),
                    )[np.newaxis, ..., np.newaxis]

                    if src_data.sum() > 0:
                        img_box = bounding_box.BoundingBox(
                            start=(src_start_x, src_start_y, 0),
                            size=(src_end_x - src_start_x, src_end_y - src_start_y, 1),
                        )

                        end_y = min(
                            min(y + tile_size_y, y + yx_size[0]),
                            yx_start[0] + target_volume.shape[1],
                        )
                        end_x = min(
                            min(x + tile_size_x, x + yx_size[1]),
                            yx_start[1] + target_volume.shape[2],
                        )
                        out_box = bounding_box.BoundingBox(
                            start=(x, y, 0), size=(end_x - x, end_y - y, 1)
                        )

                        out_start_y = min(y, y - yx_start[0])
                        out_start_x = min(x, x - yx_start[1])
                        out_end_y = min(end_y, end_y - yx_start[0])
                        out_end_x = min(end_x, end_x - yx_start[1])

                        warp_subvolume(
                            image=src_data,
                            image_box=img_box,
                            coord_map=inv_map,
                            map_box=box,
                            stride=flow_stride,
                            out_box=out_box,
                            target_volume=target_volume,
                            z=output_index,
                            out_start_y=out_start_y,
                            out_end_y=out_end_y,
                            out_start_x=out_start_x,
                            out_end_x=out_end_x,
                        )

    pool.close()
    pool.join()


def warp_fine_aligned_sections(
    stitched_section_dir: str,
    output_dir: str,
    volume_name: str,
    start_section: int,
    end_section: int,
    yx_start: tuple[int, int],
    yx_size: tuple[int, int],
    blocks: list[tuple[int, int]],
    map_zarr_dir: str,
    flow_stride: int,
):
    cvx2.setNumThreads(1)
    zarr.blosc.use_threads = False

    section_dirs = list_zarr_sections(root_dir=stitched_section_dir)

    target_dir = create_zarr(
        output_dir=output_dir,
        volume_name=volume_name,
        start_section=start_section,
        end_section=end_section,
        yx_size=yx_size,
    )

    store = parse_url(path=map_zarr_dir, mode="w").store
    map_zarr: zarr.Group = zarr.group(store=store)

    warp_sections(
        section_dirs=section_dirs,
        target_dir=target_dir,
        yx_start=yx_start,
        yx_size=yx_size,
        start_section=start_section,
        end_section=end_section,
        blocks=blocks,
        map_zarr=map_zarr,
        flow_stride=flow_stride,
    )
