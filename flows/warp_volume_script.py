import os
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from os.path import exists

from scipy import interpolate

try:
    from cvx2 import latest as cvx2
except ImportError:
    import cv2 as cvx2  # pytype:disable=import-error

import numpy as np
import zarr
from connectomics.common import bounding_box
from connectomics.volume import subvolume
from ome_zarr.io import parse_url
from sofima import map_utils
from sofima.processor import maps
from sofima.warp import _cvx2_interpolation
from tqdm import tqdm


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


def create_output_volume(source_zarr, target_volume, z_size, yx_size):
    store = parse_url(path=target_volume, mode="w").store
    target_zarr: zarr.Group = zarr.group(store=store)

    target_zarr.create_dataset(
        name="0",
        shape=(z_size, yx_size[0], yx_size[1]),
        chunks=source_zarr["0"].chunks,
        dtype=source_zarr["0"].dtype,
        compressor=source_zarr["0"].compressor,
        fill_value=source_zarr["0"].fill_value,
        order=source_zarr["0"].order,
        overwrite=True,
        write_empty_chunks=False,
    )
    target_zarr.attrs.update(source_zarr.attrs)
    return target_zarr


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


def warp_section(
    source_volume: zarr.Group,
    target_volume: zarr.Group,
    start_section: int,
    end_section: int,
    z_offset: int,
    yx_start: list[int],
    yx_size: list[int],
    blocks: list[tuple[int, int]],
    main_map_zarr: zarr.Group,
    main_inv_map_zarr: zarr.Group,
    cross_block_map_zarr: zarr.Group,
    cross_block_inv_map_zarr: zarr.Group,
    last_inv_map_zarr: zarr.Group,
    stride: int,
):
    pool = ThreadPool(processes=1)
    next_flow = pool.apply_async(
        reconcile_flow,
        kwds={
            "blocks": blocks,
            "main_map": main_map_zarr,
            "main_inv_map": main_inv_map_zarr,
            "cross_block_map": cross_block_map_zarr,
            "cross_block_inv_map": cross_block_inv_map_zarr,
            "last_inv_map": last_inv_map_zarr,
            "stride": stride,
            "start_section": start_section,
            "end_section": start_section + 1,
        },
    )
    for z in tqdm(range(start_section, end_section)):
        inv_map, box = next_flow.get()

        if z < end_section - 1:
            next_flow = pool.apply_async(
                reconcile_flow,
                kwds={
                    "blocks": blocks,
                    "main_map": main_map_zarr,
                    "main_inv_map": main_inv_map_zarr,
                    "cross_block_map": cross_block_map_zarr,
                    "cross_block_inv_map": cross_block_inv_map_zarr,
                    "last_inv_map": last_inv_map_zarr,
                    "stride": stride,
                    "start_section": z + 1,
                    "end_section": z + 2,
                },
            )

        chunk_shape = target_volume.chunks
        tile_size_y = chunk_shape[1] * 1
        tile_size_x = chunk_shape[2] * 1
        y_start = yx_start[0]
        x_start = yx_start[1]
        overlap = 2744
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

                src_data = source_volume[
                    z : z + 1, src_start_y:src_end_y, src_start_x:src_end_x
                ][np.newaxis]

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
                        stride=stride,
                        out_box=out_box,
                        target_volume=target_volume,
                        z=z - z_offset,
                        out_start_y=out_start_y,
                        out_end_y=out_end_y,
                        out_start_x=out_start_x,
                        out_end_x=out_end_x,
                    )

    pool.close()
    pool.join()


def main(
    source_volume: str,
    target_volume: str,
    start_section: int,
    end_section: int,
    yx_start: list[int],
    yx_size: list[int],
    blocks: list[tuple[int, int]],
    main_map_zarr: zarr.Group,
    main_inv_map_zarr: zarr.Group,
    cross_block_map_zarr: zarr.Group,
    cross_block_inv_map_zarr: zarr.Group,
    last_inv_map_zarr: zarr.Group,
    stride: float,
):
    print("start processing")
    print(os.environ["OMP_NUM_THREADS"])
    print(os.environ["MKL_NUM_THREADS"])
    print(os.environ["OPENBLAS_NUM_THREADS"])
    print(os.environ["BLIS_NUM_THREADS"])
    print(os.environ["VECLIB_MAXIMUM_THREADS"])
    print(os.environ["NUMEXPR_NUM_THREADS"])
    cvx2.setNumThreads(1)
    print(f"CV2 threads: {cvx2.getNumThreads()}")
    print(target_volume)
    zarr.blosc.use_threads = False
    source_zarr = zarr.group(store=(parse_url(path=source_volume, mode="r").store))

    if not exists(target_volume):
        target_zarr = create_output_volume(
            source_zarr=source_zarr,
            target_volume=target_volume,
            z_size=15850,
            yx_size=yx_size,
        )
    else:
        target_zarr = zarr.group(store=(parse_url(path=target_volume, mode="w").store))

    # warp_section(
    # **{
    # "source_volume": source_zarr['0'],
    # "target_volume": target_zarr['0'],
    # "start_section": start_section,
    # "end_section": end_section,
    # "z_offset": 0,
    # "yx_start": yx_start,
    # "yx_size": yx_size,
    # "blocks": blocks,
    # "main_map_zarr": main_map_zarr,
    # "main_inv_map_zarr": main_inv_map_zarr,
    # "cross_block_map_zarr": cross_block_map_zarr,
    # "cross_block_inv_map_zarr": cross_block_inv_map_zarr,
    # "last_inv_map_zarr": last_inv_map_zarr,
    # "stride": stride,
    # }
    # )

    futures = []
    parallelization = 20
    pool = Pool(parallelization)

    n_sections = end_section - start_section
    split = n_sections // parallelization + 1
    start = start_section

    for i in range(parallelization - 1):
        futures.append(
            pool.apply_async(
                warp_section,
                kwds={
                    "source_volume": source_zarr["0"],
                    "target_volume": target_zarr["0"],
                    "start_section": start,
                    "end_section": start + split,
                    "z_offset": 0,
                    "yx_start": yx_start,
                    "yx_size": yx_size,
                    "blocks": blocks,
                    "main_map_zarr": main_map_zarr,
                    "main_inv_map_zarr": main_inv_map_zarr,
                    "cross_block_map_zarr": cross_block_map_zarr,
                    "cross_block_inv_map_zarr": cross_block_inv_map_zarr,
                    "last_inv_map_zarr": last_inv_map_zarr,
                    "stride": stride,
                },
            )
        )
        start = start + split

    futures.append(
        pool.apply_async(
            warp_section,
            kwds={
                "source_volume": source_zarr["0"],
                "target_volume": target_zarr["0"],
                "start_section": start,
                "end_section": end_section,
                "z_offset": 0,
                "yx_start": yx_start,
                "yx_size": yx_size,
                "blocks": blocks,
                "main_map_zarr": main_map_zarr,
                "main_inv_map_zarr": main_inv_map_zarr,
                "cross_block_map_zarr": cross_block_map_zarr,
                "cross_block_inv_map_zarr": cross_block_inv_map_zarr,
                "last_inv_map_zarr": last_inv_map_zarr,
                "stride": stride,
            },
        )
    )

    pool.close()
    pool.join()


if __name__ == "__main__":
    blocks = []
    for i in range(0, 15849, 100):
        blocks.append([i, min(15849, i + 100)])

    start_section = 0
    end_section = 100
    yx_start = [0, 0]
    yx_size = [48777, 59382]
    stride = 40
    source_volume = "/path/to/source_volume.zarr"
    target_volume = "/path/to/target_volume.zarr/"

    maps_path = "/path/to/maps.zarr"
    main_map_zarr = zarr.group(store=(parse_url(path=maps_path, mode="r").store))[
        "main"
    ]

    last_inv_map_zarr = zarr.group(store=(parse_url(path=maps_path, mode="r").store))[
        "last_inv"
    ]

    main_inv_map_zarr = zarr.group(store=(parse_url(path=maps_path, mode="r").store))[
        "main_inv"
    ]

    cross_block_map_zarr = zarr.group(
        store=(parse_url(path=maps_path, mode="r").store)
    )["cross_block"]

    cross_block_inv_map_zarr = zarr.group(
        store=(parse_url(path=maps_path, mode="r").store)
    )["cross_block_inv"]

    main(
        source_volume=source_volume,
        target_volume=target_volume,
        start_section=start_section,
        end_section=end_section,
        yx_start=yx_start,
        yx_size=yx_size,
        blocks=blocks,
        main_map_zarr=main_map_zarr,
        main_inv_map_zarr=main_inv_map_zarr,
        cross_block_map_zarr=cross_block_map_zarr,
        cross_block_inv_map_zarr=cross_block_inv_map_zarr,
        last_inv_map_zarr=last_inv_map_zarr,
        stride=stride,
    )
