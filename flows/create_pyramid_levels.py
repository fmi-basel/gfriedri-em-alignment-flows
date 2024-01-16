from pathlib import Path
from typing import Union

import dask.array as da
import numpy as np
import zarr
from distributed import Client, wait
from numcodecs import Blosc
from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url
from ome_zarr.writer import _get_valid_axes, write_multiscales_metadata


def mean_cast_to(target_dtype):
    """
    Wrap np.mean to cast the result to a given dtype.
    """

    def _mean(
        a,
        axis=None,
        dtype=None,
        out=None,
        keepdims=np._NoValue,
        *,
        where=np._NoValue,
    ):
        return np.mean(
            a=a, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where
        ).astype(target_dtype)

    return _mean


def get_coordinate_transformations(
    max_layer: int, yx_binning: int = 2, spacing: tuple[int, int, int] = [1, 1, 1]
):
    transformations = []
    for s in range(max_layer + 1):
        transformations.append(
            [
                {
                    "scale": [
                        spacing[0],
                        spacing[1] * yx_binning * 2**s,
                        spacing[2] * yx_binning * 2**s,
                    ],
                    "type": "scale",
                }
            ]
        )

    return transformations


def main(
    zarr_path: Union[str, Path],
    max_layer: int,
    chunks: tuple[int, int, int] = (1, 2744, 2744),
):
    client = Client(
        n_workers=12,  # Make sure this number equals the number of CPU cores
        # on your machine.
        threads_per_worker=1,
        processes=False,
        memory_limit="4GB",  # Total memory usage is n_workers *
        # memory_limit. Make sure it fits, otherwise it spills to disk.
    )

    store = parse_url(zarr_path, mode="w").store
    group = zarr.group(store=store)
    image = da.from_zarr(url=group.store, component=str(Path(group.path, "0")))
    spacing = group.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][
        0
    ]["scale"]

    datasets = [{"path": "0"}]
    shapes = [image.shape]
    for path in range(1, max_layer + 1):
        image = da.coarsen(
            reduction=mean_cast_to(image.dtype),
            x=image,
            axes={
                image.ndim - 2: 2,
                image.ndim - 1: 2,
            },
            trim_excess=True,
        )
        options = dict(
            dimension_separator="/",
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            chunks=chunks,
            write_empty_chunks=False,
        )
        image = image.rechunk(options["chunks"])
        wait(
            client.persist(
                da.to_zarr(
                    arr=image,
                    url=group.store,
                    compute=False,
                    component=str(Path(group.path, str(path))),
                    storage_options=options,
                )
            )
        )
        datasets.append({"path": str(path)})
        shapes.append(image.shape)
        image = da.from_zarr(
            url=group.store, component=str(Path(group.path, str(path)))
        )

    coordinate_transformations = get_coordinate_transformations(
        max_layer=max_layer, yx_binning=2, spacing=spacing
    )
    fmt = CurrentFormat()
    dims = len(shapes[0])
    fmt.validate_coordinate_transformations(
        dims, len(datasets), coordinate_transformations
    )
    for dataset, transform in zip(datasets, coordinate_transformations):
        dataset["coordinateTransformations"] = transform
    axes = _get_valid_axes(dims, ["z", "y", "x"], fmt)
    write_multiscales_metadata(
        group,
        datasets,
        fmt,
        axes,
    )


if __name__ == "__main__":
    main(zarr_path="/path/to/ngff.zarr", max_layer=5, chunks=(1, 2744, 2744))
