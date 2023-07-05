import zarr
from dask.diagnostics import ProgressBar
from ome_zarr.io import parse_url
from rechunker import rechunk


def main():
    source: zarr.Array = zarr.group(
        parse_url(
            "/tungstenfs/scratch/gmicro_sem/gmicro/buchtimo"
            "/20220524_Bo_juv20210731_test_volume/aligned_volume.zarr"
        ).store
    )[0]

    intermediate = parse_url("/tungstenfs/temp/generic/intermediate.zarr", "w").store

    target = parse_url("/tungstenfs/temp/generic/aligned_volume.zarr", "w").store

    print("Compute plan.")
    rechunked = rechunk(
        source,
        target_chunks=(196, 196, 196),
        target_store=target,
        max_mem="512MB",
        temp_store=intermediate,
        target_options={
            "compressor": source.compressor,
            "order": source.order,
        },
    )

    print("Execute plan.")
    with ProgressBar():
        rechunked.execute()


if __name__ == "__main__":
    main()
