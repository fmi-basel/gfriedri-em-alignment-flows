import yaml
import zarr
from connectomics.common import bounding_box
from ome_zarr.io import parse_url
from sofima import map_utils


def main(
    map_zarr_dir: str,
    block_index: int,
    stride: int,
):
    store = parse_url(path=map_zarr_dir, mode="w").store
    map_zarr: zarr.Group = zarr.group(store=store)
    cross_block_map = map_zarr["cross_block"]
    cross_block_inv_map = map_zarr["cross_block_inv"]
    xblk = map_zarr["relaxed_cross_block_flow"][:, block_index : block_index + 1]
    main_map_size = map_zarr["main"].shape[1:][::-1]
    map_box = bounding_box.BoundingBox(start=(0, 0, 0), size=main_map_size[:-1] + (1,))
    map2x_box = map_box.scale(0.5)
    cross_block_map[:, block_index : block_index + 1, ...] = map_utils.resample_map(
        xblk, map2x_box, map_box, stride * 2, stride
    )
    cross_block_inv_map[:, block_index : block_index + 1, ...] = map_utils.invert_map(
        cross_block_map[:, block_index : block_index + 1], map_box, map_box, stride
    )

    with open(f"map_{block_index}.yaml", "w") as f:
        yaml.safe_dump(
            dict(
                map_path=map_zarr_dir,
            ),
            f,
            sort_keys=False,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, default="resample_and_invert_config.yaml"
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(
        map_zarr_dir=config["map_zarr_dir"],
        block_index=config["block_index"],
        stride=config["stride"],
    )
