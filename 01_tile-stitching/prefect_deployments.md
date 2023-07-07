# buchtimo
```shell
prefect deployment build 01_tile-stitching/prefect_tile_stitching.py:register_tiles_flow --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 01_tile-stitching/deployments/register_tiles_flow.yaml -ib process/gfriedri-em-alignment-buchtimo-gpu


prefect deployment build 01_tile-stitching/prefect_tile_stitching.py:warp_tiles_flow --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 01_tile-stitching/deployments/warp_tiles_flow.yaml -ib process/gfriedri-em-alignment-buchtimo-cpu


prefect deployment build 01_tile-stitching/prefect_tile_stitching.py:tile_stitching --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 01_tile-stitching/deployments/tile_stitching.yaml -ib process/gfriedri-em-alignment-buchtimo-orchestration
```
