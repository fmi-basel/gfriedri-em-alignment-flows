# buchtimo
```shell
prefect deployment build 01_tile-stitching/prefect_tile_stitching.py:register_tiles_flow --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 01_tile-stitching/deployments/buchtimo-register_tiles_flow.yaml -ib process/gfriedri-em-alignment-buchtimo-gpu


prefect deployment build 01_tile-stitching/prefect_tile_stitching.py:warp_tiles_flow --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 01_tile-stitching/deployments/buchtimo-warp_tiles_flow.yaml -ib process/gfriedri-em-alignment-buchtimo-cpu


prefect deployment build 01_tile-stitching/prefect_tile_stitching.py:tile_stitching --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 01_tile-stitching/deployments/buchtimo-tile_stitching.yaml -ib process/gfriedri-em-alignment-buchtimo-orchestration
```

# ganctoma
```shell
prefect deployment build 01_tile-stitching/prefect_tile_stitching.py:register_tiles_flow --name ganctoma -p SLURM -q ganctoma -sb github/gfriedri-em-alignment-flows-ganctoma --skip-upload -o 01_tile-stitching/deployments/ganctoma-register_tiles_flow.yaml -ib process/gfriedri-em-alignment-ganctoma-gpu


prefect deployment build 01_tile-stitching/prefect_tile_stitching.py:warp_tiles_flow --name ganctoma -p SLURM -q ganctoma -sb github/gfriedri-em-alignment-flows-ganctoma --skip-upload -o 01_tile-stitching/deployments/ganctoma-warp_tiles_flow.yaml -ib process/gfriedri-em-alignment-ganctoma-cpu


prefect deployment build 01_tile-stitching/prefect_tile_stitching.py:tile_stitching --name ganctoma -p SLURM -q ganctoma -sb github/gfriedri-em-alignment-flows-ganctoma --skip-upload -o 01_tile-stitching/deployments/ganctoma-tile_stitching.yaml -ib process/gfriedri-em-alignment-ganctoma-orchestration
```

# kappjoha
```shell
prefect deployment build 01_tile-stitching/prefect_tile_stitching.py:register_tiles_flow --name kappjoha -p SLURM -q kappjoha -sb github/gfriedri-em-alignment-flows-kappjoha --skip-upload -o 01_tile-stitching/deployments/kappjoha-register_tiles_flow.yaml -ib process/gfriedri-em-alignment-kappjoha-gpu


prefect deployment build 01_tile-stitching/prefect_tile_stitching.py:warp_tiles_flow --name kappjoha -p SLURM -q kappjoha -sb github/gfriedri-em-alignment-flows-kappjoha --skip-upload -o 01_tile-stitching/deployments/kappjoha-warp_tiles_flow.yaml -ib process/gfriedri-em-alignment-kappjoha-cpu


prefect deployment build 01_tile-stitching/prefect_tile_stitching.py:tile_stitching --name kappjoha -p SLURM -q kappjoha -sb github/gfriedri-em-alignment-flows-kappjoha --skip-upload -o 01_tile-stitching/deployments/kappjoha-tile_stitching.yaml -ib process/gfriedri-em-alignment-kappjoha-orchestration
```
