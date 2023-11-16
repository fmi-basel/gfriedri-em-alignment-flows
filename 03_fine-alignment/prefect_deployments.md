# buchtimo
```shell
prefect deployment build 03_fine-alignment/prefect_fine_alignment.py:estimate_z_flow_fields --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 03_fine-alignment/deployments/buchtimo-estimate_z_flow_fields.yaml -ib process/gfriedri-em-alignment-buchtimo-gpu


prefect deployment build 03_fine-alignment/prefect_fine_alignment.py:estimate_z_flow_fields_parallel --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 03_fine-alignment/deployments/buchtimo-estimate_z_flow_fields-parallel.yaml -ib process/gfriedri-em-alignment-buchtimo-orchestration


prefect deployment build 03_fine-alignment/prefect_relax_meshes.py:relax_meshes_in_blocks_flow --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 03_fine-alignment/deployments/buchtimo-relax_meshes_in_blocks.yaml -ib process/gfriedri-em-alignment-buchtimo-gpu


prefect deployment build 03_fine-alignment/prefect_relax_meshes.py:relax_mehses_cross_blocks_flow --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 03_fine-alignment/deployments/buchtimo-relax_meshes_cross_blocks.yaml -ib process/gfriedri-em-alignment-buchtimo-gpu


prefect deployment build 03_fine-alignment/prefect_relax_meshes.py:relax_meshes_flow --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 03_fine-alignment/deployments/buchtimo-relax_meshes.yaml -ib process/gfriedri-em-alignment-buchtimo-orchestration


prefect deployment build 03_fine-alignment/prefect_warp_fine_aligned.py:warp_sections_flow --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 03_fine-alignment/deployments/buchtimo-warp_sections_flow.yaml -ib process/gfriedri-em-alignment-buchtimo-cpu


prefect deployment build 03_fine-alignment/prefect_warp_fine_aligned.py:warp_fine_alignment --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 03_fine-alignment/deployments/buchtimo-warp_fine_alignment.yaml -ib process/gfriedri-em-alignment-buchtimo-orchestration
```

# ganctoma
```shell
prefect deployment build 03_fine-alignment/prefect_fine_alignment.py:estimate_z_flow_fields --name ganctoma -p SLURM -q ganctoma -sb github/gfriedri-em-alignment-flows-ganctoma --skip-upload -o 03_fine-alignment/deployments/ganctoma-estimate_z_flow_fields.yaml -ib process/gfriedri-em-alignment-ganctoma-gpu


prefect deployment build 03_fine-alignment/prefect_fine_alignment.py:estimate_z_flow_fields_parallel --name ganctoma -p SLURM -q ganctoma -sb github/gfriedri-em-alignment-flows-ganctoma --skip-upload -o 03_fine-alignment/deployments/ganctoma-estimate_z_flow_fields-parallel.yaml -ib process/gfriedri-em-alignment-ganctoma-orchestration


prefect deployment build 03_fine-alignment/prefect_relax_meshes.py:relax_meshes_in_blocks_flow --name ganctoma -p SLURM -q ganctoma -sb github/gfriedri-em-alignment-flows-ganctoma --skip-upload -o 03_fine-alignment/deployments/ganctoma-relax_meshes_in_blocks.yaml -ib process/gfriedri-em-alignment-ganctoma-gpu


prefect deployment build 03_fine-alignment/prefect_relax_meshes.py:relax_mehses_cross_blocks_flow --name ganctoma -p SLURM -q ganctoma -sb github/gfriedri-em-alignment-flows-ganctoma --skip-upload -o 03_fine-alignment/deployments/ganctoma-relax_meshes_cross_blocks.yaml -ib process/gfriedri-em-alignment-ganctoma-gpu


prefect deployment build 03_fine-alignment/prefect_relax_meshes.py:relax_meshes_flow --name ganctoma -p SLURM -q ganctoma -sb github/gfriedri-em-alignment-flows-ganctoma --skip-upload -o 03_fine-alignment/deployments/ganctoma-relax_meshes.yaml -ib process/gfriedri-em-alignment-ganctoma-orchestration
```

# kappjoha
```shell
prefect deployment build 03_fine-alignment/prefect_fine_alignment.py:estimate_z_flow_fields --name kappjoha -p SLURM -q kappjoha -sb github/gfriedri-em-alignment-flows-kappjoha --skip-upload -o 03_fine-alignment/deployments/kappjoha-estimate_z_flow_fields.yaml -ib process/gfriedri-em-alignment-kappjoha-gpu


prefect deployment build 03_fine-alignment/prefect_fine_alignment.py:estimate_z_flow_fields_parallel --name kappjoha -p SLURM -q kappjoha -sb github/gfriedri-em-alignment-flows-kappjoha --skip-upload -o 03_fine-alignment/deployments/kappjoha-estimate_z_flow_fields-parallel.yaml -ib process/gfriedri-em-alignment-kappjoha-orchestration


prefect deployment build 03_fine-alignment/prefect_relax_meshes.py:relax_meshes_in_blocks_flow --name kappjoha -p SLURM -q kappjoha -sb github/gfriedri-em-alignment-flows-kappjoha --skip-upload -o 03_fine-alignment/deployments/kappjoha-relax_meshes_in_blocks.yaml -ib process/gfriedri-em-alignment-kappjoha-gpu


prefect deployment build 03_fine-alignment/prefect_relax_meshes.py:relax_mehses_cross_blocks_flow --name kappjoha -p SLURM -q kappjoha -sb github/gfriedri-em-alignment-flows-kappjoha --skip-upload -o 03_fine-alignment/deployments/kappjoha-relax_meshes_cross_blocks.yaml -ib process/gfriedri-em-alignment-kappjoha-gpu


prefect deployment build 03_fine-alignment/prefect_relax_meshes.py:relax_meshes_flow --name kappjoha -p SLURM -q kappjoha -sb github/gfriedri-em-alignment-flows-kappjoha --skip-upload -o 03_fine-alignment/deployments/kappjoha-relax_meshes.yaml -ib process/gfriedri-em-alignment-kappjoha-orchestration
```
