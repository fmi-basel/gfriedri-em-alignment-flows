# buchtimo
```shell
prefect deployment build 03_fine-alignment/prefect_fine_alignment.py:estimate_z_flow_fields --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 03_fine-alignment/deployments/buchtimo-estimate_z_flow_fields.yaml -ib process/gfriedri-em-alignment-buchtimo-gpu


prefect deployment build 03_fine-alignment/prefect_fine_alignment.py:estimate_z_flow_fields_parallel --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 03_fine-alignment/deployments/buchtimo-estimate_z_flow_fields.yaml -ib process/gfriedri-em-alignment-buchtimo-orchestration
```

# ganctoma
```shell
prefect deployment build 03_fine-alignment/prefect_fine_alignment.py:estimate_z_flow_fields --name ganctoma -p SLURM -q ganctoma -sb github/gfriedri-em-alignment-flow-ganctoma --skip-upload -o 03_fine-alignment/deployments/ganctoma-estimate_z_flow_fields.yaml -ib process/gfriedri-em-alignment-ganctoma-gpu


prefect deployment build 03_fine-alignment/prefect_fine_alignment.py:estimate_z_flow_fields_parallel --name ganctoma -p SLURM -q ganctoma -sb github/gfriedri-em-alignment-flow-ganctoma --skip-upload -o 03_fine-alignment/deployments/ganctoma-estimate_z_flow_fields.yaml -ib process/gfriedri-em-alignment-ganctoma-orchestration
```

# kappjoha
```shell
prefect deployment build 03_fine-alignment/prefect_fine_alignment.py:estimate_z_flow_fields --name kappjoha -p SLURM -q kappjoha -sb github/gfriedri-em-alignment-flow-kappjoha --skip-upload -o 03_fine-alignment/deployments/kappjoha-estimate_z_flow_fields.yaml -ib process/gfriedri-em-alignment-kappjoha-gpu


prefect deployment build 03_fine-alignment/prefect_fine_alignment.py:estimate_z_flow_fields_parallel --name kappjoha -p SLURM -q kappjoha -sb github/gfriedri-em-alignment-flow-kappjoha --skip-upload -o 03_fine-alignment/deployments/kappjoha-estimate_z_flow_fields.yaml -ib process/gfriedri-em-alignment-kappjoha-orchestration
```
