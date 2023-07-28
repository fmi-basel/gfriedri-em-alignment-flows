# buchtimo
```shell
prefect deployment build 02_coarse-section-alignment/prefect_coarse_alignment.py:coarse_align_pairs --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 02_coarse-section-alignment/deployments/coarse_align_pairs.yaml -ib process/gfriedri-em-alignment-buchtimo-gpu


prefect deployment build 02_coarse-section-alignment/prefect_coarse_alignment.py:coarse_alignment --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 02_coarse-section-alignment/deployments/coarse_alignment.yaml -ib process/gfriedri-em-alignment-buchtimo-orchestration
```
