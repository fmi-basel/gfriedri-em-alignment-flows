# buchtimo
```shell
prefect deployment build 02_coarse-section-alignment/prefect_coarse_alignment.py:coarse_align_pairs --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 02_coarse-section-alignment/deployments/coarse_align_pairs.yaml -ib process/gfriedri-em-alignment-buchtimo-gpu


prefect deployment build 02_coarse-section-alignment/prefect_coarse_alignment.py:coarse_alignment --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 02_coarse-section-alignment/deployments/coarse_alignment.yaml -ib process/gfriedri-em-alignment-buchtimo-orchestration


prefect deployment build 02_coarse-section-alignment/prefect_create_coarse_aligned_stack.py:create_coarse_stack --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 02_coarse-section-alignment/deployments/create_coarse_stack.yaml -ib process/gfriedri-em-alignment-buchtimo-orchestration


prefect deployment build 02_coarse-section-alignment/prefect_create_coarse_aligned_stack.py:write_coarse_aligned_sections --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 02_coarse-section-alignment/deployments/write_coarse_aligned_sections.yaml -ib process/gfriedri-em-alignment-buchtimo-cpu
```

# ganctoma
```shell
prefect deployment build 02_coarse-section-alignment/prefect_coarse_alignment.py:coarse_align_pairs --name ganctoma -p SLURM -q ganctoma -sb github/gfriedri-em-alignment-flows-ganctoma --skip-upload -o 02_coarse-section-alignment/deployments/coarse_align_pairs.yaml -ib process/gfriedri-em-alignment-ganctoma-gpu


prefect deployment build 02_coarse-section-alignment/prefect_coarse_alignment.py:coarse_alignment --name ganctoma -p SLURM -q ganctoma -sb github/gfriedri-em-alignment-flows-ganctoma --skip-upload -o 02_coarse-section-alignment/deployments/coarse_alignment.yaml -ib process/gfriedri-em-alignment-ganctoma-orchestration


prefect deployment build 02_coarse-section-alignment/prefect_create_coarse_aligned_stack.py:create_coarse_stack --name ganctoma -p SLURM -q ganctoma -sb github/gfriedri-em-alignment-flows-ganctoma --skip-upload -o 02_coarse-section-alignment/deployments/create_coarse_stack.yaml -ib process/gfriedri-em-alignment-ganctoma-orchestration


prefect deployment build 02_coarse-section-alignment/prefect_create_coarse_aligned_stack.py:write_coarse_aligned_sections --name ganctoma -p SLURM -q ganctoma -sb github/gfriedri-em-alignment-flows-ganctoma --skip-upload -o 02_coarse-section-alignment/deployments/write_coarse_aligned_sections.yaml -ib process/gfriedri-em-alignment-ganctoma-cpu
```
