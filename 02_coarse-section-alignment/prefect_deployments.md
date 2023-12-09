# buchtimo
```shell
prefect deployment build 02_coarse-section-alignment/prefect_coarse_alignment.py:coarse_align_pairs --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 02_coarse-section-alignment/deployments/buchtimo-coarse_align_pairs.yaml -ib process/gfriedri-em-alignment-buchtimo-cpu


prefect deployment build 02_coarse-section-alignment/prefect_coarse_alignment.py:coarse_alignment --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 02_coarse-section-alignment/deployments/buchtimo-coarse_alignment.yaml -ib process/gfriedri-em-alignment-buchtimo-orchestration


prefect deployment build 02_coarse-section-alignment/prefect_create_coarse_aligned_stack.py:create_coarse_stack --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 02_coarse-section-alignment/deployments/buchtimo-create_coarse_stack.yaml -ib process/gfriedri-em-alignment-buchtimo-orchestration


prefect deployment build 02_coarse-section-alignment/prefect_create_coarse_aligned_stack.py:write_coarse_aligned_sections --name buchtimo -p SLURM -q buchtimo -sb github/gfriedri-em-alignment-flow-buchtimo --skip-upload -o 02_coarse-section-alignment/deployments/buchtimo-write_coarse_aligned_sections.yaml -ib process/gfriedri-em-alignment-buchtimo-cpu
```

# ganctoma
```shell
prefect deployment build 02_coarse-section-alignment/prefect_coarse_alignment.py:coarse_align_pairs --name ganctoma -p SLURM -q ganctoma -sb github/gfriedri-em-alignment-flows-ganctoma --skip-upload -o 02_coarse-section-alignment/deployments/ganctoma-coarse_align_pairs.yaml -ib process/gfriedri-em-alignment-ganctoma-cpu


prefect deployment build 02_coarse-section-alignment/prefect_coarse_alignment.py:coarse_alignment --name ganctoma -p SLURM -q ganctoma -sb github/gfriedri-em-alignment-flows-ganctoma --skip-upload -o 02_coarse-section-alignment/deployments/ganctoma-coarse_alignment.yaml -ib process/gfriedri-em-alignment-ganctoma-orchestration


prefect deployment build 02_coarse-section-alignment/prefect_create_coarse_aligned_stack.py:create_coarse_stack --name ganctoma -p SLURM -q ganctoma -sb github/gfriedri-em-alignment-flows-ganctoma --skip-upload -o 02_coarse-section-alignment/deployments/ganctoma-create_coarse_stack.yaml -ib process/gfriedri-em-alignment-ganctoma-orchestration


prefect deployment build 02_coarse-section-alignment/prefect_create_coarse_aligned_stack.py:write_coarse_aligned_sections --name ganctoma -p SLURM -q ganctoma -sb github/gfriedri-em-alignment-flows-ganctoma --skip-upload -o 02_coarse-section-alignment/deployments/ganctoma-write_coarse_aligned_sections.yaml -ib process/gfriedri-em-alignment-ganctoma-cpu
```

# kappjoha
```shell
prefect deployment build 02_coarse-section-alignment/prefect_coarse_alignment.py:coarse_align_pairs --name kappjoha -p SLURM -q kappjoha -sb github/gfriedri-em-alignment-flows-kappjoha --skip-upload -o 02_coarse-section-alignment/deployments/kappjoha-coarse_align_pairs.yaml -ib process/gfriedri-em-alignment-kappjoha-cpu


prefect deployment build 02_coarse-section-alignment/prefect_coarse_alignment.py:coarse_alignment --name kappjoha -p SLURM -q kappjoha -sb github/gfriedri-em-alignment-flows-kappjoha --skip-upload -o 02_coarse-section-alignment/deployments/kappjoha-coarse_alignment.yaml -ib process/gfriedri-em-alignment-kappjoha-orchestration


prefect deployment build 02_coarse-section-alignment/prefect_create_coarse_aligned_stack.py:create_coarse_stack --name kappjoha -p SLURM -q kappjoha -sb github/gfriedri-em-alignment-flows-kappjoha --skip-upload -o 02_coarse-section-alignment/deployments/kappjoha-create_coarse_stack.yaml -ib process/gfriedri-em-alignment-kappjoha-orchestration


prefect deployment build 02_coarse-section-alignment/prefect_create_coarse_aligned_stack.py:write_coarse_aligned_sections --name kappjoha -p SLURM -q kappjoha -sb github/gfriedri-em-alignment-flows-kappjoha --skip-upload -o 02_coarse-section-alignment/deployments/kappjoha-write_coarse_aligned_sections.yaml -ib process/gfriedri-em-alignment-kappjoha-cpu
```
