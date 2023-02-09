# Build
```
prefect deployment build flows/add_sample.py:add_sample_to_experiment_flow -n "add-sample" -q slurm -sb github/gfriedri-em-alignment-flows --skip-upload -o deployment/add_sample.yaml -ib process/slurm-gfriedri-em-alignment-flows

prefect deployment build flows/add_sections.py:add_sections_to_sample_flow -n "add-sections" -q slurm -sb github/gfriedri-em-alignment-flows --skip-upload -o deployment/add_sections.yaml -ib process/slurm-gfriedri-em-alignment-flows

prefect deployment build flows/add_volume.py:add_volume_flow -n "add-volume" -q slurm -sb github/gfriedri-em-alignment-flows --skip-upload -o deployment/add_volume.yaml -ib process/slurm-gfriedri-em-alignment-flows

prefect deployment build flows/create_experiment.py:create_experiment_flow -n "create-experiment" -q slurm -sb github/gfriedri-em-alignment-flows --skip-upload -o deployment/create_experiment.yaml -ib process/slurm-gfriedri-em-alignment-flows

prefect deployment build flows/tile_registration.py:tile_registration_flow -n "tile-registration" -q slurm -sb github/gfriedri-em-alignment-flows --skip-upload -o deployment/tile_registration.yaml -ib process/slurm-gfriedri-em-alignment-flows

prefect deployment build flows/warp_sections.py:warp_sections_flow -n "warp-sections" -q slurm -sb github/gfriedri-em-alignment-flows --skip-upload -o deployment/warp_sections.yaml -ib process/slurm-gfriedri-em-alignment-flows

prefect deployment build flows/fine_alignment.py:flow_field_estimation -n "default" -q slurm -sb github/gfriedri-em-alignment-flows --skip-upload -o deployment/fine_alignment_ffe.yaml -ib process/gfriedri-estimate-flow-field-3d

prefect deployment build flows/fine_alignment.py:parallel_flow_field_estimation -n "default" -q slurm -sb github/gfriedri-em-alignment-flows --skip-upload -o deployment/fine_alignment_pffe.yaml -ib process/slurm-gfriedri-em-alignment-flows

prefect deployment build flows/fine_alignment.py:optimize_mesh -n "default" -q slurm -sb github/gfriedri-em-alignment-flows --skip-upload -o deployment/optimize_mesh.yaml -ib process/gfriedri-estimate-flow-field-3d
```

# Deploy
```
prefect deployment apply deployment/*.yaml
```
