#!/usr/bin/env nextflow

params.config = "fine-align.yaml"
params.rm_config = "relax-meshes.yaml"

process PARSESECTIONS {
    label 'cpu'

    input:
    path config

    output:
    path 'section_dirs_chunk_*.yaml'

    script:
    """
    python $baseDir/s01_parse_sections.py --config $config
    """
}

process ESTIMATEFLOWFIELDS {
    label 'gpu'


    input:
    path config
    path section_dirs

    output:
    path "flow_paths.yaml"

    script:
    """
    python $baseDir/s02_estimate_flow_fields.py --config $config --section_dirs $section_dirs
    """
}

process PREPAREMESHRELAXATION {
    label 'cpu'

    input:
    path config
    path flow_paths

    output:
    path "mesh_relaxation_blocks_*.yaml"

    script:
    """
    python $baseDir/s03_prepare_mesh_relaxation.py --config $config --flow_paths $flow_paths
    """
}

workflow {
    stitched_section_dirs = PARSESECTIONS(params.config)
    flow_paths = ESTIMATEFLOWFIELDS(params.config, stitched_section_dirs.flatten())
    blocks = PREPAREMESHRELAXATION(params.rm_config, flow_paths.collectFile())
}
