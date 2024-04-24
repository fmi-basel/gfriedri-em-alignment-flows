#!/usr/bin/env nextflow

params.config = "fine-align.yaml"

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
    path section_dirs

    output:
    path "flow_paths.yaml"

    script:
    """
    python $baseDir/s02_estimate_flow_fields.py --config $config --section_dirs $section_dirs
    """
}

workflow {
    stitched_section_dirs = PARSESECTIONS(params.config)
    flow_paths = ESTIMATEFLOWFIELDS(stitched_section_dirs.flatten())
}
