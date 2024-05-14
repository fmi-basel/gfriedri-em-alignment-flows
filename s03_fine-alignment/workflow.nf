#!/usr/bin/env nextflow

params.config = "fine_alignment_config.yaml"
params.warp_config = "warp_config.yaml"

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

process RELAXBLOCKS {
    label 'gpu'

    input:
    path config
    path block_sections

    output:
    path "relaxed_meshes_in_blocks.yaml"

    script:
    """
    python $baseDir/s04_relax_mesh_blocks.py --config $config --block_sections $block_sections
    """
}

process RELAXCROSSBLOCKS {
    label 'gpu'

    input:
    path config
    path relaxed_blocks

    output:
    path "map.yaml"

    script:
    """
    python $baseDir/s05_relax_mesh_cross_blocks.py --config $config --relaxed_blocks $relaxed_blocks
    """
}

process PREPAREWARPING {
    label 'cpu'

    input:
    path config
    path warp_config
    path map

    output:
    path "sections_for_warping_*.yaml"

    script:
    """
    python $baseDir/s06_prepare_warping.py --config $config --warp_config $warp_config --map $map
    """
}

process WARPSECTIONS {
    label 'cpu'

    input:
    path sections_for_warping

    script:
    """
    python $baseDir/s07_warp_sections.py --config $sections_for_warping
    """
}

workflow {
    stitched_section_dirs = PARSESECTIONS(params.config)
    flow_paths = ESTIMATEFLOWFIELDS(params.config, stitched_section_dirs.flatten())
    blocks = PREPAREMESHRELAXATION(params.config, flow_paths.collectFile())
    relaxed_blocks = RELAXBLOCKS(params.config, blocks.flatten())
    map = RELAXCROSSBLOCKS(params.config, relaxed_blocks.collectFile())
    sections_for_warping = PREPAREWARPING(params.config, params.warp_config, map)
    WARPSECTIONS(sections_for_warping.flatten())
}
