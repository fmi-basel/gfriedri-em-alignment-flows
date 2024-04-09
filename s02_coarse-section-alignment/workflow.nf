#!/usr/bin/env nextflow

params.config = "coarse-align.config"

process PARSESECTIONS {
    label 'cpu_short'

    input:
    path config

    output:
    path 'section_dirs_chunk_*.yaml'

    script:
    """
    python $baseDir/s01_parse_sections.py --config $config
    """
}

process COARSEALIGN {
    label 'cpu'

    errorStrategy = 'ignore'

    input:
    path stitched_section_dirs

    output:
    path 'processed_dirs.yaml'

    script:
    """
    python $baseDir/s02_coarse_align_section_pairs.py --stitched_section_dirs $stitched_section_dirs
    """
}

process CREATEZARR {
    label 'cpu_short'

    input:
    path config
    path processed_dirs_collection

    output:
    path 'zarr_dir.yaml'
    path 'processed_chunks_*.yaml'

    script:
    """
    python $baseDir/s03_create_zarr.py --config $config --coarse_aligned_sections $processed_dirs_collection
    """
}

process WARPSECTIONS {
    label 'cpu'

    errorStrategy = 'ignore'

    input:
    path processed_dirs
    path config
    path zarr

    output:
    path 'warped_sections.yaml'

    script:
    """
    python $baseDir/s04_warp_sections.py --config $config --zarr $zarr --chunks $processed_dirs
    """
}

workflow {
    stitched_section_dirs = PARSESECTIONS(params.config)
    processed_dirs = COARSEALIGN(stitched_section_dirs.flatten())
    (zarr_dir, processed_chunks) = CREATEZARR(params.config, processed_dirs.collectFile())
    warped_tiles = WARPSECTIONS(processed_chunks.flatten(), params.config, zarr_dir.first())
}
