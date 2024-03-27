#!/usr/bin/env nextflow

params.config = "tile-stitching.config"

process PARSEDATA {
    label 'cpu'

    input:
    path config

    output:
    path 'section_yaml_files_chunk_*.yaml'

    script:
    """
    python $baseDir/s01_parse_data.py --config $config
    """
}

process REGISTERTILES {
    label 'cpu'

    maxForks 1

    errorStrategy = 'ignore'

    input:
    path config
    path section_yaml_files

    output:
    path 'meshes.yaml'
    path 'errors.yaml', optional: true

    script:
    """
    python $baseDir/s02_register_tiles.py --config $config --section_yaml_files $section_yaml_files
    """
}

process WARPTILES {
    label 'cpu'

    maxForks 2

    errorStrategy = 'ignore'

    input:
    path config
    path meshes

    output:
    path 'warped_tiles.yaml'

    script:
    """
    python $baseDir/s03_warp_tiles.py --config $config --meshes $meshes
    """
}

workflow {
    section_dirs = PARSEDATA(params.config)
    (meshes, errors) = REGISTERTILES(params.config, section_dirs.flatten())
    warped_tiles = WARPTILES(params.config, meshes)
}
