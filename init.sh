eval "$($(pwd)/infrastructure/miniforge3/bin/conda shell.bash hook)"

export PIP_CACHE_DIR="$(pwd)/infrastructure/.PIP_CACHE"

export NXF_OPTS="-Xms500M -Xmx2G"
export NXF_EXECUTOR="slurm"
export NXF_ANSI_LOG=false
export NXF_HOME="$(pwd)/infrastructure/.nxf_home"

export CONDA_OVERRIDE_CUDA=11.8

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('$(pwd)/infrastructure/apps/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$(pwd)/infrastructure/apps/miniforge3/etc/profile.d/conda.sh" ]; then
        . "$(pwd)/infrastructure/apps/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="$(pwd)/infrastructure/apps/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "$(pwd)/infrastructure/apps/miniforge3/etc/profile.d/mamba.sh" ]; then
    . "$(pwd)/infrastructure/apps/miniforge3/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<


mamba activate infrastructure/miniforge3/envs/gfriedri-em-alignment-flows*
