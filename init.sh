export PIP_CACHE_DIR="$(pwd)/infrastructure/.PIP_CACHE"

export NXF_OPTS="-Xms500M -Xmx2G"
export NXF_EXECUTOR="slurm"
export NXF_ANSI_LOG=false
export NXF_HOME="$(pwd)/infrastructure/.nxf_home"

export CONDA_OVERRIDE_CUDA=11.8

root_dir="$(pwd)"
# >>> mamba initialize >>>
# !! Contents within this block are managed by 'mamba init' !!
export MAMBA_EXE="$root_dir/infrastructure/micromamba/bin/micromamba";
export MAMBA_ROOT_PREFIX="$root_dir/infrastructure/micromamba";
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from mamba activate
fi
unset __mamba_setup
# <<< mamba initialize <<<

mamba activate infrastructure/envs/gfriedri-em-alignment-flows*
