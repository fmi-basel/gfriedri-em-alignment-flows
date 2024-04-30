#!/bin/bash
#SBATCH --job-name=Fine-Alignment
#SBATCH --cpus-per-task=2
#SBATCH --output=run-%j.out
#SBATCH --error=run-%j.err
#SBATCH --partition=orchestration
#SBATCH --oversubscribe
#SBATCH --mem=2GB
#SBATCH --time=48:00:00
if [ -z "$1" ]; then
        echo "[ERROR] [$(date -Iseconds)] [$$] SLURM account not provided."
        exit 1
fi
#SBATCH --account="$1"

set -eu

function display_memory_usage() {
        set +eu
        echo -n "[INFO] [$(date -Iseconds)] [$$] Max memory usage in bytes: "
        cat /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${SLURM_JOB_ID}/memory.max_usage_in_bytes
        echo
}

trap display_memory_usage EXIT

START=$(date +%s)
STARTDATE=$(date -Iseconds)
echo "[INFO] [$STARTDATE] [$$] Starting SLURM job $SLURM_JOB_ID"
echo "[INFO] [$STARTDATE] [$$] Running in $(hostname -s)"
echo "[INFO] [$STARTDATE] [$$] Working directory: $(pwd)"

### PUT YOUR CODE IN THIS SECTION

root_dir="$(pwd)/../.."

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('../../infrastructure/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$root_dir/infrastructure/miniforge3/etc/profile.d/conda.sh" ]; then
        . "$root_dir/infrastructure/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="$root_dir/infrastructure/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "$root_dir/infrastructure/miniforge3/etc/profile.d/mamba.sh" ]; then
    . "$root_dir/infrastructure/miniforge3/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1
mamba run -p "$root_dir/infrastructure/miniforge3/envs/nf" nextflow run "$root_dir/gfriedri-em-alignment-flows/s03_fine-alignment/workflow.nf" --config $(pwd)/fine-align.yaml --rm_config $(pwd)/relax-meshes.yaml --wf_config $(pwd)/warp-final.yaml -profile slurm -with-report -resume

### END OF PUT YOUR CODE IN THIS SECTION

END=$(date +%s)
ENDDATE=$(date -Iseconds)
echo "[INFO] [$ENDDATE] [$$] Workflow execution time \(seconds\) : $(( $END-$START ))"
