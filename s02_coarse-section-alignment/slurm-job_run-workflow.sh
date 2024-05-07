#!/bin/bash
#SBATCH --job-name=Coarse-Alignment
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
mamba run -p "$root_dir/infrastructure/envs/nf" nextflow run "$root_dir/s02_coarse-section-alignment/workflow.nf" --config $(pwd)/coarse-align.yaml -profile slurm -with-report -resume -disable-jobs-cancellation

### END OF PUT YOUR CODE IN THIS SECTION

END=$(date +%s)
ENDDATE=$(date -Iseconds)
echo "[INFO] [$ENDDATE] [$$] Workflow execution time \(seconds\) : $(( $END-$START ))"
