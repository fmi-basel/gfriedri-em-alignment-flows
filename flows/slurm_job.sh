#!/bin/bash
#SBATCH --account=dlthings
#SBATCH --job-name=warp-job
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=run-%j.out
#SBATCH --error=run-%j.err
#SBATCH --partition=gpu_short
#SBATCH --mem=96GB
#SBATCH --constraint infiniband

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

# Setup micromamba
export MAMBA_EXE="/tungstenfs/scratch/gmicro_share/_prefect/micromamba/bin/micromamba"
export MAMBA_ROOT_PREFIX="/tungstenfs/scratch/gmicro_share/_prefect/micromamba"
export MAMBA_ROOT_ENVIRONMENT="/tungstenfs/scratch/gmicro_share/_prefect/micromamba"

eval "$($MAMBA_ROOT_ENVIRONMENT/bin/micromamba shell hook -s posix)"

# Run Python script with micromamba in the 'faim-hcs' environment
micromamba run -n test_gfriedri-em-alignment-flows python warp_volume_script.py

### END OF PUT YOUR CODE IN THIS SECTION

END=$(date +%s)
ENDDATE=$(date -Iseconds)
echo "[INFO] [$ENDDATE] [$$] Workflow execution time \(seconds\) : $(( $END-$START ))"
