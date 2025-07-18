#!/bin/sh

# Base directory
BASE_DIR="/gws/nopw/j04/ai4er/users/pn341/climate-rl"

# Array of runs
runs=("$BASE_DIR"/param_tune/results/scbc*)

# Loop through each run
for run in "${runs[@]}"; do
    exp_id=$(basename "$run")
    echo "Preparing to run $exp_id ..."

    # Submit the job to SLURM
    sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=results_$exp_id
#SBATCH --output=$BASE_DIR/slurm/results_$exp_id_%j.out
#SBATCH --error=$BASE_DIR/slurm/results_$exp_id_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --account=ai4er
#SBATCH --partition=standard
#SBATCH --qos=high

conda activate venv
python $BASE_DIR/misc/scbc_generate_results.py --exp_id $exp_id
EOT
done

echo "All jobs submitted to SLURM."
