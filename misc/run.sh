#!/bin/sh

# 1a. Function to display usage
usage() {
    echo "Usage: $0 --tag <tag> --env_id <env_id>"
    exit 1
}

# 1b. Check if no arguments were passed
if [ "$#" -eq 0 ]; then
    usage
fi

# 1c. Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tag) # Extract the tag value
            TAG="$2"
            shift 2
            ;;
        --env_id) # Extract the env_id value
            ENV_ID="$2"
            shift 2
            ;;
        *) # Handle unknown option
            usage
            ;;
    esac
done

# 1d. Check if TAG is set
if [ -z "$TAG" ]; then
    echo "Error: Tag is required."
    usage
fi

# 1e. Check if ENV_ID is set
if [ -z "$ENV_ID" ]; then
    echo "Error: Environment id is required."
    usage
fi

# 2. Define the base directory
BASE_DIR="/gws/nopw/j04/ai4er/users/pn341/climate-rl"

# 3. List of algorithms
# ALGOS=("ddpg" "dpg" "td3" "reinforce" "trpo" "ppo" "sac" "tqc" "avg")
ALGOS=("avg")

# 4. Get the current date and time in YYYY-MM-DD_HH-MM format
NOW=$(date +%F_%H-%M)
WANDB_GROUP="${TAG}_${NOW}"

# 5. Loop through each algorithm and execute the script
for ALGO in "${ALGOS[@]}"; do
    # Submit each algorithm run as a separate Slurm job
    sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=pn341_${ALGO}_${TAG}
#SBATCH --output=$BASE_DIR/slurm/${ALGO}_${TAG}_%j.out
#SBATCH --error=$BASE_DIR/slurm/${ALGO}_${TAG}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --partition=orchid
#SBATCH --account=orchid

conda activate venv
cd "$BASE_DIR"
export WANDB_MODE=offline
python -u "$BASE_DIR/rl-algos/$ALGO/main.py" --env_id "$ENV_ID" --optim_group "$TAG" --wandb_group "$WANDB_GROUP" --total_timesteps 10000 --num_steps 500 --capture_video_freq 10
EOT
    sleep 60
done
