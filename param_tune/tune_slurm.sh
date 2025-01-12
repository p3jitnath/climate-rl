#!/bin/sh

#SBATCH --job-name=pn341_ray_slurm_optimise
#SBATCH --output=/gws/nopw/j04/ai4er/users/pn341/climate-rl/slurm/ray_slurm_%j.out
#SBATCH --error=/gws/nopw/j04/ai4er/users/pn341/climate-rl/slurm/ray_slurm_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --partition=orchid
#SBATCH --account=orchid

BASE_DIR=/gws/nopw/j04/ai4er/users/pn341/climate-rl
LOG_DIR="$BASE_DIR/slurm"

set -x

# 1a. Function to display usage
usage() {
    echo "Usage: sbatch $1 --algo <algo>"
    exit 1
}

# 1b. Check if no arguments were passed
if [ "$#" -eq 0 ]; then
    usage
fi

# 1c. Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --algo) # Extract the algo value
            ALGO="$2"
            shift 2
            ;;
        *) # Handle unknown option
            usage
            ;;
    esac
done

# 1d. Check if ALGO is set
if [ -z "$ALGO" ]; then
    echo "Error: Algo is required."
    usage
fi

# __doc_head_address_start__

# Checking the conda environment
echo "PYTHON: $(which python)"

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# If we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPv6 address detected. We split the IPv4 address as $head_node_ip"
fi

# __doc_head_address_end__

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --include-dashboard=False --num-gpus 2 --block & \
    --output="$LOG_DIR/ray_slurm_${SLURM_JOB_ID}.out" \
    --error="$LOG_DIR/ray_slurm_${SLURM_JOB_ID}.err"

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 30

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 2 --block & \
        --output="$LOG_DIR/ray_slurm_${SLURM_JOB_ID}.out" \
        --error="$LOG_DIR/ray_slurm_${SLURM_JOB_ID}.err"
    sleep 30
done

python -u $BASE_DIR/param_tune/tune.py --algo $ALGO --exp_id "v0-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 60000 --num_steps 200 # --actor_layer_size 64 --critic_layer_size 64
