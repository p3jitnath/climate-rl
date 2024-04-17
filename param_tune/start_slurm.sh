#!/bin/sh

#SBATCH --job-name=pn341_ray_slurm_optimise
#SBATCH --output=/gws/nopw/j04/ai4er/users/pn341/climate-rl/slurm/ray_slurm_optimise.out
#SBATCH --error=/gws/nopw/j04/ai4er/users/pn341/climate-rl/slurm/ray_slurm_optimise.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=orchid
#SBATCH --account=orchid

BASE_DIR=/gws/nopw/j04/ai4er/users/pn341/climate-rl

set -x

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
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

# __doc_head_address_end__

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 1 --block &

# # optional, though may be useful in certain versions of Ray < 1.0.
sleep 20

# # number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 1 --block &
    sleep 10
done

python $BASE_DIR/param_tune/tune.py --algo reinforce
