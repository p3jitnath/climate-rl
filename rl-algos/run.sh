# 1. define the base directory
BASE_DIR="/gws/nopw/j04/ai4er/users/pn341/climate-rl/rl-algos"

# 2. list of algorithms
ALGOS=("ddpg" "ppo" "sac" "td3" "tqc")

# 3. loop through each algorithm and execute the script
for ALGO in "${ALGOS[@]}"; do
    cd "$BASE_DIR/$ALGO"
    python main.py
done
