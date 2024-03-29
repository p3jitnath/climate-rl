# 1. define the base directory
BASE_DIR="/gws/nopw/j04/ai4er/users/pn341/climate-rl"

# 2. list of algorithms
ALGOS=("ddpg" "ppo") # "sac" "td3" "tqc"

# 3. loop through each algorithm and execute the script
cd "$BASE_DIR"
for ALGO in "${ALGOS[@]}"; do
    python "$BASE_DIR/rl-algos/$ALGO/main.py"
done
