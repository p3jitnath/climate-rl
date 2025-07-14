#!/bin/bash

BASE_DIR="/gws/nopw/j04/ai4er/users/pn341/climate-rl"
ALGOS=("ddpg" "dpg" "td3" "reinforce" "trpo" "ppo" "sac" "avg" "tqc")

GREEN="\033[0;32m"
RED="\033[0;31m"
NC="\033[0m"  # No Colour

# Print header
printf "%-50s" "Run Directory"
for algo in "${ALGOS[@]}"; do
  printf "%12s" "$algo"
done
echo

# Loop over top-level run directories
for top_dir in "$BASE_DIR"/runs/*/; do
  run_name=$(basename "$top_dir")
  printf "%-50s" "$run_name"
  for algo in "${ALGOS[@]}"; do
    count=$(find "$top_dir" -mindepth 1 -maxdepth 1 -type d -name "*_${algo}_*" | wc -l)
    if [[ "$count" -eq 1 ]] || ([[ "$count" -eq 9 ]] && [[ "$run_name" == *x9* ]]); then
      printf "${GREEN}%12d${NC}" "$count"
    else
      printf "${RED}%12d${NC}" "$count"
    fi
  done
  echo
done
