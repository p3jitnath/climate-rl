#!/bin/sh

BASE_DIR="/gws/nopw/j04/ai4er/users/pn341/climate-rl"

runs=("$BASE_DIR"/param_tune/results/rce17-v1-*)

for run in "${runs[@]}"; do
    exp_id=$(basename "$run")
    echo "Reading $run ..."
    python $BASE_DIR/param_tune/results/read.py --exp_id $exp_id
done
