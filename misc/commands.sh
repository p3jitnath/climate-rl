# List of useful commands

# 1. Run a RL algorithm
# scbc
python ./rl-algos/tqc/main.py --env_id "SimpleClimateBiasCorrection-v0" --wandb_group "scbc-v0-test" --no-track --total_timesteps 60000 --num_steps 200 --capture_video_freq 100
# rce
python ./rl-algos/tqc/main.py --env_id "RadiativeConvectiveModel-v0" --wandb_group "rce-v0-test" --no-track --total_timesteps 10000 --num_steps 500 --capture_video_freq 10

# 2. Print the last 10 lines of a SLURM job
tail -n 10 $(find slurm/ -type f -name '*25922108*.out')

# 3. Perform integrity count checks
source ./misc/count.sh

# 4. Scrape an algorithm off the runs/ and videos/ folder
find runs/ -type d -path '*/*tqc*' -exec rm -r {} +
find videos/ -type d -path '*/*tqc*' -exec rm -r {} +
