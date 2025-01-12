# sbatch ./param_tune/tune_slurm.sh --algo reinforce
# sbatch ./param_tune/tune_slurm.sh --algo dpg
# sbatch ./param_tune/tune_slurm.sh --algo ddpg
# sbatch ./param_tune/tune_slurm.sh --algo td3
# sbatch ./param_tune/tune_slurm.sh --algo ppo
# sbatch ./param_tune/tune_slurm.sh --algo trpo
# sbatch ./param_tune/tune_slurm.sh --algo sac
# sbatch ./param_tune/tune_slurm.sh --algo tqc
# sbatch ./param_tune/tune_slurm.sh --algo avg

sbatch ./param_tune/tune_slurm.sh --algo avg
