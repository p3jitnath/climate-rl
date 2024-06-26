# Towards improving weather and climate models using reinforcement learning
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8099812.svg)](https://doi.org/10.5281/zenodo.11960239) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This GitHub repository contains the code, data, and figures for the Master of Research (MRes) report [Towards improving weather and climate models using reinforcement learning](/mres/report.pdf).

## Overview

This research explores the integration of reinforcement learning (RL) with idealised climate models, marking a pioneering effort in AI-assisted climate modelling. Utilising 16 experiments across 8 prominent continuous action, model-free RL algorithms (REINFORCE, DDPG, DPG, TD3, PPO, TRPO, SAC, TQC), the study examines two distinct RL environments: Simple Climate Bias Correction and Radiative-Convective Equilibrium. Conducted on the high-performance JASMIN computing infrastructure, the findings demonstrate the effectiveness of off-policy algorithms like DDPG, TD3, and TQC in bias correction, and on-policy algorithms like DPG, PPO, and TRPO in model equilibrium. Results show significant bias reductions up to 90%, underscoring the potential of RL-based parameterisation in enhancing the accuracy and efficiency of global climate models.

## Project Structure

```
climate-rl/
│
├── climate-envs/           # Gymansium-based climate environments used in the project
├── datasets/               # Dataset files used in simulations
├── misc/                   # Script files for batch-processing runs on JASMIN
├── notebooks/              # Jupyter notebooks for data analysis and results visualization
├── param_tune/             # Code for Ray-powered parameter tuning via multiple parallel batch jobs
├── rl-algos/               # cleanRL-styled source code for RL models
├──.editorconfig            # Config file for code editor specific settings
├──.gitignore               # Config file to skip certain files from version control
├──.pre-commit-config.yaml  # Config file for pre-commit tools for maintaining code quality
├── environment.yml         # Conda environment file
└── pyproject.toml          # Config file for python project
```

## Environment Setup

To set up the project environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/nathzi1505/climate-rl.git
   cd climate-rl
   ```

2. Install dependencies:
   - Using Conda (recommended):
     ```bash
     conda env create -f environment.yml
     conda activate climate-rl-env
     ```

3. Install the climate RL environments:
    ```
    cd climate-envs/ && pip install . && cd ../
    ```

4. Replace the `BASE_DIR` location:
   ```
   find . -type f -exec sed -i "s|/gws/nopw/j04/ai4er/users/pn341/climate-rl|$(pwd)|g" {} +
   ```

5. [Optional] Download runs:
    ```bash
    wget https://zenodo.org/records/11960239/files/runs_2024-06-17_15-38.zip
    unzip -qq runs_2024-06-17_15-38.zip -d runs
    ```

## Usage

1. To run an RL algorithm (for eg. DDPG) with an environment (for eg. `RadiativeConvectiveModel-v0`) over 15000 timesteps with 500 steps in each episode.
```
python ./rl-algos/ddpg/main.py --env_id "RadiativeConvectiveModel-v0" --total_timesteps 15000 --num-steps 500
```
> [!NOTE]
> Max. value for `num-steps` can be found [here](/climate-envs/climate_envs/__init__.py).

2. For detailed information regarding each option passed to the algorithm use `-h`.
```
python ./rl-algos/ddpg/main.py -h
```
> [!NOTE]
> Command line examples to run RL algorithms using SLURM can be found in `run` files [here](/misc/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any queries or further information, please contact [Pritthijit Nath](mailto:pn341@cam.ac.uk).
