import json

from gymnasium.envs.registration import register

BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl"

with open(f"{BASE_DIR}/rl-algos/config.json", "r") as file:
    config = json.load(file)

register(
    id="SimpleClimateBiasCorrection-v0",
    entry_point="climate_envs.envs:SimpleClimateBiasCorrectionEnv",
    max_episode_steps=config["max_episode_steps"],
)

register(
    id="SimpleClimateBiasCorrection-v1a",
    entry_point="climate_envs.envs.simple_climate_bias_correction_v1a:SimpleClimateBiasCorrectionEnv",
    max_episode_steps=config["max_episode_steps"],
)

register(
    id="SimpleClimateBiasCorrection-v1b",
    entry_point="climate_envs.envs.simple_climate_bias_correction_v1b:SimpleClimateBiasCorrectionEnv",
    max_episode_steps=config["max_episode_steps"],
)

register(
    id="SimpleClimateBiasCorrection-v1c",
    entry_point="climate_envs.envs.simple_climate_bias_correction_v1c:SimpleClimateBiasCorrectionEnv",
    max_episode_steps=config["max_episode_steps"],
)
