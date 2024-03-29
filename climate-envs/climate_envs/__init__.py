import json

from gymnasium.envs.registration import register

with open("rl-algos/config.json", "r") as file:
    config = json.load(file)

register(
    id="SimpleClimateBiasCorrection-v0",
    entry_point="climate_envs.envs:SimpleClimateBiasCorrectionEnv",
    max_episode_steps=config["max_episode_steps"],
)
