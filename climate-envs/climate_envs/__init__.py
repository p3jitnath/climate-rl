import json

from gymnasium.envs.registration import register

BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl"

register(
    id="SimpleClimateBiasCorrection-v0",
    entry_point="climate_envs.envs:SimpleClimateBiasCorrectionEnv",
    max_episode_steps=200,
)

register(
    id="SimpleClimateBiasCorrection-v1",
    entry_point="climate_envs.envs.simple_climate_bias_correction_v1:SimpleClimateBiasCorrectionEnv",
    max_episode_steps=200,
)

register(
    id="SimpleClimateBiasCorrection-v2",
    entry_point="climate_envs.envs.simple_climate_bias_correction_v2:SimpleClimateBiasCorrectionEnv",
    max_episode_steps=200,
)

register(
    id="RadiativeConvectiveModel-v0",
    entry_point="climate_envs.envs:RadiativeConvectiveModelEnv",
    max_episode_steps=500,
)

register(
    id="RadiativeConvectiveModel-v1",
    entry_point="climate_envs.envs:RadiativeConvectiveModelEnv",
    max_episode_steps=500,
)
