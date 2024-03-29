from gymnasium.envs.registration import register

register(
    id="SimpleClimateBiasCorrection-v0",
    entry_point="climate_envs.envs:SimpleClimateBiasCorrectionEnv",
    max_episode_steps=200,
)
