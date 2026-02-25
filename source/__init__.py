from gymnasium.envs.registration import register

register(
    id="LiftReach-v0",
    entry_point=f"source.envs.lift_env:LiftEnv",
    max_episode_steps=100,
)
