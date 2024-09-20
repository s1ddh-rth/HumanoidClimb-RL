from gymnasium.envs.registration import register
register(
    id='HumanoidClimb-v0',
    entry_point='humanoid_climb.env:HumanoidClimbEnv'
)