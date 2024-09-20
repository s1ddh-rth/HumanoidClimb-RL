from gymnasium.envs.registration import register
register(
    id='TorsoClimb-v0',
    entry_point='torso_climb.env:TorsoClimbEnv'
)