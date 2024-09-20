from gymnasium.envs.registration import register
register(
    id='PendulumClimb-v0',
    entry_point='pendulum_climb.envs:PendulumClimbEnv'
)