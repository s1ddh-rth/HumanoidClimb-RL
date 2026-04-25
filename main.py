import gymnasium as gym
import pybullet as p
from stable_baselines3 import PPO
from humanoid_climb.climbing_config import ClimbingConfig


# load config, create and reset env, load policy
config = ClimbingConfig('./config.json')
env = gym.make('HumanoidClimb-v0',
               render_mode='human',
               max_ep_steps=10000000,
               state_file="./humanoid_climb/states/state_10_9_2_1.npz",
               config=config)
obs, info = env.reset()
model = PPO.load("./humanoid_climb/models/4_18_18_n_n.zip", env=env)
# model = PPO.load("./humanoid_climb/models/2_10_9_2_n.zip", env=env)
# model = PPO.load("./humanoid_climb/models/3_10_9_2_1.zip", env=env)
# model = PPO.load("./humanoid_climb/models/4_10_13_2_1.zip", env=env)

# prepare variables for the while loop
done, truncated = False, False
score, step = 0, 0
paused = False

print('In PyBullet window, press:')
print('\tr          reset episode')
print('\tspacebar   pause episode')
print('\tq          quit')

while True:
    if not paused:
        # use policy to predict next action, then step environment
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        step += 1

    if (step % 200 == 0) and not paused:
        print(f'step {step}: score {score}')

    # if episode terminates, pause (user needs to reset)
    # note that episode currently does not terminate after goal has successfully been reached
    if (done or truncated) and not paused:
        print(f'terminated after {step} steps. score: {score}')
        paused = True

    # get keys
    keys = p.getKeyboardEvents()
    # reset on r
    r_key = ord('r')
    if r_key in keys and keys[r_key] & p.KEY_WAS_TRIGGERED:
        print('resetting...')
        done = False
        truncated = False
        paused = False
        score = 0
        step = 0
        obs, info = env.reset()

    # pause on space
    pause_key = ord(' ')
    if pause_key in keys and keys[pause_key] & p.KEY_WAS_TRIGGERED:
        paused = not paused
        print('paused' if paused else 'unpaused')

    # quit on q
    q_key = ord('q')
    if q_key in keys and keys[q_key] & p.KEY_WAS_TRIGGERED:
        print('quitting...')
        break

env.close()
