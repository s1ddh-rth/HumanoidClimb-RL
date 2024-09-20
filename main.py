import os
import random

import gymnasium as gym
import pendulum_climb
import torso_climb
import humanoid_climb
from torso_climb.env.torso_climb_env import Reward
import pybullet as p
import time
from stable_baselines3 import PPO, SAC
import humanoid_climb.stances as stances


stances.set_root_path("./humanoid_climb")
stance = stances.STANCE_4


# env = gym.make('TorsoClimb-v0', render_mode='human', max_ep_steps=600, reward=Reward.NEGATIVE_DIST, motion_path=MOTION, state_file=STATEFILE)
env = gym.make('HumanoidClimb-v0',
               render_mode='human',
               max_ep_steps=10000000,
               **stance.get_args())

ob, info = env.reset(seed=42)

state = env.reset()
done = False
truncated = False
score = 0
step = 0
pause = False
hold = True

action = [0.0 for i in range(env.action_space.shape[0])]
action[-4] = 1
action[-3] = 1
action[-2] = 1
action[-1] = 1

while True:

    if not pause:
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        step += 1

    # Reset on backspace
    keys = p.getKeyboardEvents()

    # rarrow
    if 65296 in keys and keys[65296] & p.KEY_WAS_TRIGGERED:
        pass

    # r
    if 114 in keys and keys[114] & p.KEY_WAS_TRIGGERED:
        print(f"Score: {score}, Steps {step}")
        done = False
        truncated = False
        pause = False
        score = 0
        step = 0
        env.reset()

    # Pause on space
    if 32 in keys and keys[32] & p.KEY_WAS_TRIGGERED:
        pause = not pause
        print("Paused" if pause else "Unpaused")

    if done or truncated:
        pause = True

env.close()
