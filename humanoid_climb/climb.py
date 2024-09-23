import os
import random

import humanoid_climb
import gymnasium as gym
import pybullet as p
import time

import humanoid_climb.stances as stances
from stable_baselines3 import PPO, SAC

stances.set_root_path("./")
STANCES = [stances.STANCE_1, stances.STANCE_2, stances.STANCE_3, stances.STANCE_4]

MOTION = [s.stance for s in STANCES]
EXCLUDE = [s.exclude_targets for s in STANCES]
O_ACTION = [s.action_override for s in STANCES]

env = gym.make('HumanoidClimb-v0', render_mode='human', max_ep_steps=50000, motion_path=MOTION, state_file=None, motion_exclude_targets=EXCLUDE)
obs, info = env.reset()

state = env.reset()
done = False
truncated = False
score = 0
step = 0
pause = False

STANCE_TOLERANCE = 700
ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = ["/models/1_10_9_n_n.zip",
              "/models/2_10_9_2_n.zip",
              "/models/3_10_9_2_1.zip",
              "/models/4_14_13_n_n_new.zip"]


MODELS = [PPO.load(ROOT+MODEL_PATH[i], env=env) for i in range(len(MODEL_PATH))]
CUR_MODEL = 0
REWARDS = [0 for i in range(len(MODELS))]
STEPS = [0 for i in range(len(MODELS))]

last_completed_stance = None
climb_attempts = 0
successful_attempts = 0

while True:
    # action = env.action_space.sample()

    if not pause:
        action, _state = MODELS[CUR_MODEL].predict(obs, deterministic=True)

        for i in range(4):
            if O_ACTION[CUR_MODEL][i] != -1:
                action[17+i] = O_ACTION[CUR_MODEL][i]

        obs, reward, done, truncated, info = env.step(action)
        score += reward
        step += 1

        REWARDS[CUR_MODEL] += reward
        STEPS[CUR_MODEL] += 1

    if STEPS[CUR_MODEL] > STANCE_TOLERANCE:
        truncated = True

    # Reset on backspace
    keys = p.getKeyboardEvents()

    if 114 in keys and keys[114] & p.KEY_WAS_TRIGGERED:
        print(f"Score: {score}, Steps {step}")
        CUR_MODEL = 0
        REWARDS = [0 for i in range(len(MODELS))]
        STEPS = [0 for i in range(len(MODELS))]
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

    if info["is_success"]:
        print(f"Finished stance {CUR_MODEL+1} with {REWARDS[CUR_MODEL]} ({REWARDS[CUR_MODEL] - 1000}) reward in {STEPS[CUR_MODEL]} steps")
        CUR_MODEL += 1
        if CUR_MODEL > len(MODELS) - 1:
            CUR_MODEL = 0

    if done or truncated:
        CUR_MODEL = 0
        REWARDS = [0 for i in range(len(MODELS))]
        STEPS = [0 for i in range(len(MODELS))]
        env.reset()

        climb_attempts += 1
        if info["is_success"]: successful_attempts += 1

        print(f"ENV TERMINATED SUCCESS RATE {successful_attempts/climb_attempts*100} %\n")

    time.sleep(1/240)

env.close()
