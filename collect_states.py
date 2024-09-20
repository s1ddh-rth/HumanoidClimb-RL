import random

import pybullet as p
import stable_baselines3 as sb
import gymnasium as gym
import numpy as np
import time
import torso_climb
import humanoid_climb
import humanoid_climb.stances as stances
from torso_climb.env.torso_climb_env import Reward

NUM_SAMPLES = 1000

stances.set_root_path("./humanoid_climb")
STANCE = stances.STANCE_13_2
MODEL_FILE = "./humanoid_climb/models/13_18_20_10_9.zip"


def get_state(bodyIndex, pid):
	# pos[3], ori[4], (jointPos, jointVel), (jointPos, jointVel)...

	pos, ori = p.getBasePositionAndOrientation(bodyIndex, physicsClientId=pid)
	numJoints = p.getNumJoints(bodyIndex, physicsClientId=pid)
	_jointStates = p.getJointStates(bodyIndex, [n for n in range(numJoints)])
	jointStates = [s[:2] for s in _jointStates]

	final_state = []
	final_state += pos
	final_state += ori
	for s in jointStates:
		final_state += s
	final_state += STANCE.stance

	return np.array(final_state)

env = gym.make("HumanoidClimb-v0", max_ep_steps=600, **STANCE.get_args())
model = sb.PPO.load(MODEL_FILE, env=env)
obs = env.reset()[0]

start = time.perf_counter()
saved = 0
states = []
for i in range(NUM_SAMPLES):
	done = False
	truncated = False
	info = None
	while not done and not truncated:
		action, _state = model.predict(np.array(obs), deterministic=True)
		obs, reward, done, truncated, info = env.step(action)

	if info['is_success']:
		states.append(get_state(2, 0))
		saved += 1

	print(f"Collected {saved} states")
	env.reset()

env.close()
np.savez(r'./humanoid_climb/states/out.npz', states)
print(f"Saved {saved} samples in {time.perf_counter() - start} seconds with a {(saved/NUM_SAMPLES)*100}% success rate")
