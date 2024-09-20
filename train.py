import time
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation
import pybullet as p
import stable_baselines3 as sb
import os
import argparse

from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder, VecFrameStack
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback

from torso_climb.env.torso_climb_env import Reward
from typing import List

import wandb
from wandb.integration.sb3 import WandbCallback
import humanoid_climb.stances as stances

import pendulum_climb
import torso_climb
import humanoid_climb

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


class CustomCallback(BaseCallback):
	def __init__(self, verbose: int = 0):
		super().__init__(verbose)
		self.rollout_count = 0

	def _on_step(self) -> bool:
		return True

	def _on_rollout_end(self) -> None:
		self.rollout_count += 1
		_steps_till_success = []
		_success = []

		# for entry in self.model.ep_info_buffer:
		# 	_success.append(entry['is_success'])
		#
		# success_rate = np.mean(_success) if len(_success) > 0 else None
		#
		# self.logger.record("climb/success_rate", success_rate)
		self.logger.record("climb/rollout_count", self.rollout_count)


def make_env(env_id: str, rank: int, seed: int = 0, max_steps: int = 1000, stance: stances.Stance = stances.STANCE_NONE) -> gym.Env:
	def _init():
		env = gym.make(env_id, render_mode=None, max_ep_steps=max_steps, **stance.get_args())
		m_env = Monitor(env)
		m_env.reset(seed=seed + rank)
		return m_env

	set_random_seed(seed)
	return _init


def train(env_name, sb3_algo, workers, path_to_model=None):
	config = {
		"policy_type": "MlpPolicy",
		"total_timesteps": 50000000,
		"env_name": env_name,
	}
	run = wandb.init(
		project="HumanoidClimb-2",
		config=config,
		sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
		monitor_gym=False,  # auto-upload the videos of agents playing the game
		save_code=False,  # optional
		# id="6sbyjyfr"
	)

	max_ep_steps = 600
	stances.set_root_path("./humanoid_climb")
	stance = stances.STANCE_14_1
	vec_env = SubprocVecEnv([make_env(env_name, i, max_steps=max_ep_steps, stance=stance) for i in range(workers)], start_method="spawn")

	model = None
	save_path = f"{model_dir}/{run.id}"

	eval_callback = EvalCallback(vec_env, best_model_save_path=f"{save_path}/models/", log_path=f"{save_path}/logs/", eval_freq=500, deterministic=True, render=False)
	cust_callback = CustomCallback()

	if sb3_algo == 'PPO':
		if path_to_model is None:
			model = sb.PPO('MlpPolicy', vec_env, verbose=1, device='cuda', tensorboard_log=log_dir)
		else:
			model = sb.PPO.load(path_to_model, env=vec_env)
	elif sb3_algo == 'SAC':
		if path_to_model is None:
			model = sb.SAC('MlpPolicy', vec_env, verbose=1, device='cuda', tensorboard_log=log_dir)
		else:
			model = sb.SAC.load(path_to_model, env=vec_env)
	else:
		print('Algorithm not found')
		return

	model.learn(
		total_timesteps=config["total_timesteps"],
		progress_bar=True,
		callback=[WandbCallback(
			gradient_save_freq=5000,
			model_save_freq=5000,
			model_save_path=save_path,
			verbose=2,
		), eval_callback, cust_callback],
	)
	run.finish()


def test(env, sb3_algo, path_to_model):
	if sb3_algo == 'SAC':
		model = sb.SAC.load(path_to_model, env=env)
	elif sb3_algo == 'TD3':
		model = sb.TD3.load(path_to_model, env=env)
	elif sb3_algo == 'A2C':
		model = sb.A2C.load(path_to_model, env=env)
	elif sb3_algo == 'DQN':
		model = sb.DQN.load(path_to_model, env=env)
	elif sb3_algo == 'PPO':
		model = sb.PPO.load(path_to_model, env=env)
	else:
		print('Algorithm not found')
		return

	vec_env = model.get_env()
	obs = vec_env.reset()
	score = 0
	step = 0

	while True:
		action, _state = model.predict(obs, deterministic=True)
		obs, reward, done, info = vec_env.step(action)
		score += reward
		step += 1

		# env.reset() auto called on vec_env?
		if done:
			print(f"Episode Over, Score: {score}, Steps {step}")
			score = 0
			step = 0

		# Reset on backspace
		keys = p.getKeyboardEvents()
		if 114 in keys and keys[114] & p.KEY_WAS_TRIGGERED:
			score = 0
			step = 0
			env.reset()

	env.close()


if __name__ == '__main__':

	# Parse command line inputs
	parser = argparse.ArgumentParser(description='Train or test model.')
	parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
	parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
	parser.add_argument('-w', '--workers', type=int)
	parser.add_argument('-t', '--train', action='store_true')
	parser.add_argument('-f', '--file', required=False, default=None)
	parser.add_argument('-s', '--test', metavar='path_to_model')
	args = parser.parse_args()

	if args.train:
		if args.file is None:
			print(f'<< Training from scratch! >>')
			train(args.gymenv, args.sb3_algo, args.workers)
		elif os.path.isfile(args.file):
			print(f'<< Continuing {args.file} >>')
			train(args.gymenv, args.sb3_algo, args.workers, args.file)

	if args.test:
		if os.path.isfile(args.test):
			stances.set_root_path("./humanoid_climb")
			stance = stances.STANCE_14_1
			max_steps = 600

			env = gym.make(args.gymenv, render_mode='human', max_ep_steps=max_steps, **stance.get_args())
			test(env, args.sb3_algo, path_to_model=args.test)
		else:
			print(f'{args.test} not found.')
