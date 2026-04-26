
# Using Reinforcement Learning to Plan Motion for a Rock Climber: Enhancing Limb Coordination for Complex Multi-limb Maneuvers

## Overview

Humanoid Climb is an advanced reinforcement learning project aimed at teaching a humanoid agent to climb vertical surfaces. The project specifically deals with dynamic movement (a 4 limb transition, also known as a dyno movement in Rock Climbing). This repository contains a custom OpenAI Gym environment implementation and training scripts utilizing Stable Baselines 3.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Environment Details](#environment-details)
- [Results](#results)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [Work In Progress](#work-in-progress)
- [Parent Repo](#parent-repo)

## Features

- Custom OpenAI Gym environment (`HumanoidClimbEnv`) for simulating humanoid climbing
- Integration with Stable Baselines 3 for state-of-the-art reinforcement learning algorithms
- Support for multiple RL algorithms (PPO, SAC)
- Customizable climbing scenarios via `config.json` (`ClimbingConfig`): hold positions, stance path, joint forces, collision groups
- Multiple reward functions available; active one is selected in `HumanoidClimbEnv.step()`
- Integration with Weights & Biases for comprehensive experiment tracking

## Installation

```cmd
conda create -n climb python=3.10
conda activate climb
conda install numpy pybullet gymnasium stable-baselines3 wandb --channel conda-forge
pip install stable-baselines3[extra]
```

## Usage

To train a new model, use the following command:

```bash
python train.py <env_name> <algorithm> -w <num_workers> -t
```

Example:
```bash
python train.py HumanoidClimb-v0 PPO -w 8 -t
```

Parameters:

- env_name: The Gymnasium environment ID (e.g., HumanoidClimb-v0)
- algorithm: The Stable Baselines 3 algorithm to use (PPO or SAC)
- -w: Number of parallel workers for training
- -t: Flag to indicate training mode

To test a new model, use the following command:

```bash
python train.py <env_name> <algorithm> -s <path_to_model>
```

Example:
```bash
python train.py HumanoidClimb-v0 PPO -s models/best_model.zip
```

## Environment Details

The `HumanoidClimbEnv` class in `humanoid_climb/env/humanoid_climb_env.py` defines the custom Gym environment.

**Observation:** 306-d `Box` (per-joint world pose / velocity, per-effector target distance, current/desired stance, contact flags).

**Action space:** selectable via the `discrete_grasp` kwarg.
- `discrete_grasp=False` (default): legacy `Box(-1, 1, (21,))` — 17 joint torques + 4 thresholded grasp signals.
- `discrete_grasp=True`: `MultiDiscrete([21]*17 + [2]*4)` — 17 binned torques (0.1 step) + 4 binary grasp dims with their own categorical heads. This was added to fix the dissertation-era problem where PPO never learned grasp timing because the grasp dim shared one Gaussian with the torque dims (see `CLAUDE.md`).

**Reward shaping:** the env exposes several reward functions; the active one is the call site inside `step()`. Currently `calculate_reward_negative_distance` — sum-of-distances to target holds + a floor-contact penalty. Alternate static methods (`calculate_improved_reward`, `calculate_reward_eq1`) add vertical velocity, slouch posture, and stance-completion bonuses, and are referenced (with the corresponding equation numbers) in the dissertation. Optional event-based grasp shaping is available via the `grasp_reward=True` kwarg, which adds attach / wrong-attach / waste / premature-release credit on the grasp dim. A `grasp_persist_steps=N` kwarg locks each grasp dim's binary intent for N steps after a flip (frameskip-style).

## Results

Across the five experimental runs in the dissertation, the agent showed signs of learning to leap (rising mean reward, longer episodes) but the success rate for a complete dyno transition remained ≈0%. The recurring failure mode in the most refined run was the agent briefly touching the floor mid-leap to bounce off and gain upward momentum — see `CLAUDE.md` for the per-run breakdown.

## Visualization

```bash
python -m humanoid_climb.climb
```

## Contributing

We welcome contributions to the Humanoid Climb project. To contribute:
- Fork the repository
- Create a new branch for your feature or bug fix
- Commit your changes with clear, descriptive messages
- Push your branch and submit a pull request
- Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## WORK IN PROGRESS

Humanoid trying a one stance transition (dyno movement)
https://drive.google.com/file/d/18ITYeknRvYPDnZW0PLlvtKbo2EeDTHAG/view?usp=sharing

## Parent repo

https://github.com/dylanjoao/CS3IP

