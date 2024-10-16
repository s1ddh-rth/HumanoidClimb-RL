
# Humanoid Climb: Reinforcement Learning for Climbing Humanoids

## Overview

Humanoid Climb is an advanced reinforcement learning project aimed at teaching a humanoid agent to climb vertical surfaces. The project specifically deals with dynamic movement (a 4 limb transition, also known as a dyno movement in Rock Climbing). This repository contains a custom OpenAI Gym environment implementation and training scripts utilizing Stable Baselines 3.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Environment Details](#environment-details)
- [Results](#results)
- [Contributing](#contributing)
- [Work In Progress](#work-in-progress)
- [Parent Repo](#parent-repo)

## Features

- Custom OpenAI Gym environment (`HumanoidClimbEnv`) for simulating humanoid climbing
- Integration with Stable Baselines 3 for state-of-the-art reinforcement learning algorithms
- Support for multiple RL algorithms (PPO, SAC)
- Customizable climbing scenarios with configurable target positions (a separate config file is being worked upon)
- Advanced reward shaping for efficient learning
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

The HumanoidClimbEnv class in humanoid_climb_env.py defines the custom Gym environment:

Action Space: 21-dimensional vector (gym.spaces.Box).

Reward Function: Combines multiple components-
- Distance to target holds
- Vertical velocity
- Body orientation (slouch angle)
- Wall impact penalty
- Floor contact penalty
- Stance completion bonus

## Results

The Humanoid was able to show signs of learning a dyno movement, though no complete transition was achieved.

Detailed metrics logged during training:

- Distance reward
- Velocity reward
- Slouch reward
- Wall impact penalty
- Floor contact reward
- Stance completion reward
- Total reward
## Contributing

We welcome contributions to the Humanoid Climb project. To contribute:
- Fork the repository
- Create a new branch for your feature or bug fix
- Commit your changes with clear, descriptive messages
- Push your branch and submit a pull request
- Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## WORK IN PROGRESS

https://github.com/dylanjoao/HumanoidClimbEnv/assets/64186394/d154f391-c658-49c2-9898-b1dba4da93fe
https://drive.google.com/file/d/18ITYeknRvYPDnZW0PLlvtKbo2EeDTHAG/view?usp=sharing

## Parent repo

https://github.com/dylanjoao/CS3IP

