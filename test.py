import math
import time
from time import sleep
import pybullet as p
import pybullet_data
import numpy as np
import os

from pybullet_utils.bullet_client import BulletClient

from humanoid_climb.assets.humanoid import Humanoid
from pendulum_climb.assets.pendulum import Pendulum
import humanoid_climb.stances as stance

if __name__ == "__main__":

	current_directory = os.getcwd()

	stance.set_root_path(current_directory)

	# Can alternatively pass in p.DIRECT
	bullet_client = BulletClient(connection_mode=p.GUI)
	bullet_client.setGravity(0, 0, -9.8)
	bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
	# p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 240., numSolverIterations=100, numSubSteps=10)
	p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
	# bullet_client.setRealTimeSimulation(1)

	# flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS | p.URDF_GOOGLEY_UNDEFINED_COLORS
	flags = p.URDF_GOOGLEY_UNDEFINED_COLORS
	plane = bullet_client.loadURDF("plane.urdf")

	humanoid = Humanoid(bullet_client, [0, 0, 2], [0, 0, 0, 1], 0.48, None, True)
	params = [bullet_client.addUserDebugParameter(motor.joint_name, -1, +1) for motor in humanoid.motors]

	while bullet_client.isConnected():
		actions = [bullet_client.readUserDebugParameter(param) for param in params]
		actions += (0.0, 0.0, 0.0, 0.0)
		humanoid.apply_action(actions)
		bullet_client.stepSimulation()


	bullet_client.disconnect()
