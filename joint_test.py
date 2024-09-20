import pybullet_data
import pybullet as p
from pybullet_utils import bullet_client
from torso_climb.assets.torso import Torso

client = bullet_client.BulletClient(connection_mode=p.GUI)
physicsClientId = client._client

p.setGravity(0, 0, -9.8, physicsClientId=physicsClientId)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# SET REAL TIME SIMULATION FUCKS setJointMotorControl
# p.setRealTimeSimulation(1, physicsClientId)

plane = p.loadURDF("plane.urdf")
torso = Torso(physicsClientId, [0, 0, 1], [0, 0, 0, 1], fixedBase=True)

debug_params = []

for i in range(len(torso.ordered_joints)):
	joint = torso.ordered_joints[i]
	debug_params.append(p.addUserDebugParameter(torso.motor_names[i], -1, 1, 0.0, physicsClientId))


while True:
	actions = [p.readUserDebugParameter(i, physicsClientId) for i in debug_params]
	actions += (0.0, 0.0)
	torso.apply_action(actions)
	p.stepSimulation(physicsClientId)