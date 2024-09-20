import numpy as np
import pybullet_data
import pybullet as p
from pybullet_utils import bullet_client

from humanoid_climb.assets.humanoid import Humanoid
from humanoid_climb.assets.target import Target
from torso_climb.assets.torso import Torso


def collision_check(bullet_client, robot: Humanoid, target: Target):
    eff_pos = robot.parts["right_hand"].current_position()

    dist = np.linalg.norm(np.array(eff_pos) - np.array(target.pos))
    if dist < 0.1:
        bullet_client.addUserDebugLine(eff_pos, target.pos, [0, 1, 0], lifeTime=5.0)

_p = bullet_client.BulletClient(connection_mode=p.GUI)
_p.setGravity(0, 0, -9.8)
_p.setAdditionalSearchPath(pybullet_data.getDataPath())
_p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 240., numSubSteps=5)
_p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# plane = _p.loadURDF("plane.urdf")
robot = Humanoid(_p, [0, 0, 2], [0, 0, 0, 1], 0.48, fixedBase=True)
target = Target(_p, [0.5, 0, 2])
params = [_p.addUserDebugParameter(motor.joint_name, -1, +1) for motor in robot.motors]

robot.targets = [target]
for geom in robot.parts:
    _p.changeVisualShape(robot.robot, robot.parts[geom].bodyPartIndex, rgbaColor=[1, 1, 1, 1])
_p.changeVisualShape(target.id, -1, rgbaColor=[1, 0, 0, 0.0])

pause = False
while _p.isConnected():

    if not pause:
        actions = [_p.readUserDebugParameter(param) for param in params]
        actions += (1.0, 1.0, 0.0, 0.0)
        robot.apply_action(actions)
        _p.stepSimulation()
        # collision_check(_p, robot, target)

    keys = p.getKeyboardEvents()

    # Pause on space
    if 32 in keys and keys[32] & p.KEY_WAS_TRIGGERED:
        pause = not pause
        print("Paused" if pause else "Unpaused")

    # R
    if 114 in keys and keys[114] & p.KEY_WAS_TRIGGERED:
        robot.reset()
        _p.removeAllUserDebugItems()
        pause = True



_p.disconnect()
