import numpy as np
import pybullet as p
import os
import math


class Pendulum:
    def __init__(self, client, pos):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'pendulum.urdf')
        self.id = p.loadURDF(fileName=f_name,
                             basePosition=pos,
                             physicsClientId=client)

        # Joint indices as found by p.getJointInfo()
        self.joints = [0, 1]

        # Joint speed
        self.top_momentum = 0

        # Target grasped
        self.top_held = None
        self.bottom_hold = None

        self.targets = []

    def get_ids(self):
        return self.id, self.client

    def apply_action(self, action):

        # 0 = +Vel (1)
        # 1 = -Vel (1)
        # 2 = Grab (1)
        # 3 = Release (1)

        # 4 = +Vel (2)
        # 5 = -Vel (2)
        # 6 = Grab (2)
        # 7 = Release (2)

        joint = None
        if action < 4:
            joint = 0
        elif action > 3:
            joint = 1

        if action == 0 or action == 4:
            p.setJointMotorControl2(bodyIndex=self.id,
                                    jointIndex=joint,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=150.0)

        elif action == 1 or action == 5:
            p.setJointMotorControl2(bodyIndex=self.id,
                                    jointIndex=joint,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=-150.0)

        elif (action == 2 and self.top_held is None) or (action == 6 and self.bottom_hold is None):
            in_range = self.target_in_range(joint)
            if in_range is not None:
                self.create_hold(joint, in_range.id)

        elif action == 3 or action == 7:
            self.remove_hold(joint)

    def target_in_range(self, joint):
        link_state = p.getLinkState(self.id, joint)
        target_in_range = None

        for i in range(len(self.targets)):
            target_pos, _ = p.getBasePositionAndOrientation(self.targets[i].id)
            distance = np.linalg.norm(np.array(link_state[0]) - np.array(target_pos))
            if distance < 0.1:
                target_in_range = self.targets[i]
                break

        return target_in_range

    def remove_hold(self, joint):

        if joint is 0 and self.top_held is not None:
            p.removeConstraint(self.top_held)
            self.top_held = None

        if joint is 1 and self.bottom_hold is not None:
            p.removeConstraint(self.bottom_hold)
            self.bottom_hold = None

    def create_hold(self, joint, target):
        self.remove_hold(joint)
        constraint = p.createConstraint(parentBodyUniqueId=self.id,
                                        parentLinkIndex=joint,
                                        childBodyUniqueId=target,
                                        childLinkIndex=-1,
                                        jointType=p.JOINT_POINT2POINT,
                                        jointAxis=[0, 0, 0],
                                        parentFramePosition=[0, 0, 0],
                                        childFramePosition=[0, 0, 0])

        if joint is 0:
            self.top_held = constraint
        if joint is 1:
            self.bottom_hold = constraint

    def get_observation(self):
        ###
        #   Agent centre position   (x y z)
        #   Agent velocity          (x y z)
        #   Agent angle             (x y z)
        #   Joint Held              (1, 1)

        # Get the position and orientation of the pendulum in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.id, self.client)
        ang = p.getEulerFromQuaternion(ang)

        # Get the velocity of the pendulum
        vel = p.getBaseVelocity(self.id, self.client)[0]

        holds = [0 if self.top_held is None else 1] + [0 if self.bottom_hold is None else 1]

        # Concatenate holds[2], position[3], angle[3], velocity[3]
        # ([1, 0], [0.0, 0.0, 1.5], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        observation = np.concatenate((holds, pos, ang, vel), dtype=np.float32)

        return observation
