import random
from typing import List

import numpy as np
import pybullet as p
import os

from humanoid_climb.assets.robot_util import *


class Humanoid:

    def __init__(self, bullet_client, config, fixedBase=False):
        f_name = os.path.join(os.path.dirname(__file__), 'humanoid_symmetric.xml')

        power = config['power']
        position = config['position']
        orientation = config['orientation']

        self._p = bullet_client
        self.power = power

        # TODO: make dynamic
        self.robot = bullet_client.loadMJCF(f_name, flags=p.URDF_USE_SELF_COLLISION)[0]
        bullet_client.resetBasePositionAndOrientation(self.robot, position, orientation)
        if fixedBase:
            self.base_constraint = bullet_client.createConstraint(self.robot, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0],
                                                                  [0, 0, 0, 1], position)

        self.parts, self.joints, self.ordered_joints, self.robot_body = addToScene(bullet_client, [self.robot])

        self.motor_names = [k for k in config['joint_forces']]
        self.motor_power = [config['joint_forces'][k] for k in config['joint_forces']]
        self.motors = [self.joints[n] for n in self.motor_names]

        self.effectors = [self.parts[k] for k in config['end_effectors']]
        self.effector_attached_to = [-1 for k in config['end_effectors']]
        self.effector_constraints = [-1 for i in range(len(self.effectors))]

        collision_groups = config['collision_groups']

        # Change colour and set collision groups
        for geom in self.parts:
            self._p.changeVisualShape(self.robot, self.parts[geom].bodyPartIndex, rgbaColor=[0, 0, 1, 1])
            if geom in collision_groups:
                self._p.setCollisionFilterGroupMask(self.robot, self.parts[geom].bodyPartIndex,
                                                    collision_groups[geom][0],
                                                    collision_groups[geom][1])
                # print(f"{geom} set to group {collision_groups[geom][0]} & mask {collision_groups[geom][1]}")

        self.targets = None
        self.exclude_targets = []

    def apply_action(self, a, override=None):
        # Actions are split into joint actions and grasping actions
        # Grasping actions are the last x elements of the array, where x is the number of end-effectors
        body_actions = a[0:len(self.motors)]
        grasp_actions = a[-len(self.effectors):]

        if override is not None:
            for i in range(len(override)):
                if override[i] != None:
                    grasp_actions[i] = override[i]

        force_gain = 1
        for i, m, power in zip(range(17), self.motors, self.motor_power):
            m.set_motor_torque(float(force_gain * power * self.power * np.clip(body_actions[i], -1, +1)))

        for eff_index in range(len(self.effectors)):
            if grasp_actions[eff_index] > 0:
                self.attach(eff_index)
            else:
                self.detach(eff_index)

    def attach(self, eff_index):
        if self.effector_constraints[eff_index] != -1:
            return

        effector = self.effectors[eff_index]
        effector_pos = effector.current_position()
        for key in self.targets:
            target = self.targets[key]
            cp = self._p.getClosestPoints(target.id, self.robot, 1.0, -1, effector.bodyPartIndex)
            if len(cp) < 1:
                continue
            contact_distance = cp[0][8]

            if contact_distance < 0.0:
                self.force_attach(eff_index=eff_index, target_key=key, force=5000, attach_pos=effector_pos)
                break

            # dist = np.linalg.norm(np.array(eff_pos) - np.array(target.pos))
            # if dist < 0.1:
            #     self.force_attach(limb_link=effector, target=target, force=1000, attach_pos=eff_pos)
            #     break

    def force_attach(self, eff_index, target_key, force=-1, attach_pos=None):
        constraint = self.effector_constraints[eff_index]
        if constraint != -1:  # if already attached, de-attach
            self.detach(eff_index)

        target = self.targets[target_key]
        exclude_list = self.exclude_targets[eff_index]
        if len(exclude_list) > 0:
            if target_key in exclude_list:
                return

        if attach_pos is None:
            attach_pos = [0, 0, 0]

        constraint = self._p.createConstraint(parentBodyUniqueId=self.robot,
                                              parentLinkIndex=self.effectors[eff_index].bodyPartIndex,
                                              childBodyUniqueId=target.id, childLinkIndex=-1,
                                              jointType=p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
                                              parentFramePosition=[0, 0, 0],
                                              childFramePosition=np.subtract(attach_pos, target.body.initialPosition))
        self._p.changeConstraint(userConstraintUniqueId=constraint, maxForce=force)

        self.effector_attached_to[eff_index] = target_key
        self.effector_constraints[eff_index] = constraint

    def detach(self, eff_index):
        constraint = self.effector_constraints[eff_index]
        if constraint == -1:
            return

        self._p.removeConstraint(userConstraintUniqueId=constraint)
        self.effector_attached_to[eff_index] = -1
        self.effector_constraints[eff_index] = -1

    def reset(self):
        for eff_index in range(len(self.effectors)):
            self.detach(eff_index)

        # TODO: pose?
        self.robot_body.reset_pose(self.robot_body.initialPosition, self.robot_body.initialOrientation)
        for joint in self.joints:
            self.joints[joint].reset_position(0, 0)

    def set_state(self, state):
        pos = state[0:3]
        ori = state[3:7]
        stance = state[-4:]
        numJoints = self._p.getNumJoints(self.robot)
        joints = [state[(i * 2) + 7:(i * 2) + 9] for i in range(numJoints)]

        self._p.resetBasePositionAndOrientation(self.robot, pos, ori)
        for joint in range(numJoints):
            self._p.resetJointState(self.robot, joint, joints[joint][0], joints[joint][1])

        for i, eff in enumerate(self.effectors):
            if stance[i] == -1: continue
            target = self.targets[stance[i].astype(int)]
            self.force_attach(eff_index=eff, target=target, force=1000, attach_pos=eff.current_position())

    def initialise_from_state(self):
        upper = len(self.state_file['arr_0'])
        rand = random.randint(0, upper - 1)
        state = self.state_file['arr_0'][rand]
        self.set_state(state)
    def speed(self):
        linear_velocity, _ = self._p.getBaseVelocity(self.robot)
        return linear_velocity
    def get_orientation(self):
        _, orientation = self._p.getBasePositionAndOrientation(self.robot)
        return p.getEulerFromQuaternion(orientation)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)



