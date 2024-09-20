import random
from typing import List

import numpy as np
import pybullet as p
import os

from humanoid_climb.assets.robot_util import *
from humanoid_climb.assets.target import Target


class Humanoid:

    def __init__(self, bullet_client, pos, ori, power, statefile=None, fixedBase=False):
        f_name = os.path.join(os.path.dirname(__file__), 'humanoid_symmetric.xml')

        self._p = bullet_client
        self.power = power

        self.robot = bullet_client.loadMJCF(f_name)[0]
        bullet_client.resetBasePositionAndOrientation(self.robot, pos, ori)
        if fixedBase:
            bullet_client.createConstraint(self.robot, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0, 1], pos)

        if statefile is not None:
            self.state_file = np.load(statefile)
        self.exclude_targets = []

        (self.parts, self.joints, self.ordered_joints, self.robot_body) = addToScene(bullet_client, [self.robot])

        self.motor_names = ["abdomen_z", "abdomen_y", "abdomen_x"]
        self.motor_power = [100, 100, 100]
        self.motor_names += ["right_hip_x", "right_hip_z", "right_hip_y", "right_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["left_hip_x", "left_hip_z", "left_hip_y", "left_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["right_shoulder1", "right_shoulder2", "right_elbow"]
        self.motor_power += [75, 75, 75]
        self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
        self.motor_power += [75, 75, 75]
        self.motors = [self.joints[n] for n in self.motor_names]

        self.LEFT_HAND = self.parts["left_hand"]
        self.RIGHT_HAND = self.parts["right_hand"]
        self.LEFT_FOOT = self.parts["left_foot"]
        self.RIGHT_FOOT = self.parts["right_foot"]
        self.effectors = [self.LEFT_HAND, self.RIGHT_HAND, self.LEFT_FOOT, self.RIGHT_FOOT]

        self.lh_cid = -1
        self.rh_cid = -1
        self.lf_cid = -1
        self.rf_cid = -1

        self.targets = None

    def set_targets(self, targets: List[Target]):
        self.targets = targets

    def apply_action(self, a, override=None):
        body_actions = a[0:17]
        grasp_actions = a[17:21]

        if override is not None:
            for i in range(len(override)):
                if override[i] != -1:
                    grasp_actions[i] = override[i]

        force_gain = 1
        for i, m, power in zip(range(17), self.motors, self.motor_power):
            m.set_motor_torque(float(force_gain * power * self.power * np.clip(body_actions[i], -1, +1)))

        for i, eff in enumerate(self.effectors):
            if grasp_actions[i] > 0:
                self.attach(eff)
            else:
                self.detach(eff)

    def attach(self, effector):
        if effector == self.LEFT_HAND and self.lh_cid != -1:
            return
        elif effector == self.RIGHT_HAND and self.rh_cid != -1:
            return
        elif effector == self.LEFT_FOOT and self.lf_cid != -1:
            return
        elif effector == self.RIGHT_FOOT and self.rf_cid != -1:
            return

        eff_pos = effector.current_position()
        for target in self.targets:
            dist = np.linalg.norm(np.array(eff_pos) - np.array(target.pos))
            if dist < 0.1:
                self.force_attach(limb_link=effector, target=target, force=1000, attach_pos=eff_pos)
                break

    def force_attach(self, limb_link, target, force=-1, attach_pos=None):
        if limb_link == self.LEFT_HAND and self.lh_cid != -1:
            self.detach(self.LEFT_HAND)
        elif limb_link == self.RIGHT_HAND and self.rh_cid != -1:
            self.detach(self.RIGHT_HAND)
        elif limb_link == self.LEFT_FOOT and self.lf_cid != -1:
            self.detach(self.LEFT_FOOT)
        elif limb_link == self.RIGHT_FOOT and self.rf_cid != -1:
            self.detach(self.RIGHT_FOOT)

        target_index = self.targets.index(target)
        if target_index in self.exclude_targets[self.effectors.index(limb_link)]:
            return

        local_pos = [0, 0, 0]
        if attach_pos is not None:
            direction = np.subtract(attach_pos, target.pos)
            norm = normalized(direction)
            offset_eff = attach_pos + np.multiply(norm, -0.04)[0]
            local_pos = offset_eff - target.pos
            # print(f"Direction {direction}\nDistance {distance}\nOffset {offset_eff}")
            # self._p.addUserDebugLine(offset_eff, target.pos, [1, 1, 0])

        constraint = self._p.createConstraint(parentBodyUniqueId=self.robot, parentLinkIndex=limb_link.bodyPartIndex,
                                              childBodyUniqueId=target.id, childLinkIndex=-1,
                                              jointType=p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
                                              parentFramePosition=[0, 0, 0], childFramePosition=local_pos)
        self._p.changeConstraint(userConstraintUniqueId=constraint, maxForce=force)

        if limb_link == self.LEFT_HAND: self.lh_cid = constraint
        if limb_link == self.RIGHT_HAND: self.rh_cid = constraint
        if limb_link == self.LEFT_FOOT: self.lf_cid = constraint
        if limb_link == self.RIGHT_FOOT: self.rf_cid = constraint

    def detach(self, limb_link):
        if limb_link == self.LEFT_HAND and self.lh_cid != -1:
            self._p.removeConstraint(userConstraintUniqueId=self.lh_cid)
            self.lh_cid = -1
        elif limb_link == self.RIGHT_HAND and self.rh_cid != -1:
            self._p.removeConstraint(userConstraintUniqueId=self.rh_cid)
            self.rh_cid = -1
        elif limb_link == self.LEFT_FOOT and self.lf_cid != -1:
            self._p.removeConstraint(userConstraintUniqueId=self.lf_cid)
            self.lf_cid = -1
        elif limb_link == self.RIGHT_FOOT and self.rf_cid != -1:
            self._p.removeConstraint(userConstraintUniqueId=self.rf_cid)
            self.rf_cid = -1

    def reset(self):
        self.detach(self.LEFT_HAND)
        self.detach(self.RIGHT_HAND)
        self.detach(self.LEFT_FOOT)
        self.detach(self.RIGHT_FOOT)

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
            self.force_attach(limb_link=eff, target=target, force=1000, attach_pos=eff.current_position())

    def initialise_from_state(self):
        upper = len(self.state_file['arr_0'])
        rand = random.randint(0, upper - 1)
        state = self.state_file['arr_0'][rand]
        self.set_state(state)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)
