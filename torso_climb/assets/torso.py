import random

import numpy as np
import pybullet as p
import os
import math


# Reference https://www.gymlibrary.dev/environments/mujoco/humanoid/
class Torso:
	def __init__(self, client, pos, ori, statefile=None, fixedBase=False):
		f_name = os.path.join(os.path.dirname(__file__), 'pyb_torso.xml')

		self.client = client
		self.id = p.loadURDF(fileName=f_name, flags=p.URDF_MAINTAIN_LINK_ORDER + p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS, basePosition=pos, baseOrientation=ori,
							 globalScaling=1.0, useFixedBase=fixedBase, physicsClientId=self.client)

		# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/tensorflow/humanoid_running.py#L35
		self.human = self.id
		self.base_pos = []
		self.base_ori = []
		self.ordered_joints = []
		self.ordered_joint_indices = []

		self.RIGHT_HAND = -1
		self.LEFT_HAND = -1
		self.rhand_cid = -1
		self.lhand_cid = -1

		self.base_pos, self.base_ori = p.getBasePositionAndOrientation(bodyUniqueId=self.human, physicsClientId=self.client)

		if statefile is not None:
			self.state_file = np.load(statefile)

		jdict = {}
		for j in range(p.getNumJoints(self.human, physicsClientId=client)):
			info = p.getJointInfo(self.human, j, physicsClientId=client)
			link_name = info[12].decode("ascii")
			if link_name == "left_hand_tip": self.LEFT_HAND = j
			if link_name == "right_hand_tip": self.RIGHT_HAND = j
			self.ordered_joint_indices.append(j)

			jname = info[1].decode("ascii")
			jdict[jname] = j
			lower, upper = (info[8], info[9])

			if info[2] == p.JOINT_SPHERICAL:
				targetPosition = [0, 0, 0, 1]
				p.setJointMotorControlMultiDof(self.human, j, p.POSITION_CONTROL, targetPosition, targetVelocity=[0, 0, 0], positionGain=0, velocityGain=1, force=[0, 0, 0],
											   physicsClientId=client)
			elif info[2] == p.JOINT_REVOLUTE:
				self.ordered_joints.append((j, lower, upper))
				p.setJointMotorControl2(self.human, j, controlMode=p.VELOCITY_CONTROL, force=0, targetVelocity=0, physicsClientId=client)

		self.motor_names = ["right_shoulder1", "right_shoulder2", "right_elbow"]
		self.motor_power = [75, 75, 75]
		self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
		self.motor_power += [75, 75, 75]
		self.motors = [jdict[n] for n in self.motor_names]

	def apply_action(self, actions):
		body_actions = actions[0:6]
		grasp_actions = actions[6:8]

		forces = [0.] * len(self.motors)
		for m in range(len(self.motors)):
			limit = 15
			ac = np.clip(body_actions[m], -limit, limit)
			forces[m] = self.motor_power[m] * ac
		p.setJointMotorControlArray(self.human, self.motors, controlMode=p.TORQUE_CONTROL, forces=forces)

		# Left hand value
		if grasp_actions[0] > 0.5:
			self.attach(self.LEFT_HAND)
		else:
			self.detach(self.LEFT_HAND)

		# Right hand value
		if grasp_actions[1] > 0.5:
			self.attach(self.RIGHT_HAND)
		else:
			self.detach(self.RIGHT_HAND)

	def force_attach(self, limb_link, target_id, force=-1):
		if limb_link == self.LEFT_HAND and self.lhand_cid != -1: self.detach(self.LEFT_HAND)
		if limb_link == self.RIGHT_HAND and self.rhand_cid != -1: self.detach(self.RIGHT_HAND)

		constraint = p.createConstraint(parentBodyUniqueId=self.human,
										parentLinkIndex=limb_link,
										childBodyUniqueId=target_id,
										childLinkIndex=-1,
										jointType=p.JOINT_POINT2POINT,
										jointAxis=[0, 90, 0],
										parentFramePosition=[0, 0, 0],
										childFramePosition=[0, 0, 0],
										physicsClientId=self.client)
		p.changeConstraint(userConstraintUniqueId=constraint, maxForce=force, physicsClientId=self.client)

		if limb_link == self.LEFT_HAND:
			self.lhand_cid = constraint
		else:
			self.rhand_cid = constraint

	# Attach to the closest target
	def attach(self, limb_link):
		# If already attached return
		if limb_link == self.LEFT_HAND and self.lhand_cid != -1: return
		if limb_link == self.RIGHT_HAND and self.rhand_cid != -1: return

		body_count = p.getNumBodies(physicsClientId=self.client)
		limb_pos = p.getLinkState(bodyUniqueId=self.human, linkIndex=limb_link, physicsClientId=self.client)[0]

		for body_id in range(body_count):
			body_info = p.getBodyInfo(body_id, physicsClientId=self.client)
			body_name = body_info[0].decode("utf-8")

			if body_name == "target":
				target_pos, _ = p.getBasePositionAndOrientation(bodyUniqueId=body_id, physicsClientId=self.client)
				dist = np.linalg.norm(np.array(limb_pos) - np.array(target_pos))
				if dist < 0.06:
					self.force_attach(limb_link=limb_link, target_id=body_id, force=1000)
					break

	def detach(self, limb_link):
		if limb_link == self.LEFT_HAND and self.lhand_cid != -1:
			p.removeConstraint(userConstraintUniqueId=self.lhand_cid, physicsClientId=self.client)
			self.lhand_cid = -1
		elif limb_link == self.RIGHT_HAND and self.rhand_cid != -1:
			p.removeConstraint(userConstraintUniqueId=self.rhand_cid, physicsClientId=self.client)
			self.rhand_cid = -1

	def reset_state(self):
		self.detach(self.RIGHT_HAND)
		self.detach(self.LEFT_HAND)
		p.resetBasePositionAndOrientation(bodyUniqueId=self.human, posObj=self.base_pos, ornObj=self.base_ori, physicsClientId=self.client)
		for i, v in enumerate(self.ordered_joint_indices):
			p.resetJointState(self.human, v, targetValue=0.0, targetVelocity=0.0, physicsClientId=self.client)

	def set_state(self, state):
		pos = state[0:3]
		ori = state[3:7]
		numJoints = p.getNumJoints(self.human, physicsClientId=self.client)
		joints = [state[(i * 2) + 7:(i * 2) + 9] for i in range(numJoints)]

		p.resetBasePositionAndOrientation(self.human, pos, ori, physicsClientId=self.client)
		for joint in range(numJoints):
			p.resetJointState(self.human, joint, joints[joint][0], joints[joint][1], physicsClientId=self.client)

	def initialise_random_state(self):
		upper = len(self.state_file['arr_0'])
		rand = random.randint(0, upper-1)
		state = self.state_file['arr_0'][rand]
		self.set_state(state)