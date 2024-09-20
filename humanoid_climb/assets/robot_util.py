import numpy as np
import pybullet as p

class Pose_Helper:
	def __init__(self, body_part):
		self.body_part = body_part

	def xyz(self):
		return self.body_part.current_position()

	def rpy(self):
		return p.getEulerFromQuaternion(self.body_part.current_orientation())

	def orientation(self):
		return self.body_part.current_orientation()


class BodyPart:
	def __init__(self, bullet_client, body_name, bodies, bodyIndex, bodyPartIndex):
		self.bodies = bodies
		self._p = bullet_client
		self.bodyIndex = bodyIndex
		self.bodyPartIndex = bodyPartIndex
		self.initialPosition = self.current_position()
		self.initialOrientation = self.current_orientation()
		self.bp_pose = Pose_Helper(self)

	def state_fields_of_pose_of(
			self, body_id, link_id=-1
	):  # a method you will most probably need a lot to get pose and orientation
		if link_id == -1:
			(x, y, z), (a, b, c, d) = self._p.getBasePositionAndOrientation(body_id)
		else:
			(x, y, z), (a, b, c, d), _, _, _, _ = self._p.getLinkState(body_id, link_id)
		return np.array([x, y, z, a, b, c, d])

	def get_position(self):
		return self.current_position()

	def get_pose(self):
		return self.state_fields_of_pose_of(self.bodies[self.bodyIndex], self.bodyPartIndex)

	def speed(self):
		if self.bodyPartIndex == -1:
			(vx, vy, vz), _ = self._p.getBaseVelocity(self.bodies[self.bodyIndex])
		else:
			(x, y, z), (a, b, c, d), _, _, _, _, (vx, vy, vz), (vr, vp, vy) = self._p.getLinkState(
				self.bodies[self.bodyIndex], self.bodyPartIndex, computeLinkVelocity=1
			)
		return np.array([vx, vy, vz])

	def current_position(self):
		return self.get_pose()[:3]

	def current_orientation(self):
		return self.get_pose()[3:]

	def get_orientation(self):
		return self.current_orientation()

	def reset_position(self, position):
		self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], position, self.get_orientation())

	def reset_orientation(self, orientation):
		self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], self.get_position(), orientation)

	def reset_velocity(self, linearVelocity=None, angularVelocity=None):
		if angularVelocity is None:
			angularVelocity = [0, 0, 0]
		if linearVelocity is None:
			linearVelocity = [0, 0, 0]
		self._p.resetBaseVelocity(self.bodies[self.bodyIndex], linearVelocity, angularVelocity)

	def reset_pose(self, position, orientation):
		self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], position, orientation)

	def pose(self):
		return self.bp_pose

	def contact_list(self):
		return self._p.getContactPoints(self.bodies[self.bodyIndex], -1, self.bodyPartIndex, -1)


class Joint:
	def __init__(self, bullet_client, joint_name, bodies, bodyIndex, jointIndex):
		self.bodies = bodies
		self._p = bullet_client
		self.bodyIndex = bodyIndex
		self.jointIndex = jointIndex
		self.joint_name = joint_name

		jointInfo = self._p.getJointInfo(self.bodies[self.bodyIndex], self.jointIndex)
		self.lowerLimit = jointInfo[8]
		self.upperLimit = jointInfo[9]

		self.power_coeff = 0

	def set_state(self, x, vx):
		self._p.resetJointState(self.bodies[self.bodyIndex], self.jointIndex, x, vx)

	def current_position(self):  # just some synonym method
		return self.get_state()

	def get_mid_point(self):
		return 0.5 * (self.lowerLimit + self.upperLimit)

	def current_relative_position(self):
		pos, vel = self.get_state()
		# Pos and vel rescaled to [-1, 1]
		# Note: limits seems to be wrong for cheetah
		scaled_pos = 2 * (pos - self.lowerLimit) / (self.upperLimit - self.lowerLimit) - 1.0
		return scaled_pos, 0.1 * vel

	def get_state(self):
		x, vx, _, _ = self._p.getJointState(self.bodies[self.bodyIndex], self.jointIndex)
		return x, vx

	def get_position(self):
		x, _ = self.get_state()
		return x

	def get_orientation(self):
		_, r = self.get_state()
		return r

	def get_velocity(self):
		_, vx = self.get_state()
		return vx

	def set_position(self, position):
		self._p.setJointMotorControl2(
			self.bodies[self.bodyIndex], self.jointIndex, p.POSITION_CONTROL, targetPosition=position
		)

	def set_velocity(self, velocity):
		self._p.setJointMotorControl2(
			self.bodies[self.bodyIndex], self.jointIndex, p.VELOCITY_CONTROL, targetVelocity=velocity
		)

	def set_motor_torque(self, torque):  # just some synonyme method
		self.set_torque(torque)

	def set_torque(self, torque):
		self._p.setJointMotorControl2(
			bodyIndex=self.bodies[self.bodyIndex],
			jointIndex=self.jointIndex,
			controlMode=p.TORQUE_CONTROL,
			force=torque,
		)  # , positionGain=0.1, velocityGain=0.1)

	def reset_current_position(self, position, velocity):  # just some synonyme method
		self.reset_position(position, velocity)

	def reset_position(self, position, velocity):
		self._p.resetJointState(self.bodies[self.bodyIndex], self.jointIndex, targetValue=position,
								targetVelocity=velocity)
		self.disable_motor()

	def disable_motor(self):
		self._p.setJointMotorControl2(
			self.bodies[self.bodyIndex],
			self.jointIndex,
			controlMode=p.POSITION_CONTROL,
			targetPosition=0,
			targetVelocity=0,
			positionGain=0.1,
			velocityGain=0.1,
			force=0,
		)


def addToScene(bullet_client, bodies):
	_p = bullet_client

	parts = {}
	joints = {}
	ordered_joints = []
	robot_body = None
	robot_name = None

	dump = 0
	for i in range(len(bodies)):
		if _p.getNumJoints(bodies[i]) == 0:
			part_name, robot_name = _p.getBodyInfo(bodies[i])
			robot_name = robot_name.decode("utf8")
			part_name = part_name.decode("utf8")
			parts[part_name] = BodyPart(_p, part_name, bodies, i, -1)
		for j in range(_p.getNumJoints(bodies[i])):
			_p.setJointMotorControl2(
				bodies[i], j, p.POSITION_CONTROL, positionGain=0.1, velocityGain=0.1, force=0
			)
			jointInfo = _p.getJointInfo(bodies[i], j)
			joint_name = jointInfo[1]
			part_name = jointInfo[12]

			joint_name = joint_name.decode("utf8")
			part_name = part_name.decode("utf8")

			if dump:
				print("ROBOT PART '%s'" % part_name)
			if dump:
				print(
					"ROBOT JOINT '%s'" % joint_name
				)  # limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((joint_name,) + j.limits()) )

			parts[part_name] = BodyPart(_p, part_name, bodies, i, j)

			if part_name == robot_name:
				robot_body = parts[part_name]

			if i == 0 and j == 0 and robot_body is None:  # if nothing else works, we take this as robot_body
				parts[robot_name] = BodyPart(_p, robot_name, bodies, 0, -1)
				robot_body = parts[robot_name]

			if joint_name[:6] == "ignore":
				Joint(_p, joint_name, bodies, i, j).disable_motor()
				continue

			if joint_name[:8] != "jointfix":
				joints[joint_name] = Joint(_p, joint_name, bodies, i, j)
				ordered_joints.append(joints[joint_name])

				joints[joint_name].power_coef = 100.0

	return parts, joints, ordered_joints, robot_body