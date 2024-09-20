import pybullet as p
import os


class Target:
    def __init__(self, client, base):
        f_name = os.path.join(os.path.dirname(__file__), 'target.urdf')
        self.id = p.loadURDF(fileName=f_name,
                             basePosition=base,
                             physicsClientId=client)
        self.constraint = None