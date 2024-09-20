import pybullet as p
import os


class Target:
    def __init__(self, client, pos):
        f_name = os.path.join(os.path.dirname(__file__), 'target.xml')
        self.id = p.loadURDF(fileName=f_name,
                             basePosition=pos,
                             physicsClientId=client)