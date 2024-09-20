import os
import pybullet as p


class Wall:
    def __init__(self, client, pos):
        f_name = os.path.join(os.path.dirname(__file__), 'wall.xml')

        self.client = client
        self.id = client.loadURDF(fileName=f_name, basePosition=pos)
