import numpy as np
from graph_tool.all import *
import sys

if "../" not in sys.path:
    sys.path.append("../")

from lib.envs.simple_rooms import SimpleRoomsEnv
from lib.simulation import Experiment

class Agent(object):

    def __init__(self, actions):
        self.actions = actions
        self.num_actions = len(actions)

    def act(self, state):
        raise NotImplementedError

class ECM():

    def __init__(self, actions, percepts):
        ECM = Graph()
        p_clips = []
        a_clips = []

       ### Creating percepts-clips
        for p in percepts:
            percept_clip = ECM.add_vertex()
            p_clips.append(percept_clip)

        ### Creating action-clips
        for a in actions:
            action_clip = ECM.add_vertex()
            a_clips.append(action_clip)

        ### Creating edges between percepts-clips and action-clips
        for p in p_clips:
            for a in a_clips:
                ECM.add_edge(p,a)

    def random_walk(self, percept):
        pass

    def update(self):
        pass

    def composition(self):
        pass

    def delete_clip(self):
        pass

class PS_agent(Agent):

    def __init__(self, actions, eta, gamma, ECM):
        self.actions = actions
        self.num_actions = len(actions)
        self.eta = eta
        self.gamma = gamma
        self.ECM = ECM

    def act(self, percept):
        action = self.ECM.random_walk(percept)

    def learn(self, reward):
        self.ECM.update(reward)

        if reward > 0:
            self.ECM.composition()

        self.ECM.delete_clip()

