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
        self.ECM = Graph()
        self.p_clips = []
        self.a_clips = []
        # TODO revise action and percept types
        action = self.ECM.new_vertex_property("string")
        self.ECM.vertex_properties["action"] = action
        percept = self.ECM.new_vertex_property("double")
        self.ECM.vertex_properties["percept"] = percept

       ### Creating percepts-clips
        for p in percepts:
            percept_clip = self.ECM.add_vertex()
            percept[percept_clip] = p
            self.p_clips.append(percept_clip)

        ### Creating action-clips
        for a in actions:
            action_clip = self.ECM.add_vertex()
            action[action_clip] = a
            self.a_clips.append(action_clip)

        ### Creating edges between percepts-clips and action-clips
        for p in p_clips:
            for a in a_clips:
                self.ECM.add_edge(p,a)

        ### Initializing h-value
        h_value = self.ECM.new_edge_property("double")
        edges = self.ECM.get_edges()
        for e in edges:
            h_value[e] = 1
        self.ECM.edge_properties["h_value"] = h_value

    def random_walk(self, percept):
        for v in self.ECM.vertices():
            if self.ECM.vp.percept[v] == percept:
                hopping_clip = v
                break

        while self.ECM.vp.action[hopping_clip] == "":
            out_edges = hopping_clip.out_edges()

            total_value = 0.0
            for e in out_edges:
                total_value += self.ECM.ep.h_value[e]

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

