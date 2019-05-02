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
        self.new_p_clips = []
        self.a_clips = []
        self.a_composition = []

        # Creating action properties
        action = self.ECM.new_vertex_property("object")
        self.ECM.vertex_properties["action"] = action
        action_size = self.ECM.new_vertex_property("int")
        self.ECM.vertex_properties["action_size"] = action_size
        immunity = self.ECM.new_vertex_property("int")
        self.ECM.vertex_properties["immunity"] = immunity

        # Creating percept properties
        percept = self.ECM.new_vertex_property("object")
        self.ECM.vertex_properties["percept"] = percept

        # Creating percepts-clips
        for p in percepts:
            percept_clip = self.ECM.add_vertex()
            percept[percept_clip] = p
            self.p_clips.append(percept_clip)

        # Creating action-clips
        for a in actions:
            action_clip = self.ECM.add_vertex()
            action[action_clip] = a
            action_size[action_clip] = 1
            immunity[action_clip] = 10
            self.a_clips.append(action_clip)

        # Creating edges between percepts-clips and action-clips
        for p in p_clips:
            for a in a_clips:
                self.ECM.add_edge(p,a)

        # Initializing edge properties h-value and glow
        h_value = self.ECM.new_edge_property("double")
        glow = self.ECM.new_edge_property("double")

        edges = self.ECM.get_edges()
        for e in edges:
            h_value[e] = 1
            glow[e] = 0

        self.ECM.edge_properties["h_value"] = h_value
        self.ECM.edge_properties["glow"] = glow

    def random_walk(self, percept):
        # Finding clip that matches percept
        for v in self.ECM.vertices():
            if self.ECM.vp.percept[v] == percept:
                hopping_clip = v
                break

        # Random Walk until find action-clip
        while self.ECM.vp.action[hopping_clip] == None:
            h_values = []
            sum_h_values = 0.0
            probabilities = []

            # Retrieving out edges from clip
            out_edges_list = self.ECM.get_out_edges(hopping_clip)
            out_edges = hopping_clip.out_edges()

            # Setting probabilitie of hop
            h_values = [self.ECM.ep.h_value[e] for e in out_edges]
            sum_h_values = sum(h_values)
            probabilities = [h/sum_h_values for h in h_values]

            # Hopping
            selected_edge = out_edges_list[np.random.choice(
                                                        len(out_edges_list),
                                                        1,
                                                        p=probabilities)]

            # Setting glow parameter to 1
            self.ECM.ep.glow[self.ECM.edge(
                                        selected_edge[0][0],
                                        selected_edge[0][1])] = 1

            # Setting clip for next iteration
            hopping_clip = self.ECM.vertex(selected_edge[0][1])

        self.a_composition.append(self.ECM.vp.action[hopping_clip])
        return self.ECM.vp.action[hopping_clip], self.ECM.vp.action_size[hopping_clip]

    def update(self, reward, gamma, eta):
        for e in self.ECM.edges():
            self.ECM.ep.h_value[e] = self.ECM.ep.h_value[e] - gamma * (self.ECM.ep.h_value[e] - 1) + (reward * self.ECM.ep.glow[e])
            self.ECM.ep.glow[e] = self.ECM.ep.glow[e] - (eta * self.ECM.ep.glow[e])

    def composition(self):
        action_composition = self.ECM.add_vertex()
        self.ECM.vp.action[action_composition] = self.a_composition
        self.ECM.vp.action_size[action_composition] = len(self.a_composition)
        self.ECM.vp.immunity[action_composition] = 10

        # Add edges to new action
        for p in self.ECM.p_clips():
            new_edge = self.ECM.add_edge(p, action_composition)
            self.ECM.ep.h_value[new_edge] = 1
            self.ECM.ep.glow[new_edge] = 0

    def add_percept(self):
        self.p_clips.extend(new_p_clip)
        self.new_p_clips = []

    def clip_deletion_percept(self):
        for n_p in self.new_p_clips:
            self.ECM.remove_vertex(n_p)
        self.update_clip_list()
        self.n_p_clips = []

    def clip_deletion_action(self):
        sum_h_values = []
        probabilities = []

        delete_actions = [a for a in a_clips if self.ECM.vp.immunity == 0]
        for d_a in delete_actions:
            sum_h = 0
            for i_e in a.in_edges():
                sum_h += self.ECM.ep.h_value[i_e]
            sum_h_values.append(sum_h)

        probabilities = [np.power(
                                len(self.p_clips)/sum_h_values[i],
                                len(self.p_clips))
                         for i,_ in enumerate(delete_actions)]

        deleted_action = delete_actions[np.random.choice(
                                                    len(delete_actions),
                                                    1,
                                                    p=probabilities)]

        self.ECM.remove_vertex(deleted_action)
        self.update_clip_list()

    def update_clip_list(self):
        self.a_clips = []
        self.p_clips = []

        for v in self.ECM.vertices():
            if self.ECM.vp.action != None:
                self.a_clips.append(v)
            elif self.ECM.vp.percept != None:
                self.p_clips.append(v)

class PS_agent(Agent):

    def __init__(self, actions, eta, gamma, ECM):
        self.actions = actions
        self.num_actions = len(actions)
        self.eta = eta
        self.gamma = gamma
        self.memory = ECM

    def act(self, percept):
        if percept not in self.memory.p_clips:
            new_percept = self.memory.ECM.add_vertex()
            self.memory.ECM.vp.percept[new_percept] = percept
            self.memory.new_p_clips.append(new_percept)
            for a in self.memory.a_clips:
                self.memory.ECM.add_edge(p,a)

        return self.ECM.random_walk(percept)

    def learn(self, reward):
        # Update h-values from edges
        self.memory.update(reward, self.gamma, self.eta)

        # If succesful create a new action-clip with last action sequence
        # Else clear circuit and start over
        if reward > 0:
            self.memory.add_percept()
            self.memory.composition()
            self.memory.clip_deletion_action()
        else:
            self.memory.clip_deletion_percept()

