import numpy as np
from graph_tool.all import *

class Agent(object):

    def __init__(self, actions):
        self.actions = actions
        self.num_actions = len(actions)

    def act(self, state):
        raise NotImplementedError

class ECM():

    def __init__(self, actions, percepts):
        self.ECM = Graph()         # Network of clips that represents the memory
        self.p_clips = []          # List of active percept clips
        self.new_p_clips = []      # List of new percept clips that might be
                                   # removed
        self.a_clips = []          # List of action clips
        self.a_composition = []    # List of actions that might become a new
                                   # compose action

        # Initializing action properties
        action = self.ECM.new_vertex_property("object")
        action_size = self.ECM.new_vertex_property("int")
        immunity = self.ECM.new_vertex_property("int")

        # Initializing percept properties
        percept = self.ECM.new_vertex_property("object")

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
        for p in self.p_clips:
            for a in self.a_clips:
                self.ECM.add_edge(p,a)

        # Initializing edge properties h_value and glow
        h_value = self.ECM.new_edge_property("double")
        glow = self.ECM.new_edge_property("double")

        # Setting initial values for h_value and glow
        edges = self.ECM.get_edges()
        for e in edges:
            h_value[e] = 1
            glow[e] = 0

        # Adding properties to Graph
        self.ECM.vertex_properties["action"] = action
        self.ECM.vertex_properties["action_size"] = action_size
        self.ECM.vertex_properties["immunity"] = immunity
        self.ECM.vertex_properties["percept"] = percept

        self.ECM.edge_properties["h_value"] = h_value
        self.ECM.edge_properties["glow"] = glow

    def random_walk(self, percept):
        # Finding clip that matches percept
        for v in self.ECM.vertices():
            # Ad-hoc solution for simple room
            if np.array_equal(self.ECM.vp.percept[v], percept):
                hopping_clip = v
                break

        # Random Walk until find action-clip
        while self.ECM.vp.action[hopping_clip] == None:
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

        # Add selected action to future action composition
        self.a_composition.append(self.ECM.vp.action[hopping_clip])

        return self.ECM.vp.action[hopping_clip]#, self.ECM.vp.action_size[hopping_clip]

    def update(self, reward, gamma, eta):
        for e in self.ECM.edges():
            self.ECM.ep.h_value[e] = max(1, self.ECM.ep.h_value[e] - gamma * (self.ECM.ep.h_value[e] - 1) + (reward * self.ECM.ep.glow[e]))
            self.ECM.ep.glow[e] = max(0, self.ECM.ep.glow[e] - (eta * self.ECM.ep.glow[e]))

# TODO Check again later how to do composition
    def composition(self):
        # Initializing new action clip
        action_composition = self.ECM.add_vertex()

        self.ECM.vp.action[action_composition] = self.a_composition
        self.ECM.vp.action_size[action_composition] = len(self.a_composition)

        # Setting new action clip properties
        self.ECM.vp.immunity[action_composition] = 10

        # Adding edges to new action clip
        for p in self.ECM.p_clips():
            new_edge = self.ECM.add_edge(p, action_composition)
            self.ECM.ep.h_value[new_edge] = 1
            self.ECM.ep.glow[new_edge] = 0

        # Resetting a_compositon
        self.a_composition = []

    def add_percept(self):
        self.p_clips.extend(self.new_p_clips)
        self.new_p_clips = []

    def print_index(self, vertices):
        indexes = [self.ECM.vertex_index[v] for v in vertices]
        return indexes

    def clip_deletion_percept(self):
        for n_p in reversed(sorted(self.new_p_clips)):
            self.ECM.remove_vertex(n_p)
        self.update_clip_list()
        self.new_p_clips = []

    def clip_deletion_action(self):
        delete_actions = [a for a in a_clips if self.ECM.vp.immunity == 0]

        list_sum_h_values = [sum(self.ECM.ep.h_value[i_e] for i_e in d_a.in_edges()) for d_a in delete_actions]

        probabilities = [np.power(len(self.p_clips)/sum_h,
                                  len(self.p_clips))
                         for sum_h in list_sum_h_values]

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
            if self.ECM.vp.action[v] is None:
                self.p_clips.append(v)
            if self.ECM.vp.percept[v] is None:
                self.a_clips.append(v)

class PS_agent(Agent):

    def __init__(self, actions, percepts, eta, gamma, composition_active=False):
        self.actions = actions
        self.num_actions = len(actions)
        self.eta = eta
        self.gamma = gamma
        self.memory = ECM(actions, percepts)
        self.composition_active = composition_active

    def act(self, percept):
        for p in self.memory.p_clips:
            # Ad-hoc solution for simple room
            if np.array_equal(self.memory.ECM.vp.percept[p], percept):
                #print("Percept already in ECM", percept)
                return self.memory.random_walk(percept)

        for new_p in self.memory.new_p_clips:
            if np.array_equal(self.memory.ECM.vp.percept[new_p], percept):
                #print("Percept already in New Clips", percept)
                return self.memory.random_walk(percept)
        else:
            #print("New percept found", percept)
            new_percept = self.memory.ECM.add_vertex()
            self.memory.ECM.vp.percept[new_percept] = percept
            self.memory.new_p_clips.append(new_percept)
            for a in self.memory.a_clips:
                e = self.memory.ECM.add_edge(new_percept,a)
                self.memory.ECM.ep.h_value[e] = 1
            return self.memory.random_walk(percept)

    def learn(self, reward, done):
        # Update h-values from edges
        self.memory.update(reward, self.gamma, self.eta)

        # If succesful create a new action-clip with last action sequence
        # Else clear circuit and start over
        if reward > 0:
            print("We did it!!")
            self.memory.add_percept()
            if self.composition_active:
                self.memory.composition()
                self.memory.clip_deletion_action()
        else:
            if done:
                self.memory.clip_deletion_percept()
                self.memory.a_composition = []
'''
    def show_ECM(self):
        print("ECM:\n")
        print("Vertices\n")
        for v in self.memory.ECM.vertices():
            print("Vertice: ", self.memory.ECM.vertex_index[v])
            print("Percept: ", self.memory.ECM.vp.percept[v])
            print("Action: ", self.memory.ECM.vp.action[v])
            print("--------------------------------")

        edges = self.memory.ECM.get_edges()
        for e in self.memory.ECM.edges():
            index = self.memory.ECM.edge_index[e]
            print("Edge connecting vertex ")
'''
