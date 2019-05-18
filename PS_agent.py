import numpy as np
from graph_tool.all import *

class ECM():

    def __init__(self, actions, percepts):
        self.ECM = Graph()         # Network of clips that represents the memory
        self.p_clips = []          # List of active percept clips
        self.new_p_clips = []      # List of new percept clips that might be
                                   # removed
        self.a_clips = []          # List of action clips

        # Initializing action properties
        action = self.ECM.new_vertex_property("object")

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
        self.ECM.vertex_properties["percept"] = percept

        self.ECM.edge_properties["h_value"] = h_value
        self.ECM.edge_properties["glow"] = glow

    def random_walk(self, percept):
        # Finding clip that matches percept
        for v in self.ECM.vertices():
            if np.array_equal(self.ECM.vp.percept[v], percept):
                hopping_clip = v
                break

        # Random Walk until action-clip is found
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

        return self.ECM.vp.action[hopping_clip]

    def update(self, reward, gamma, eta):
        # Update h-value and glow of each edge
        for e in self.ECM.edges():
            self.ECM.ep.h_value[e] = max(1, self.ECM.ep.h_value[e] - gamma * (self.ECM.ep.h_value[e] - 1) + (reward * self.ECM.ep.glow[e]))
            self.ECM.ep.glow[e] = max(0, self.ECM.ep.glow[e] - (eta * self.ECM.ep.glow[e]))

# TODO Implement composition function
    def composition(self):
        pass

    def add_percept(self):
        # If the last sequence of actions resulted in a reward, add the
        # corresponding percept clips to the permanent list
        self.p_clips.extend(self.new_p_clips)
        self.new_p_clips = []

    def clip_deletion_percept(self):
        # If the last sequence of actions didn't result in a reward, delete the
        # percepts created during the episode
        for n_p in reversed(sorted(self.new_p_clips)):
            self.ECM.remove_vertex(n_p)
        self.update_clip_list()
        self.new_p_clips = []

#TODO Implement later after composition function
    def clip_deletion_action(self):
        pass

    def update_clip_list(self):
        # Reevaluate clips indexes after a deletion action
        self.a_clips = []
        self.p_clips = []

        for v in self.ECM.vertices():
            if self.ECM.vp.action[v] is None:
                self.p_clips.append(v)
            if self.ECM.vp.percept[v] is None:
                self.a_clips.append(v)

class PS_agent:

    def __init__(self, actions, percepts, eta, gamma):
        self.actions = actions                       # List of possible actions
        self.num_actions = len(actions)              # Total number of actions
        self.eta = eta                               # Glow damping parameter
        self.gamma = gamma                           # H-value damping parameter
        self.memory = ECM(actions, percepts)         # Agent memory

    def act(self, percept):
        # If percept is already in memory look for corresponding clip and start
        # a random walk
        for p in self.memory.p_clips:
            if np.array_equal(self.memory.ECM.vp.percept[p], percept):
                return self.memory.random_walk(percept)

        # If percept is a new percept but was already added in a temporary
        # percept-clip, find corresponding clip and start random-walk
        for new_p in self.memory.new_p_clips:
            if np.array_equal(self.memory.ECM.vp.percept[new_p], percept):
                return self.memory.random_walk(percept)
        else:
            # If percept not in memory, create corresponding clip and star
            # random walk
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

        # If last episode was succesful, add new percepts to the permanent
        # list. Else, clear recently created percept clips
        if done:
            if reward > 0:
                self.memory.add_percept()
            else:
                self.memory.clip_deletion_percept()
