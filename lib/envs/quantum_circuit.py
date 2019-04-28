import numpy as np

### Interface
class Environment(object):

    def reset(self):
        raise NotImplementedError('Inheriting classes must override reset.')

    def actions(self):
        raise NotImplementedError('Inheriting classes must override actions.')

    def step(self):
        raise NotImplementedError('Inheriting classes must override step')

class ActionSpace(object):

    def __init__(self, actions):
        self.actions = actions
        self.n = len(actions)

### QuantumCircuitEnv environment

class QuantumCircuitEnvironment2Qubit(Environment):

    def __init__(self, circuit_depth):
        # define state and action space

        # define reward structure

        # define transitions

        self.circuit_depth = circuit_depth

    def reset(self):
        pass

    def actions(self):
        pass

    def step(self):
        pass

    def reset(self):
        pass
