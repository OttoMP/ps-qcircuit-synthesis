import numpy as np
from copy import deepcopy

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

class QuantumCircuitEnv2Qubits(Environment):

    def __init__(self, max_circuit_depth, goal_state):
        #super(QuatumCircuitEnv2Qubits, self).__init__()
        # define state and action space
        #self.S
        self.action_space = ActionSpace(['Z', 'H'])

        # define reward structure
        #self.trace_distance()

        # define transitions
        #self.operate()

        self.max_circuit_depth = max_circuit_depth
        self.goal_state = goal_state
        self.num_qubits = 1
        self.tolerance = 0.001

        self.circuit_depths = np.zeros(self.num_qubits)
        self.circuit_gates = [[] for q in range(self.num_qubits)]

    def reset(self):
        self.s = self.init_comp_basis()
        self.circuit_depths = np.zeros(self.num_qubits)
        self.circuit_gates = [[] for q in range(self.num_qubits)]
        self.is_reset = True

        return self.s

    def action2matrix(self, action):
        return {
                'X':np.matrix([[0,1],[1,0]]),
                'Y':np.matrix([[0,-1j],[1j,0]]),
                'Z':np.matrix([[1,0],[0,-1]]),
                'H':1/np.sqrt(2) * np.matrix([[1,1],[1,-1]]),
                }[action]

    def calculate_circuit_depth(self, a):
        if a == 'X' or a == 'Y' or a == 'Z' or a == 'H':
            self.circuit_depths[0] += 1
            if max(self.circuit_depths) > self.max_circuit_depth:
                return
            else:
                self.circuit_gates[0].append(a)

    def trace_distance(self, s):
        density_s = self.density_matrix(s)
        density_goal = self.density_matrix(self.goal_state)
        trace = sum(abs(np.linalg.eigvals(density_s - density_goal)))/2

        if trace < self.tolerance:
            return 100
        else:
            return 0

    def density_matrix(self,s):
        new_s = deepcopy(s)
        new_s.shape = (2,1)
        return new_s*np.conj(new_s).T

    def operate(self, s, a):
        return np.dot(s,a)

    def step(self,action):
        s_prev = self.s
        a = self.action2matrix(action)
        self.s = self.operate(self.s, a)
        reward = self.trace_distance(self.s)
        self.calculate_circuit_depth(action)
        self.is_reset = False

        if (reward < -1. * (self.tolerance) or reward > self.tolerance) or max(self.circuit_depths) > self.max_circuit_depth:
            print("Gates: ", self.circuit_gates)
            self.reset()

        return (self.s, reward, self.is_reset, '')

    def init_comp_basis(self):
        basis = np.array([1,0])
        return basis
        '''
        basis = np.array([1,0])
        basis.shape = (2,1)
        tensor_basis = np.tensordot(basis,basis,axes=(0,0)).flatten()
        tensor_basis.shape = (4,1)
        return tensor_basis.transpose()
        '''
