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

class QuantumCircuitEnv2Qubits(Environment):

    def __init__(self, max_circuit_depth, goal_state):
        super(QuatumCircuitEnv2Qubits, self).__init__()
        # define state and action space
        #self.S
        self.action_space = ActionSpace(['X', 'Y', 'Z', 'H'])

        # define reward structure
        #self.trace_distance()

        # define transitions
        #self.operate()

        self.max_circuit_depth = max_circuit_depth
        self.goal_state = goal_state
        self.num_qubits = 1
        self.tolerance = 0.001

        self.circuit_depths = np.zeros(self.num_qubits)
        self.circuit_gates = [[] for q in range(num_qubits)]

    def reset(self):
        self.s = self.init_comp_basis()
        self.circuit_depths = np.zeros(self.num_qubits)
        self.is_reset = True

        return self.s

    def action2matrix(self, action):
        return {
                'X':np.matrix([[0,1],[1,0]]),
                'Y':np.matrix([[0,-1j],[1j,0]]),
                'Z':np.matrix([[1,0],[0,-1]]),
                'H':np.matrix([[1,1],[1,-1]]),
                }[action]

    def step(self,action):
        s_prev = self.s
        a = action2matrix(action)
        self.s = self.operate(self.s, a)
        reward = self.trace_distance(self.s)
        calculate_circuit_depth(action)
        self.is_reset = False

        if (reward < -1. * (self.tolerance) or reward > self.tolerance) or max(self.circuit_depths) > self.max_circuit_depth:
            print("Gates: ", circuit_gates)
            self.reset()

        return (self.s, reward, self.is_reset, '')

    def calculate_circuit_depth(self, a):
        if a == 'X' or a == 'Y' or a == 'Z' or a == 'H':
            self.circuit_gates[0].append(a)
            self.circuit_depths[0] += 1

    def trace_distance(self, s):
        density_s = density_matrix(s)
        density_goal = density_matrix(self.goal_state)
        trace = sum(abs(np.linalg.eigvals(density_s - density_b)))/2

        if trace < self.tolerance:
            return 100
        else:
            return 0

    def density_matrix(self,s):
        return s*np.conj(s).T

    def operate(self, s, a):
        return np.dot(s,a)

    def init_comp_basis(self):
        return np.array([1,0])
        '''
        basis = np.array([1,0])
        basis.shape = (2,1)
        basis = basis.transpose()
        tensor_basis = np.tensordot(basis,basis,axes=(0,0)).flatten()
        tensor_basis.shape = (4,1)
        return tensor_basis.transpose()
        '''
