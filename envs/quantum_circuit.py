import numpy as np
from copy import deepcopy

### QuantumCircuitEnv environment

class QuantumCircuitEnv2Qubits:

    def __init__(self, max_circuit_depth, goal_state, tolerance):
        # define state and action space
        #self.S
        self.action_space = ['X0', 'Y0', 'Z0', 'H0', 'X1', 'Y1', 'Z1', 'H1', 'CNOT10']

        # define reward structure
        #self.trace_distance()

        # define transitions
        #self.operate()

        self.max_circuit_depth = max_circuit_depth
        self.min_circuit_depth = max_circuit_depth
        self.goal_state = goal_state
        self.num_qubits = 2
        self.tolerance = tolerance
        self.gate_matrices = {
                'X0':(np.matrix([[0,0,1,0],[0,0,0,1], [1,0,0,0], [0,1,0,0]]), 0.77),
                'Y0':(np.matrix([[0,0,-1j,0],[0,0,0,-1j], [1j,0,0,0], [0,1j,0,0]]), 0.77),
                'Z0':(np.matrix([[1,0,0,0],[0,1,0,0], [0,0,-1,0], [0,0,0,-1]]), 0.77),
                'H0':(1/np.sqrt(2) * np.matrix([[1,0,1,0],[0,1,0,1], [1,0,-1,0], [0,1,0,-1]]), 0.77),
                'X1':(np.matrix([[0,1,0,0],[1,0,0,0], [0,0,0,1], [0,0,1,0]]), 1.46),
                'Y1':(np.matrix([[0,-1j,0,0],[-1j,0,0,0], [0,0,0,1j], [0,0,1j,0]]), 1.46),
                'Z1':(np.matrix([[1,0,0,0],[0,-1,0,0], [0,0,1,0], [0,0,0,-1]]), 1.46),
                'H1':(1/np.sqrt(2) * np.matrix([[1,1,0,0],[1,-1,0,0], [0,0,1,1], [0,0,1,-1]]), 1.46),
                'CNOT10':(np.matrix([[1,0,0,0],[0,0,0,1], [0,0,1,0], [0,1,0,0]]), 2.7),
                }

        self.circuit_depths = np.zeros(self.num_qubits)
        self.circuit_gates = [[] for q in range(self.num_qubits)]
        self.sum_error = 0

    def reset(self):
        self.s = self.init_comp_basis()
        self.circuit_depths = np.zeros(self.num_qubits)
        self.circuit_gates = [[] for q in range(self.num_qubits)]
        self.sum_error = 0
        self.is_reset = True

        return self.s

    def action2matrix(self, action):
        return self.gate_matrices[action]

    def calculate_circuit_depth(self, a):
        if a == 'X0' or a == 'Y0' or a == 'Z0' or a == 'H0':
            self.circuit_depths[0] += 1
            if max(self.circuit_depths) > self.max_circuit_depth:
                return self.max_circuit_depth
            else:
                self.circuit_gates[0].append(a)
                return max(self.circuit_depths)
        elif a == 'X1' or a == 'Y1' or a == 'Z1' or a == 'H1':
            self.circuit_depths[1] += 1
            if max(self.circuit_depths) > self.max_circuit_depth:
                return self.max_circuit_depth
            else:
                self.circuit_gates[1].append(a)
                return max(self.circuit_depths)
        elif a == 'CNOT10':
            self.circuit_depths[1] = max(self.circuit_depths[0], self.circuit_depths[1])
            self.circuit_depths[0] = max(self.circuit_depths[0], self.circuit_depths[1])
            self.circuit_depths[1] += 1
            self.circuit_depths[0] += 1
            if max(self.circuit_depths) > self.max_circuit_depth:
                return self.max_circuit_depth
            else:
                self.circuit_gates[1].append(a)
                self.circuit_gates[0].append(a)
                return max(self.circuit_depths)

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
        new_s.shape = (4,1)
        return new_s*np.conj(new_s).T

    def operate(self, s, a):
        return np.dot(s,a)

    def step(self,action):
        s_prev = self.s
        a, e = self.action2matrix(action)
        self.sum_error += e*e
        self.s = self.operate(self.s, a)
        reward = self.trace_distance(self.s)
        depth = self.calculate_circuit_depth(action)
        self.is_reset = False

        if reward > 0 or max(self.circuit_depths) > self.max_circuit_depth:
            if self.min_circuit_depth > depth:
                self.min_circuit_depth = depth
            output = open('output.out', 'a')
            print("Gates:\n", file = output)
            print("qubit 0: ", self.circuit_gates[0], file = output)
            print("qubit 1: ", self.circuit_gates[1], file = output)
            print("min circuit depth: ", self.min_circuit_depth, file = output)
            if reward > 0:
                reward = (reward-self.sum_error) * (self.min_circuit_depth/(depth*depth))
                print("Right circuit", reward, file = output)
            print("\n", file = output)
            output.close()

            self.reset()

        return (self.s, reward, self.is_reset, '')

    def init_comp_basis(self):
        basis = np.array([1,0,0,0])
        return basis
        '''
        basis = np.array([1,0])
        basis.shape = (2,1)
        tensor_basis = np.tensordot(basis,basis,axes=(0,0)).flatten()
        tensor_basis.shape = (4,1)
        return tensor_basis.transpose()
        '''

