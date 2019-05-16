import numpy as np
import sys

if "../" not in sys.path:
    sys.path.append("../")

from PS_agent import PS_agent
from envs.quantum_circuit import QuantumCircuitEnv2Qubits
from lib.simulation import Simulation

interactive = False
zero = np.array([1,0,0,0])
one = np.array([0,1,0,0])
two = np.array([0,0,1,0])
three = np.array([0,0,0,1])
goal_state = 1/np.sqrt(2) * (zero+three)
#env = SimpleRoomsEnv()
env = QuantumCircuitEnv2Qubits(4,goal_state, 1e-13)
agent = PS_agent(env.action_space, [env.reset()], eta=0.1, gamma=0.005)
experiment = Simulation(env, agent)
experiment.run_ps(10)
