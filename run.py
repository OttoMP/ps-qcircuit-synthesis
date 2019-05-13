import numpy as np
import sys

if "../" not in sys.path:
    sys.path.append("../")

from PS_agent import PS_agent
from lib.envs.simple_rooms import SimpleRoomsEnv
from lib.envs.quantum_circuit import QuantumCircuitEnv2Qubits
from lib.simulation import Experiment

interactive = False
zero = np.array([1,0,0,0])
one = np.array([0,1,0,0])
two = np.array([0,0,1,0])
three = np.array([0,0,0,1])
goal_state = 1/np.sqrt(2) * (zero+three)
#env = SimpleRoomsEnv()
env = QuantumCircuitEnv2Qubits(4,goal_state)
agent = PS_agent(env.action_space.actions, [env.reset()], eta=0.1, gamma=0.1)
experiment = Experiment(env, agent)
experiment.run_ps(100, interactive)
