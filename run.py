import numpy as np
import sys

if "../" not in sys.path:
    sys.path.append("../")

from PS_agent import PS_agent
from lib.envs.simple_rooms import SimpleRoomsEnv
from lib.envs.quantum_circuit import QuantumCircuitEnv2Qubits
from lib.simulation import Experiment

interactive = False
goal_state = np.array([0,1])
#env = SimpleRoomsEnv()
env = QuantumCircuitEnv2Qubits(3,goal_state)
agent = PS_agent(env.action_space.actions, [env.reset()], eta=0.1, gamma=0.001)
experiment = Experiment(env, agent)
experiment.run_ps(10, interactive)
