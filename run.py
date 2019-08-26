from qutip import tensor, basis
import numpy as np
import sys

if "../" not in sys.path:
    sys.path.append("../")

from PS_agent import PS_agent
from envs.quantum_circuit import QuantumCircuitEnv2Qubits, QuantumCircuitEnv3Qubits, QuantumCircuitEnv4Qubits, QuantumCircuitEnv5Qubits
from lib.simulation import Simulation

interactive = False

# Computational basis |00>, |01>, |10>, |11>
zero  = tensor(basis(2,0), basis(2,0))
one   = tensor(basis(2,0), basis(2,1))
two   = tensor(basis(2,1), basis(2,0))
three = tensor(basis(2,1), basis(2,1))

b30 = tensor(basis(2,0), basis(2,0), basis(2,0))
b31 = tensor(basis(2,1), basis(2,1), basis(2,1))

b40 = tensor(basis(2,0), basis(2,0), basis(2,0), basis(2,0))
b41 = tensor(basis(2,1), basis(2,1), basis(2,1), basis(2,1))

b50 = tensor(basis(2,0), basis(2,0), basis(2,0), basis(2,0), basis(2,0))
b51 = tensor(basis(2,1), basis(2,1), basis(2,1), basis(2,1), basis(2,1))


# Bell State to be reached
goal_state = 1/np.sqrt(2) * (zero+three)
#goal_state = 1/np.sqrt(2) * (b30+b31)
#goal_state = 1/np.sqrt(2) * (b40+b41)
#goal_state = 1/np.sqrt(2) * (b50+b51)

# Enviroments instantiation
env = QuantumCircuitEnv2Qubits(4, goal_state, 1e-13)
#env = QuantumCircuitEnv3Qubits(5, goal_state, 1e-13)
#env = QuantumCircuitEnv4Qubits(7, goal_state, 1e-13)
#env = QuantumCircuitEnv5Qubits(9, goal_state, 1e-13)

# Agents instantiation
agent = PS_agent(env.action_space, [env.reset()], eta=0.1, gamma=0.1)

# Simulation instantiation
experiment = Simulation(env, agent)

# Run simulation 200 times
experiment.run_ps(2)
