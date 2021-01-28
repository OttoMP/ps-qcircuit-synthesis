from qutip import tensor, basis
import numpy as np
import sys

if "../" not in sys.path:
    sys.path.append("../")

from PS_agent import PS_agent
from envs.ibm_qx import Melbourne, Athens, Santiago, Valencia, Vigo, Yorktown
from lib.simulation import Simulation

print("Defining goal state")
# The state |110> is divided as:
# Qubit 0: 1
# Qubit 1: 1
# Qubit 2: 0
# Qubit 0 is always the qubit with the highest value (leftmost position)
# tensor(basis(2,1), basis(2,1), basis(2,0)) -> state |110>

# A goal state can also be defined by operating two different states
# e.g. a bell state
# bin_zero   = tensor(basis(2,0), basis(2,0)) -> |00>
# bin_three  = tensor(basis(2,1), basis(2,1)) -> |11>
# bell_state = 1/np.sqrt(2) * (zero+three)


# Define here your goal state
goal_state = tensor(basis(2,0), basis(2,0), basis(2,0), basis(2,1))
print(goal_state)


print("Creating Environment")
# Enviroments instantiation
# To define an environment you must choose from one of the available
# architectures It is possible to use less qubits than the maximum available by
# the architecture however the qubits used will be the ones listed with the
# smallest number until the required amount of qubits is filled
#
# Environment = (Number of Qubits, Maximum Circuit Depth, Goal State, Reward, Tolerance)
env = Melbourne(4, 3, goal_state, 50, 1e-13)


print("Creating Agent")
# Agents instantiation
# An action space is a list with all possible actions from the agent
# Action = (Gate Name, Controlled?, Control Qubit, Target Qubit)
action_space = [('X', False, 0, 0), ('CNOT', True, 0, 1), ('X', False, 1, 1), ('X', False, 2, 2), ('X', False, 3, 3)]
agent = PS_agent(action_space, [env.reset()], eta=0.01, gamma=0.001)

# Simulation instantiation
print("Setting simulation parameters")
experiment = Simulation(env, agent)

print("Starting simulation")
# Run simulation
number_of_episodes = 200
experiment.run_ps(number_of_episodes)
