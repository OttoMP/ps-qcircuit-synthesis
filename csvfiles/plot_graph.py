import matplotlib.pyplot as plt
import csv

x = []
y = []
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels

with open('5qubits/circuits_found_5qubits.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(float(row[0]))
        y.append(float(row[1]))

plt.plot(x,y)
plt.xlabel('episode')
plt.ylabel('new circuits found')
plt.title('5-qubits circuits found by synthetizer.')
plt.legend()
plt.show()
plt.close()

x = []
y = []
'''
with open('3qubits/reward_3qubits.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(float(row[0]))
        y.append(float(row[1]))

plt.plot(x,y)
plt.xlabel('episode')
plt.ylabel('reward')
plt.title('Reward obtained by each episode.')
plt.legend()
plt.show()
plt.close()

x = []
y = []

with open('3qubits/length_3qubits.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(float(row[0]))
        y.append(float(row[1]))

plt.plot(x,y)
plt.xlabel('episode')
plt.ylabel('episode length')
plt.title('Episode length.')
plt.legend()
plt.show()
'''
