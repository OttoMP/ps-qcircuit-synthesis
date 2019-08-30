import matplotlib.pyplot as plt
import csv

x = []
y = []

with open('circuits_found.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(float(row[0]))
        y.append(float(row[1]))

plt.plot(x,y)
plt.xlabel('episode')
plt.ylabel('new circuits found')
plt.title('Circuits found by synthetizer.')
plt.legend()
plt.show()
