import matplotlib.pyplot as plt
from Neuron import *




N = Neuron()

Data = [
    [[1,1],0],
    [[1,0],1],
    [[0,1],0],
    [[0,0],0],
]

for i in Data:
    print(f'{i} : {N.output(i[0])}')

Error, Weight1, Weight2, Bias = N.learn(Data,100)

for i in Data:
    print(f'{i} : {N.output(i[0])}')


plt.xlabel('Weights and Bias')
plt.ylabel('Error')
plt.plot(Weight1, Error, label='Weight1')
plt.plot(Weight2, Error, label='Weight2')
plt.plot(Bias, Error, label='Bias')

plt.legend()
plt.show()
