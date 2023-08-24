import matplotlib.pyplot as plt
from FCN import *


N = Network([2,1])





Data = [
    [[1,1],[0]],
    [[1,0],[1]],
    [[0,1],[0]],
    [[0,0],[0]],
]

VisualCost, VisualStep = N.learn(Data, 25000)

for i in Data:
    print(f'{i} : {N.output(i[0])}')


plt.xlabel('Time')
plt.ylabel('Cost')
plt.plot(VisualStep, VisualCost, label='Cost')

plt.legend()
plt.show()
