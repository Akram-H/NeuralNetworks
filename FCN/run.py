from FCN import *


N = Network([2,2])





Data = [
    [[1,1],[0,1]],
    [[1,0],[1,1]],
    [[0,1],[0,0]],
    [[0,0],[0,0]],
]


N.learn(Data, 1000)

for i in Data:
    print(f'{i} : {N.output(i[0])}')