import net


Con = net.Config([64,10])
Net = net.Network(Con)




#Preparing Data
from sklearn.datasets import load_digits
digits = load_digits()
TrainData = []
TestData = []

for i in range(digits.data.shape[0]):
    if i<1500:
        TrainData.append([digits.images[i].flatten().tolist(), digits.target[i]])
    else:
        TestData.append([digits.images[i].flatten().tolist(), digits.target[i]])


Net.train(TrainData[4:5])

