from net.config import *
import math

class Network():
    def __init__(self, config:Config):
        self.Layers = config.Layers
        self.Weights = config.Weights
    
    def sigmoid(self,x):
        return 1/(1 + math.exp(-x))

    def predict(self, input):
        self.Layers[0] = input
        for layIndex, Layer in enumerate(self.Layers):
            for index, node in enumerate(Layer):
                if not layIndex == 0:
                    self.Layers[layIndex][index] = self.sigmoid(self.Layers[layIndex][index])
                if not layIndex+1 == len(self.Layers):
                    for weightindex in range(len(self.Layers[layIndex+1])):
                        self.Layers[layIndex+1][weightindex] += self.Weights[layIndex][index][weightindex] * self.Layers[layIndex][index]
        print(f'Predicted OutPut: {self.Layers[-1]}')