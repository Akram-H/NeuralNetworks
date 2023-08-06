from net.config import *
import math
import time

class Network():
    def __init__(self, config:Config):
        self.Layers = config.Layers
        self.Weights = config.Weights
        self.Biases = config.biases

        self.learning_rating = 0.25


    def sigmoid(self,x):
        return 1/(1 + math.exp(-x))
    def sigmoid_derivative(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    
    
    def clear_layers(self):
        for index,layer in enumerate(self.Layers):
            for ind, node in enumerate(layer):
                self.Layers[index][ind] = 1
    def predict(self, input):
        if len(self.Layers[0]) != len(input):
            return "Invalid Input"
        self.clear_layers()
        self.Layers[0] = input
        for layIndex, Layer in enumerate(self.Layers):
            for index, node in enumerate(Layer):
                if not layIndex == 0:
                    self.Layers[layIndex][index] = self.sigmoid(self.Layers[layIndex][index]+self.Biases[layIndex][index])
                if not layIndex+1 == len(self.Layers):
                    for weightindex in range(len(self.Layers[layIndex+1])):
                        self.Layers[layIndex+1][weightindex] += self.Weights[layIndex][index][weightindex] * self.Layers[layIndex][index]
        return self.Layers[-1]
    

    def train(self, TrainData):
        for data in TrainData:
            self.clear_layers()
            self.Layers[0] = data[0]
            X_Values = []
            for layIndex, Layer in enumerate(self.Layers):
                X = []
                for index, node in enumerate(Layer):
                    if not layIndex == 0:
                        X.append(self.Layers[layIndex][index]+self.Biases[layIndex][index])
                        self.Layers[layIndex][index] = self.sigmoid(self.Layers[layIndex][index]+self.Biases[layIndex][index])
                    if not layIndex+1 == len(self.Layers):
                        for weightindex in range(len(self.Layers[layIndex+1])):
                            self.Layers[layIndex+1][weightindex] += self.Weights[layIndex][index][weightindex] * self.Layers[layIndex][index]
                X_Values.append(X)

            

            Target = []
            for i in range(10):
                if i == data[1]:
                    Target.append(1)
                else:
                    Target.append(0)

            print(self.Layers[-1])
            print(Target)


            Cost = 0
            for j,node in enumerate(self.Layers[-1]):
                Cost += (Target[j]-node)

                bias_derivative = (Target[j]-node) * self.sigmoid_derivative(X_Values[-1][j])
                self.Biases[-1][j] += bias_derivative * self.learning_rating
                for i,previus_node in enumerate(self.Layers[-2]):
                    weight_derivative = (Target[j]-node) * self.sigmoid_derivative(X_Values[-1][j]) * previus_node
                    self.Weights[-2][i][j] += weight_derivative * self.learning_rating
            Cost *= 0.5

            print(Cost)