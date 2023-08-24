import math


class Network():
    def __init__(self, layers):
        self.lr = 0.02
        

        self.Layers = []
        self.Weights = []
        self.Biases = []
        self.z = []
        for LayerIndex, LayerAmount in enumerate(layers):
            z_values = []
            layer = []
            weights = []
            bias = []
            
            for i in range(LayerAmount):
                layer.append(0)
                z_values.append(0)
                if not LayerIndex == 0:
                    bias.append(0)
                if not LayerIndex+1 == len(layers):
                    nextLayer = []
                    for nextindex in range(layers[LayerIndex+1]):
                        nextLayer.append(.5)
                    weights.append(nextLayer)
            
            self.z.append(z_values)
            self.Layers.append(layer)
            if not LayerIndex+1 == len(layers):
                self.Weights.append(weights)
            self.Biases.append(bias)

    def sigmoid(self,x):
        try:
            return 1/(1 + math.exp(-x))
        except OverflowError:
            if x < 0:
                return 0
            else:
                return 1


    def dsigmoid(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))


    def output(self, input):
        self.Layers[0] = input

        for LayerIndex, Layer in enumerate(self.Layers):
            if not LayerIndex == 0:
                for NodeIndex, Node in enumerate(self.Layers[LayerIndex]):
                    self.Layers[LayerIndex][NodeIndex] = 0
                    for previusNodeIndex,previusNode in enumerate(self.Layers[LayerIndex-1]):
                        self.Layers[LayerIndex][NodeIndex] += previusNode*self.Weights[LayerIndex-1][previusNodeIndex][NodeIndex]
                    self.z[LayerIndex][NodeIndex] = self.Layers[LayerIndex][NodeIndex] 
                    self.Layers[LayerIndex][NodeIndex] = self.sigmoid(self.Layers[LayerIndex][NodeIndex]+self.Biases[LayerIndex][NodeIndex])
        
        return self.Layers[-1]


    def learn(self, Data, Epochs=100):
        
        VisualCost = []
        VisualStep = []
        


        for e in range(Epochs):
            c = 0
            for input,target in Data:
                predicted = self.output(input)
                
                Errors = []

                firstLayerError = []
                for i,y in enumerate(predicted):
                    firstLayerError.append(target[i]-y)
                Errors.append(firstLayerError)

                Cost = 0
                for i in firstLayerError:
                    Cost += i**2 
                Cost *= 1/len(firstLayerError)
                c = Cost
                for MirrorLayerIndex in range(len(self.Layers)):
                    LayerIndex = len(self.Layers)-MirrorLayerIndex-1
                    Layer = self.Layers[LayerIndex]

                    if MirrorLayerIndex == 0:
                        for NodeIndex, Weights in enumerate(self.Weights[LayerIndex-1]):
                            for index,weight in enumerate(Weights):
                                de = -firstLayerError[index]
                                dy = self.dsigmoid(self.z[LayerIndex][index])
                                dz = self.Layers[LayerIndex-1][NodeIndex]
                                self.Weights[LayerIndex-1][NodeIndex][index] -= de * dz * dz *self.lr
                                
                        for NodeIndex, Node in enumerate(Layer):
                            de = -firstLayerError[NodeIndex]
                            dy = self.dsigmoid(self.z[LayerIndex][index])
                            self.Biases[LayerIndex][NodeIndex] -= de * dy * self.lr
                    else:
                        if not MirrorLayerIndex+1 == len(self.Layers):
                            SecondLayerError = []
                            for NodeIndex, Node in enumerate(Layer):
                                de = 0
                                for LastIndex, lastLayerError in enumerate(Errors[MirrorLayerIndex-1]):
                                    de += lastLayerError * self.dsigmoid(self.z[LayerIndex+1][LastIndex]) * self.Weights[LayerIndex][NodeIndex][LastIndex]
                                

                                for NodeIndex, Weights in enumerate(self.Weights[LayerIndex-1]):
                                    for index,weight in enumerate(Weights):
                                        dy = self.dsigmoid(self.z[LayerIndex][index])
                                        dz = self.Layers[LayerIndex-1][NodeIndex]
                                        self.Weights[LayerIndex-1][NodeIndex][index] -= -de * dz * dz *self.lr
                                for NodeIndex, Node in enumerate(Layer):
                                    dy = self.dsigmoid(self.z[LayerIndex][index])
                                    self.Biases[LayerIndex][NodeIndex] -= -de * dy * self.lr
                                SecondLayerError.append(de)
                            Errors.append(SecondLayerError)
            VisualCost.append(c)
            VisualStep.append(e)

        return VisualCost, VisualStep