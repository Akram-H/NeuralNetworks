import math



class Neuron():
    def __init__(self) -> None:
        self.x1 = 0
        self.x2 = 0
        self.w1 = .5
        self.w2 = .5
        self.b = 0
        self.z = 0
        self.y = 0

        self.lr = .01

    def sigmoid(self,x):
        return 1/(1 + math.exp(-x))
    def dsigmoid(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def learn(self, Data, epochAmount=100):

        Error = []
        Weight1 = []
        Weight2 = []
        Bias = []



        for epoch in range(epochAmount):
            avg_error = 0

            for i in Data:
                self.x1, self.x2 = i[0]
                self.z = self.x1 * self.w1 + self.x2 * self.w2 + self.b
                self.y = self.sigmoid(self.z)
                
                avg_error += (i[1]-self.y)**2

                de = -2*(i[1]-self.y)
                dw1 = de * self.dsigmoid(self.z) * self.x1
                dw2 = de * self.dsigmoid(self.z) * self.x2
                db =  de * self.dsigmoid(self.z)

                self.w1 -= dw1 * self.lr
                self.w2 -= dw2 * self.lr
                self.b -= db * self.lr
            
            Error.append(avg_error/len(Data))
            Weight1.append(self.w1)
            Weight2.append(self.w2)
            Bias.append(self.b)

        return Error, Weight1, Weight2, Bias


    def output(self, input):
        self.x1, self.x2 = input
        self.z = self.x1 * self.w1 + self.x2 * self.w2 + self.b
        self.y = self.sigmoid(self.z)
        return self.y