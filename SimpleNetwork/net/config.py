import random as r


class Config():
    def __init__(self, LayersAmount:list):
        self.LayersAmount = LayersAmount
        self.Layers = []
        self.Weights = []
        self.create_layers()
    def create_layers(self):
        for index,layerAmount in enumerate(self.LayersAmount):
            Layer = []
            Weights = []
            for ind in range(layerAmount):
                Layer.append(0)
                WeightsNode = []
                if not index+1 == len(self.LayersAmount):
                    for i in range(self.LayersAmount[index+1]):
                        WeightsNode.append(round(r.uniform(-1,1),2))
                Weights.append(WeightsNode)
            self.Weights.append(Weights)
            self.Layers.append(Layer)