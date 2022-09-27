import torch

class BehaviorCloning(torch.nn.Module):
    def __init__(self, encoder, predictor):
        super(BehaviorCloning, self).__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, input_data):
        representation = self.encoder(input_data)
        action = self.predictor(representation)
        return action

class BehaviorCloningRep(torch.nn.Module):
    def __init__(self, encoder, predictor):
        super(BehaviorCloningRep, self).__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, input_data):
        with torch.no_grad():
            representation = self.encoder(input_data)
            
        action = self.predictor(representation)
        return action