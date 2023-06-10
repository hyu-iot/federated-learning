import torch
import torch.nn as nn
import torch.optim as optim


import flwr as fl
from flwr.common import Metrics

from utils import train, test, get_parameters, set_parameters

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, device, net, trainloader, valloader):
        self.device = device
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
    
    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.01,
                            momentum=0.9, weight_decay=5e-4)
        train(self.device, self.net, self.trainloader, criterion, optimizer, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}
    
    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        criterion = nn.CrossEntropyLoss()
        loss, metrics = test(self.device, self.net, self.valloader, criterion)
        return float(loss), len(self.valloader), metrics
