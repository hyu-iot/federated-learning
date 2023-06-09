import torch
import torch.nn as nn
import torch.optim as optim


import flwr as fl
from flwr.common import Metrics

from utils import train, test, get_parameters, set_parameters

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, device, net, trainloader, valloader, train_config):
        self.device = device
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.train_config = train_config

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), **(self.train_config))
        train(self.device, self.net, self.trainloader, criterion, optimizer, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}
    
    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        criterion = nn.CrossEntropyLoss()
        loss, metrics = test(self.device, self.net, self.valloader, criterion)
        return float(loss), len(self.valloader), metrics
