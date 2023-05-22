import torch
import flwr as fl

from models import Net

basic_config = {
    'target' : {
        'name': 'strategy',
        'item': ['FedAvg', 'FedAvgM', 'FedOpt', 'FedProx'],
    }, 
    'model' : {
        'type': 'basic',
        'path': '',
    },
    'data': {
        'type': 'cifar10',
    },
    'run': {
        'random_seed': 1234,
        'num_clients': 10,
        'round': 100,
        'local_epoch': 1,
        'learning_rate': 0.01,
    },
    'save': {
        'model_path': '',
        'log_path': '',
    },
}

class Experiment:
    models = {'basic': Net}
    

    def __init__(self, config=basic_config):
        self.config = config

    def intialize(self):
        ## model ##
        pass

    def run(self):
        pass

    
    
    
        
