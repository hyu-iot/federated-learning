from collections import namedtuple
import json

class Config(object):
    """Configuration module."""

    def __init__(self, config):
        self.path = ""
        # Load config file
        with open(config, 'r') as config:
            self.config = json.load(config)
        # Extract configuration
        self.extract()
    
    def extract(self):
        config = self.config

        # --- Clients ---
        fields = ['total', 'per_round', 'validation_ratio']
        defaults = (0, 0, 0.0)
        params = [config['clients'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.clients = namedtuple('clients', fields)(*params)

        # --- Data ---
        fields = ['dataset', 'path', 'batch_size', 'remain_ratio', 'random_seed']
        defaults = ('cifar10', '', 32, 0.0, 1234)
        params = [config['data'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.data = namedtuple('data', fields)(*params)

        # --- Model ---
        self.models = config['model']

        # --- Strategy ---
        self.strategies = config['strategy']

        # --- Federated learning ---
        fields = ['rounds', 'epochs', 'learning_rate', 'momentum', 'weight_decay',
                  'fraction_fit', 'fraction_evaluate', 'min_fit_clients', 
                  'min_evaluate_clients', 'min_available_clients']
        defaults = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        params = [config['federated_learning'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.fl = namedtuple('fl', fields)(*params)        

        # --- Paths ---
        fields = ['data', 'models']
        defaults = ('./data', './models')
        params = [config['clients'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.paths = namedtuple('paths', fields)(*params)

        # --- Visualization ---
        self.visualization = config['visualization']
