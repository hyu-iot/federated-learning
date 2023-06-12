from collections import namedtuple
import json
import logging

class Config(object):
    """Configuration module."""

    def __init__(self, args):
        self.path = ""
        # Load config file
        with open(args.config, 'r') as config:
            self.config = json.load(config)
        # Extract configuration
        self.extract(args)
    
    def extract(self, args):
        config = self.config

        # --- Clients ---
        fields = ['total', 'per_round', 'validation_ratio']
        defaults = (0, 0, 0.0)
        params = [config['clients'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.clients = namedtuple('clients', fields)(*params)

        # --- Data ---
        fields = ['dataset', 'path', 'custom_path','batch_size', 'remain_ratio', 'random_seed']
        defaults = ('cifar10', '', '', 32, 0.0, 1234)
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
        if args.vis != 'null' and config['visualization']['only_visualization'] == 'null':
            logging.info('The --vis option is true, but the config.json\'s only_visualization option says false.\n\
                The only_visualization option will be true.\n\
                If you do not want to see this message, change the config.json\'s only_visualization option to \'false.\' ')
            config['visualization']['only_visualization'] = args.vis
        elif args.vis == 'null' and config['visualization']['only_visualization'] != 'null':
            logging.debug('The --vis option is defalutly null, but the config.json\'s only_visualization option passes a path.\n\
                        This will only do the visualization with the give path.\n\
                        If you do not want to see this message, add the \"--vis \'path\'\"')
        else: 
            config['visualization']['only_visualization'] = args.vis
        self.visualization = config['visualization']
