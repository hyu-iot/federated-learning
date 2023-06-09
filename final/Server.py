import logging
import os
from datetime import datetime

from load_data import DataManager


class Server:
    def __init__(self, config):
        self.config = config

        logging.info('Starting...')
        
        self.create_directory()
        self.load_data()

    def run(self):
        self.iterate_simulation()
        pass


    def load_data(self):
        config = self.config

        data_config = {
            'dataset': config.data.dataset,
            'batch_size': config.data.batch_size,
            'remain_ratio': config.data.remain_ratio,
            'validation_ratio': config.clients.validation_ratio,
            'num_clients': config.clients.total,
            'random_seed': config.data.random_seed,
        }

        dm = DataManager(**data_config)
        self.trainloader, self.testloader, self.dl_clients, _ = dm.get_data()

        print(self.config.models)
        pass

    # Make the dirctory named using the timestamp.
    def create_directory(self):
        if not os.path.exists("./result"):
            os.makedirs("./result")
        self.mydir = os.path.join(os.getcwd(), "./result/", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.mydir)
        # print(self.mydir)
        
    def create_simulation(self):
        pass
    def iterate_simulation(self):
        for key in self.config.models:
            for strategy in self.config.strategies:
                # print(key)
                # print(strategy)
                self.create_simulation()

        pass


