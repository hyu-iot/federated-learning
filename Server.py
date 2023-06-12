import logging
import os
from datetime import datetime
from visualization import Visualization
import shutil

import torch


from load_data import DataManager
from client import FlowerClient
from utils import train, test, get_parameters, set_parameters
from strategy.strategy import get_strategy
from server.basic_simulation import Simulation_Unit

class Server:
    def __init__(self, config):
        self.config = config

        logging.info('Starting...')
        
        if config.visualization['only_visualization'] == 'null':
            self.create_directory()
            self.load_data()

    def run(self):
        # self.iterate_simulation()
        config = self.config
        for strategy in config.strategies:
            for modelname, model_config in config.models.items():
                config.model = {"modelname": modelname, **model_config}
                config.strategy = strategy
                config.data = {"trainloader": self.trainloader, "testloader": self.testloader, "clients": self.dl_clients}
                config.paths = {"models": "./models", "result_dir": self.mydir}
                simunit = Simulation_Unit(config)
                #simunit = Simulation_Unit(self.__make_unit_simulation_config(strategy, modelname, model_config))
                simunit.run()


    def load_data(self):
        config = self.config

        data_config = {
            'dataset': config.data.dataset,
            'path': config.data.path,
            'custom_path' : config.data.custom_path,
            'batch_size': config.data.batch_size,
            'remain_ratio': config.data.remain_ratio,
            'validation_ratio': config.clients.validation_ratio,
            'num_clients': config.clients.total,
            'random_seed': config.data.random_seed,
        }

        dm = DataManager(**data_config)
        self.trainloader, self.testloader, self.dl_clients, _ = dm.get_data()


    # Make the dirctory named using the timestamp.
    def create_directory(self):
        if not os.path.exists("./result"):
            os.makedirs("./result")
        self.mydir = os.path.join(os.getcwd(), "./result/", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.mydir)
        shutil.copyfile('./configs/config.json', self.mydir +'/config.json')


    def __make_unit_simulation_config(self, strategy, modelname, model_config):
        config = self.config
        simulation_config = {
            "num_clients": config.clients.total,
            "num_rounds": config.fl.rounds,
            "strategy": strategy,
            "model": {
                "modelname": modelname,
                **model_config,
            },
            "data": {
                "trainloader": self.trainloader,
                "testloader" : self.testloader,
                "clients": self.dl_clients
            },
            "train_config": {
                "lr": config.fl.learning_rate,
                "momentum": config.fl.momentum,
                "weight_decay": config.fl.weight_decay
            },
            "paths": {
                "models": "./models",
                "result_dir": self.mydir,
            }
        }
        return simulation_config    