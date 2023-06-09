import logging
import os

from typing import List, Tuple, Dict, Optional

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import flwr as fl
from flwr.common import Metrics

from client import FlowerClient
from models.models import load_model
from strategy.strategy import get_strategy
from utils import train, test, get_parameters, set_parameters


class Simulation_Unit(object):
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = load_model(**(config['model'])).to(self.device)

    # def load_model(self):
    #     return load_model(**(self.config.model))


    def run(self):
        config = self.config
        net = self.model
        DEVICE = self.device

        # TODO : make this client_resources to be defined by config.json
        clinet_resources = None
        if DEVICE.type == "cuda":
            client_resources = {"num_gpus": 1}

        # TODO : modify config.json and config.py
        #        to let one strategy have one config
        #        (refer the case of models)
        strategy_config = {
            "fraction_fit": config.fl.fraction_fit,
            "fraction_evaluate": config.fl.fraction_evaluate,
            "min_fit_clients": config.fl.min_fit_clients,
            "min_evaluate_clients": config.fl.min_evaluate_clients,
            "min_available_clients": config.fl.min_available_clients,
            "evaluate_metrics_aggregation_fn" : weighted_average,
            "evaluate_fn": evaluate,
            "initial_parameters": fl.common.ndarrays_to_parameters(get_parameters(net))
        
        }

        # TODO: modify the following sentence.
        #       it should not do hard coding
        target_config = {**strategy_config, 'proximal':0.1} if config.strategy == 'FedProx' else target_config

        strategy = config.strategy
        hist = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=config.num_clients,
            num_rounds=config.num_rounds,
            strategy=get_strategy(strategy, target_config),
            client_resources=client_resources,
        )

        self.save_result(hist)

        def client_fn(cid: str) -> FlowerClient:
            """Create a Flower client representing a single organization."""
            global idx_fl_scale

            # Load model
            trainloader = config.data.clients[int(cid)]['train']
            valloader = config.data.clients[int(cid)]['val']

            # Create a single Flower client representing a single organization
            return FlowerClient(DEVICE, net.clone(), trainloader, valloader, config.train_config)

        def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

            criterion = nn.CrossEntropyLoss()
            net = load_model(**(config.model))
            set_parameters(net, parameters)
            net = net.to(DEVICE)
            loss, metrics = test(DEVICE, net, config.data.testloader, criterion)
            
            return loss, metrics # The return type must be (loss, metric tuple) form

        def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:

            averages = {}
            targets = ["accuracy", "recall", "precision", "f1_score"]

            total_examples = sum([num_examples for num_examples, _ in metrics])
            for target in targets:
                target_distributed = [num_examples * m[target] for num_examples, m in metrics]
                averages[target] = sum(target_distributed) / total_examples
            
            # Aggregate and return custom metric (weighted average)
            return averages
    
    def save_result(self, hist):
        config = self.config
        
        df_result = pd.DataFrame()
        df_result['round'] = [i for i in range(1, config.num_roudns + 1)]
        df_result['strategy'] = config.strategy     # TODO: change this line if the 'strategy' in config.json is modified to object shape
        df_result['model'] = config.model.modelname

        # centralized metrics
        metrics_cen = list(hist.metrics_centralized.keys())
        metrics_dis = list(hist.metrics_distributed.keys())

        for metric in metrics_cen:
            df_result[f"c_{metric}"] = [h[1] for h in hist.metrics_centralized[metric][1:]]
        for metric in metrics_dis:
            df_result[f"d_{metric}"] = [h[1] for h in hist.metrics_distributed[metric]]

        df_result.to_csv(os.path.join(config.paths.result_dir, 'result.csv'), index=False)
