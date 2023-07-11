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
        logging.info("Training on " + self.device.type)
        self.model = load_model(**(config.model)).to(self.device)
        self.moving_average_k = 7
        self.moving_average = 0
        self.losses = []

    def run(self):
        if str(self.config.strategy).lower() == 'centralized':
            self.run_centralized()
        else:
            self.run_fl()


    def run_centralized(self):
        run_config = self.config
        net = self.model
        DEVICE= self.device

        train_config = {
            "lr": run_config.fl.learning_rate,
            "momentum": run_config.fl.momentum,
            "weight_decay": run_config.fl.weight_decay
        }        

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), **train_config)
        
        best_acc = 0
        result_path = os.path.join(run_config.paths['result_dir'], 'result.csv')
        df_final = pd.DataFrame()
        for epoch in range(1, run_config.fl.centralized_epochs + 1):
            logging.info(f"Epoch: {epoch}")
            train(DEVICE, net, run_config.data['trainloader'], criterion, optimizer)
            test_loss, metrics = test(DEVICE, net, run_config.data['testloader'], criterion)
            if epoch % 10 == 0:
                logging.info(f"[Epoch {epoch}]", end="")
                for k, v in metrics.items():
                    logging.info(f"{k}: {v}", end=" ")
            if best_acc < metrics['accuracy']:
                logging.info(f'Saving...(epoch {epoch})')
                state = {
                    'net': net.state_dict(),
                    'acc': metrics['accuracy'],
                    'epoch': epoch,
                }
                torch.save(state, os.path.join(run_config.paths['result_dir'], "best_model.pth"))
                best_acc = metrics['accuracy']
            
            df_result = pd.DataFrame()
            df_result['round'] = epoch,
            df_result['model'] = self.config.model['modelname'],
            df_result['strategy'] = 'Centralized',
            for k, v in metrics.items():
                df_result[f"c_{k}"] = v
            df_result['d_accuracy'] = 0.0

            df_final = pd.concat([df_final, df_result], axis=0)
            if not os.path.exists(result_path):
                df_result.to_csv(result_path, index=False, mode='w')
            else:
                df_result.to_csv(result_path, index=False, mode='a', header=False)

    def run_fl(self):
        
        run_config = self.config
        net = self.model
        DEVICE = self.device

        train_config = {
            "lr": run_config.fl.learning_rate,
            "momentum": run_config.fl.momentum,
            "weight_decay": run_config.fl.weight_decay
        }

        def client_fn(cid: str) -> FlowerClient:
            """Create a Flower client representing a single organization."""
            global idx_fl_scale

            # Load model
            trainloader = run_config.data['clients'][int(cid)]['train']
            valloader = run_config.data['clients'][int(cid)]['val']

            # Create a single Flower client representing a single organization
            return FlowerClient(DEVICE, net, trainloader, valloader, train_config)

        def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

            criterion = nn.CrossEntropyLoss()
            net = load_model(**(run_config.model))
            set_parameters(net, parameters)
            net = net.to(DEVICE)
            loss, metrics = test(DEVICE, net, run_config.data['testloader'], criterion)
            logging.info(f"Server round: {server_round}")
            
            # TODO: modify the acc and epoch
            state = {
                'net': net.state_dict(),
                'acc': 0,
                "epoch": -1,
            }
            if (server_round > 0):
                torch.save(state, "./models/result_model.pth")

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

        
        # TODO : make this client_resources to be defined by config.json
        clinet_resources = None
        if DEVICE.type == "cuda":
            client_resources = {"num_gpus": 1}

        # TODO : modify config.json and config.py
        #        to let one strategy have one config
        #        (refer the case of models)
        strategy_config = {
            "fraction_fit": run_config.fl.fraction_fit,
            "fraction_evaluate": run_config.fl.fraction_evaluate,
            "min_fit_clients": run_config.fl.min_fit_clients,
            "min_evaluate_clients": run_config.fl.min_evaluate_clients,
            "min_available_clients": run_config.fl.min_available_clients,
            "evaluate_metrics_aggregation_fn" : weighted_average,
            "evaluate_fn": evaluate,
            "initial_parameters": fl.common.ndarrays_to_parameters(get_parameters(net))
        
        }

        # TODO: modify the following sentence.
        #       it should not do hard coding
        strategy_config = {**strategy_config, 'proximal':0.1} if run_config.strategy == 'FedProx' else strategy_config

        # TODO model load -> simulation for 1 round -> result save -> model save        

        for i in range(1, 1+run_config.fl.rounds):
            if (i > 1):
                net = load_model(**({**(self.config.model), "pretrained_path":"./models/result_model.pth"})).to(self.device)
            
            target_config = {**strategy_config, 
                'initial_parameters': fl.common.ndarrays_to_parameters(get_parameters(net))
            }

            hist = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=run_config.clients.total,
                config=fl.server.ServerConfig(num_rounds=1),
                strategy=get_strategy(run_config.strategy, target_config),
                client_resources=client_resources,
            )

            self.save_result_2(hist, i)
            if (self.check_saturation(hist)):    break

        # hist = fl.simulation.start_simulation(
        #     client_fn=client_fn,
        #     num_clients=run_config.clients.total,
        #     config=fl.server.ServerConfig(num_rounds=run_config.fl.rounds),
        #     strategy=get_strategy(run_config.strategy, target_config),
        #     client_resources=client_resources,
        # )

        # self.save_result(hist)
    
    def check_saturation(self, hist):
        
        
        # if (len(self.losses) > 0):
        #     del self.losses[0]         # 
        # else:
        #     self.moving_average = 0     

        current_loss = hist.metrics_centralized['loss'][1][1]
        logging.info(f"Loss: {current_loss}")
        self.losses.append(current_loss)

        if (len(self.losses) >= self.moving_average_k):
            self.moving_average = sum(self.losses[-self.moving_average_k:]) / self.moving_average_k
            if (abs((self.moving_average - current_loss) / self.moving_average) < 0.01):
                return True
        
        return False

    def save_result(self, hist):
        config = self.config
        
        df_result = pd.DataFrame()
        df_result['round'] = [i for i in range(1, config.fl.rounds + 1)]
        df_result['model'] = config.model['modelname']
        df_result['strategy'] = config.strategy     # TODO: change this line if the 'strategy' in config.json is modified to object shape

        # centralized metrics
        metrics_cen = list(hist.metrics_centralized.keys())
        metrics_dis = list(hist.metrics_distributed.keys())

        logging.info(f"MC_list: {metrics_cen}\nMD_list: {metrics_dis}")
        logging.info(f"MC: {hist.metrics_centralized}\nMD: {hist.metrics_distributed}")
        result_path = os.path.join(config.paths['result_dir'], 'result.csv')
        for metric in metrics_cen:
            df_result[f"c_{metric}"] = [h[1] for h in hist.metrics_centralized[metric][1:]]
        for metric in metrics_dis:
            df_result[f"d_{metric}"] = [h[1] for h in hist.metrics_distributed[metric]]
        if not os.path.exists(result_path):
            df_result.to_csv(result_path, index=False, mode='w')
        else:
            df_result.to_csv(result_path, index=False, mode='a', header=False)


    def save_result_2(self, hist, current_round):
        config = self.config

        df_result = pd.DataFrame()
        df_result['round'] =[current_round]
        df_result['model'] = config.model['modelname']
        df_result['strategy'] = config.strategy

        # centralized metrcis
        metrics_cen = list(hist.metrics_centralized.keys())
        metrics_dis = list(hist.metrics_distributed.keys())

        logging.info(f"MC_list: {metrics_cen}\nMD_list: {metrics_dis}")
        logging.info(f"MC: {hist.metrics_centralized}\nMD: {hist.metrics_distributed}")
        result_path = os.path.join(config.paths['result_dir'], 'result.csv')
        
        for metric in metrics_cen:
            df_result[f"c_{metric}"] = [h[1] for h in hist.metrics_centralized[metric][1:]]
        for metric in metrics_dis:
            df_result[f"d_{metric}"] = [h[1] for h in hist.metrics_distributed[metric]]
        if not os.path.exists(result_path):
            df_result.to_csv(result_path, index=False, mode='w')
        else:
            df_result.to_csv(result_path, index=False, mode='a', header=False)