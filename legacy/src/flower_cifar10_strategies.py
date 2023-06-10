from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
import os
import sys
import time
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision.datasets import CIFAR10


import flwr as fl
from flwr.common import Metrics

from models import Net
from utils import train, test, get_parameters, set_parameters
from client import FlowerClient
from strategy import get_strategy
from data import DataManager


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_path = "../result/best_ckpt.pth"
result_path = "../result/result_strategies.csv"
# General
NUM_CLIENTS = 10
BATCH_SIZE = 128
num_rounds = 100


print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)


dm = DataManager()
df_final = pd.DataFrame()

#######################
# Centralized Setting #
#######################
print("Experiment on centralized manner.")

if (os.path.exists(result_path)):
    df_final = pd.read_csv(result_path)
    removeIndex = df_final[df_final['strategy'] == 'Central'].index
    df_final.drop(removeIndex, inplace=True)
else:
    df_final = pd.DataFrame()


#net = Net().to(DEVICE)
net = torchvision.models.vgg19(num_classes=10).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)

best_acc = 0
for epoch in range(1, num_rounds + 1):
    print(f"Epoch: {epoch}")
    train(DEVICE, net, dm.trainloader, criterion, optimizer, epochs=1)
    test_loss, metrics = test(DEVICE, net, dm.testloader, criterion)
    if epoch % 10 == 0:
        print(f"[Epoch {epoch}]", end="")
        for k, v in metrics.items():
            print(f"{k}: {v}", end=" ")
        print()
    if best_acc < metrics['accuracy']:
        print(f'Saving...(epoch {epoch})')
        state = {
            'net': net.state_dict(),
            'acc': metrics['accuracy'],
            'epoch': epoch,
        }
        torch.save(state, save_path)
        best_acc = metrics['accuracy']
    
    df_result = pd.DataFrame()
    df_result['round'] = epoch,
    df_result['strategy'] = 'Central',
    for k, v in metrics.items():
        df_result[f"c_{k}"] = v
    # df_result['c_loss'] = test_loss,
    # df_result['c_accuracy'] = metrics['accuracy'],
    df_result['d_accuracy'] = 0.0

    df_final = pd.concat([df_final, df_result], axis=0)

loss, metrics = test(DEVICE, net, dm.testloader, criterion)
print(f"Final test set performance: ")
for k, v in metrics.items(): print(f"{k}: {v}")

dl_client_train, dl_client_val = dm.get_FL_data(total=50000)
print(f"number of data for client: {len(dl_client_train)}")

def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    global idx_fl_scale

    # Load model
    #net = Net().to(DEVICE)
    net = torchvision.models.vgg19(num_classes=10).to(DEVICE)
    trainloader = dl_client_train[int(cid)]
    valloader = dl_client_val[int(cid)]

    # Create a single Flower client representing a single organization
    return FlowerClient(DEVICE, net, trainloader, valloader)


def evaluate(
    server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

    criterion = nn.CrossEntropyLoss()
    #net = Net().to(DEVICE)
    net = torchvision.models.vgg19(num_classes=10).to(DEVICE)
    set_parameters(net, parameters)
    net = net.to(DEVICE)
    loss, metrics = test(DEVICE, net, dm.testloader, criterion)
    
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


# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}


config = {
    'fraction_fit':1.0,
    'fraction_evaluate':0.5,
    'min_fit_clients':10,
    'min_evaluate_clients':5,
    'min_available_clients':10,
    'evaluate_metrics_aggregation_fn':weighted_average,   # aggregate evaluation of local model
    'evaluate_fn':evaluate,   # evaluate global model
    #'initial_parameters':fl.common.ndarrays_to_parameters(get_parameters(Net())),

    'initial_parameters':fl.common.ndarrays_to_parameters(get_parameters(torchvision.models.vgg19(num_classes=10))),
}

strategies = ['FedAvg', 'FedAvgM', 'FedOpt']
strategies = ['FedAvg', 'FedAvgM', 'FedOpt', 'FedProx', 'FedAdagrad']



print("Experiment on federated manner.")
for strategy in strategies:
    print(f"{strategy} simulation")

    target_config = {**config, 'proximal_mu':0.1} if strategy == 'FedProx' else config

    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=get_strategy(strategy, target_config),
        client_resources=client_resources,
    )

    df_result = pd.DataFrame()
    df_result['round'] = [i for i in range(1, num_rounds + 1)]
    df_result['strategy'] = strategy

    # centralized metrics
    metrics_cen = list(hist.metrics_centralized.keys())
    metrics_dis = list(hist.metrics_distributed.keys())

    for metric in metrics_cen:
        df_result[f"c_{metric}"] = [h[1] for h in hist.metrics_centralized[metric][1:]]
    for metric in metrics_dis:
        df_result[f"d_{metric}"] = [h[1] for h in hist.metrics_distributed[metric]]

    df_final = pd.concat([df_final, df_result], axis=0)

df_final.to_csv(result_path, index=False)