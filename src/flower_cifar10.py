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

from sklearn.metrics import f1_score, precision_score, recall_score

import flwr as fl
from flwr.common import Metrics



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_path = "../result/best_ckpt.pth"
result_path = "../result/result.csv"

print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# General
NUM_CLIENTS = 10
BATCH_SIZE = 128
num_rounds = 100

save_path = "../result/best_ckpt.pth"
result_path = "../result/result.csv"


def load_datasets():
    # Download and transform CIFAR-10 (train and test)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform_train)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))
    testloader = DataLoader(testset, batch_size=32)

    print(f"Trainset: {len(trainset)}, Testset: {len(testset)}")
    print(f"Trainloader: {len(trainloader)}, Testloader: {len(testloader)}")
    return trainloader, testloader, trainloaders, valloaders

#trainloaders, valloaders, testloader = load_datasets(divide=False)
trainloader, testloader, trainloaders, valloaders = load_datasets()


def train(net, trainloader, epochs=1):
    """Train the network on the training set."""
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(epochs):

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        scheduler.step()


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    
    net.eval()
    test_loss, correct, total = 0, 0, 0    
    criterion = nn.CrossEntropyLoss()

    pred = torch.Tensor([])
    target = torch.Tensor([])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pred = torch.cat([pred, predicted.cpu()], dim=0)
            target = torch.cat([target, targets.cpu()], dim=0)

    test_loss /= len(testloader)
    accuracy = correct / total
    recall = recall_score(pred, target, average='macro')
    precision = precision_score(pred, target, average='macro')
    f1 = f1_score(pred, target, average='macro')

    return test_loss, {"loss": test_loss, "accuracy": accuracy, "recall": recall, "precision": precision, "f1_score": f1}


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


net = Net().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


best_acc = 0
for epoch in range(1, 101):
    print(f"Epoch: {epoch}")
    train(net, trainloader)
    test_loss, metrics = test(net, testloader)
    scheduler.step()
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

loss, metrics = test(net, testloader)
print(f"Final test set performance: ")
for k, v in metrics.items(): print(f"{k}: {v}")



#####################
# Federated Setting #
#####################
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    #print(f"Net: {len(net.state_dict().keys())} Params: {len(parameters)}")
    params_dict = zip(net.state_dict().keys(), parameters)
    # for k, v in params_dict:
    #     continue
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=False)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
    
    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}
    
    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, metrics = test(self.net, self.valloader)
        return float(loss), len(self.valloader), metrics


def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)
    #net = ResNet50().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client 
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    # Create a single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader)

def evaluate(
    server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

    net = Net().to(DEVICE)
    #net = ResNet50().to(DEVICE)
    set_parameters(net, parameters)
    net = net.to(DEVICE)
    loss, metrics = test(net, testloader)
    
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


fedavg = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=10,
    min_evaluate_clients=5,
    min_available_clients=10,
    evaluate_metrics_aggregation_fn=weighted_average,   # aggregate evaluation of local model
    evaluate_fn=evaluate,   # evaluate global model
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())),
)

fedavgM = fl.server.strategy.FedAvgM(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=10,
    min_evaluate_clients=5,
    min_available_clients=10,
    evaluate_metrics_aggregation_fn=weighted_average,   # aggregate evaluation of local model
    evaluate_fn=evaluate,   # evaluate global model
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())),
)

qfedavg = fl.server.strategy.QFedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=10,
    min_evaluate_clients=5,
    min_available_clients=10,
    evaluate_metrics_aggregation_fn=weighted_average,   # aggregate evaluation of local model
    evaluate_fn=evaluate,   # evaluate global model
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())),
)

ftfedavg = fl.server.strategy.FaultTolerantFedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=10,
    min_evaluate_clients=5,
    min_available_clients=10,
    evaluate_metrics_aggregation_fn=weighted_average,   # aggregate evaluation of local model
    evaluate_fn=evaluate,   # evaluate global model
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())),
)

fedopt = fl.server.strategy.FedOpt(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=10,
    min_evaluate_clients=5,
    min_available_clients=10,
    evaluate_metrics_aggregation_fn=weighted_average,   # aggregate evaluation of local model
    evaluate_fn=evaluate,   # evaluate global model
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())),
)

fedprox = fl.server.strategy.FedProx(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=10,
    min_evaluate_clients=5,
    min_available_clients=10,
    evaluate_metrics_aggregation_fn=weighted_average,   # aggregate evaluation of local model
    evaluate_fn=evaluate,   # evaluate global model
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())),
    proximal_mu=0.1,
)

fedadagrad = fl.server.strategy.FedAdagrad(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=10,
    min_evaluate_clients=5,
    min_available_clients=10,
    evaluate_metrics_aggregation_fn=weighted_average,   # aggregate evaluation of local model
    evaluate_fn=evaluate,   # evaluate global model
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())),
)

fedadam = fl.server.strategy.FedAdam(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=10,
    min_evaluate_clients=5,
    min_available_clients=10,
    evaluate_metrics_aggregation_fn=weighted_average,   # aggregate evaluation of local model
    evaluate_fn=evaluate,   # evaluate global model
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())),
)

fedyogi = fl.server.strategy.FedYogi(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=10,
    min_evaluate_clients=5,
    min_available_clients=10,
    evaluate_metrics_aggregation_fn=weighted_average,   # aggregate evaluation of local model
    evaluate_fn=evaluate,   # evaluate global model
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())),
)



# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

strategies = {
    'FedAvg': fedavg,
    'FedAvgM': fedavgM,
    'QFedAvg': qfedavg,
    'FaultTolerantFedAvg': ftfedavg,
    'FedOpt': fedopt,
    'FedProx': fedprox,
    'FedAdagrad': fedadagrad,
    'FedAdam': fedadam,
    'FedYogi': fedyogi,
}

print("Experiment on federated manner.")
for sname, strategy in strategies.items():
    print(f"{sname} simulation")

    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    df_result = pd.DataFrame()
    df_result['round'] = [i for i in range(1, num_rounds + 1)]
    df_result['strategy'] = sname

    # centralized metrics
    metrics_cen = list(hist.metrics_centralized.keys())
    metrics_dis = list(hist.metrics_distributed.keys())

    for metric in metrics_cen:
        df_result[f"c_{metric}"] = [h[1] for h in hist.metrics_centralized[metric][1:]]
    for metric in metrics_dis:
        df_result[f"d_{metric}"] = [h[1] for h in hist.metrics_distributed[metric]]

    df_final = pd.concat([df_final, df_result], axis=0)

df_final.to_csv('../result/result.csv', index=False)