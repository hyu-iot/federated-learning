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

print(f"Trainset: {len(trainset)}, Testset: {len(testset)}")
print(f"Trainloader: {len(trainloader)}, Testloader: {len(testloader)}")


print(f"Make federated dataset...")
li_trainset = [torch.empty([0, 3, 32, 32]) for _ in range(10)]
labels = [[] for _ in range(10)]

for x, label in trainset:
    labels[label].append(label)
    li_trainset[label] = torch.cat((li_trainset[label], x[None, :]), dim=0)
    
labels = [torch.tensor(x) for x in labels]

d_train = [torch.empty([0, 3, 32, 32]) for _ in range(NUM_CLIENTS)]
d_train_labels = [torch.tensor([], dtype=int) for _ in range(NUM_CLIENTS)]
ds_clients = []         # dataset for each client
dl_clients = []         # dataloader for each client

for ds, label in zip(li_trainset, labels):
    indices = torch.randperm(5000)
    ds = ds[indices]
    label = label[indices]
    num_data = 5000 // NUM_CLIENTS
    for i in range(NUM_CLIENTS):
        d_train[i] = torch.cat((d_train[i], ds[i*num_data : (i+1)*num_data]), dim=0)
        d_train_labels[i] = torch.cat((d_train_labels[i], label[i*num_data : (i+1)*num_data]))

for i in range(NUM_CLIENTS):
    indices = torch.randperm(5000)
    d_train[i] = d_train[i][indices]
    d_train_labels[i] = d_train_labels[i][indices]
    ds_client = TensorDataset(d_train[i], d_train_labels[i])
    dl_client = DataLoader(
        ds_client,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    ds_clients.append(ds_client)
    dl_clients.append(dl_client)


print(len(ds_clients[0]), len(dl_clients[0]))


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

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss /= len(testloader)
    accuracy = correct / total

    return test_loss, accuracy



df_final = pd.DataFrame()

#######################
# Centralized Setting #
#######################
print("Experiment on centralized manner.")
net = Net().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

best_acc = 0
for epoch in range(1, 101):
    print(f"[Centralized] Epoch: {epoch}")
    train(net, trainloader, 1)
    loss, accuracy = test(net, testloader)
    scheduler.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: validation loss {loss}, accuracy {accuracy}")
    if best_acc < accuracy:
        print(f'Saving...(epoch {epoch})')

        state = {
            'net': net.state_dict(),
            'acc': accuracy,
            'epoch': epoch,
        }
        torch.save(state, save_path)
        best_acc = accuracy
    
    df_result = pd.DataFrame()
    df_result['round'] = epoch,
    df_result['strategy'] = 'Central',
    df_result['c_loss'] = loss,
    df_result['c_accuracy'] = accuracy,
    df_result['d_accuracy'] = 0.0

    df_final = pd.concat([df_final, df_result], axis=0)

loss, accuracy = test(net, testloader)
print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")



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
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)
    #net = ResNet50().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client 
    # will train and evaluate on their own unique data
    trainloader = dl_clients[int(cid)]
    #valloader = valloaders[int(cid)]
    valloader = testloader

    # Create a single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader)

def evaluate(
    server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

    net = Net().to(DEVICE)
    #net = ResNet50().to(DEVICE)
    set_parameters(net, parameters)
    net = net.to(DEVICE)
    loss, accuracy = test(net, testloader)
    
    return loss, {"loss": loss, "accuracy": accuracy} # The return type must be (loss, metric tuple) form

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


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