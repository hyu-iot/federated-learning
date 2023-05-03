# %%
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

import flwr as fl
from flwr.common import Metrics
import flwr.common


DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

NUM_CLIENTS = 10

BATCH_SIZE = 32

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
testset = CIFAR10("./dataset", train=False, download=True, transform=transform)



lengths = [45000, 5000]
split_trainset, valset = random_split(trainset, lengths, torch.Generator().manual_seed(42)) 
    
# 5000
full_valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)

# 45000
full_split_trainloader = DataLoader(split_trainset, batch_size=BATCH_SIZE, shuffle=True)
# 10000
full_testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)


def load_datasets():
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)



    lengths = [45000, 5000]
    split_trainset, valset = random_split(trainset, lengths, torch.Generator().manual_seed(42)) 
    
    # 5000
    full_valset = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)

    # 45000
    full_split_trainloader = DataLoader(split_trainset, batch_size=BATCH_SIZE, shuffle=True)
    # 10000
    full_testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    # Split training set into 10 partitions to simulate the individual dataset
    partition_size = len(trainset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    datasets= random_split(trainset, lengths, torch.Generator().manual_seed(42))

    

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for idx, ds in enumerate(datasets):
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        # Always splits in the same way.

        # print(dict(Counter(ds_train.targets)))
        # custom_subset(ds_train)
        # arr = []
        # print("ds_train", idx)
        # for a in ds_train:
        #     arr.append(a[1])
        # for i in range(10):
        #     print(i, ":", arr.count(i))
        # Data is not perfectly distributed


        trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, valloaders, testloader


trainloaders, valloaders, testloader = load_datasets()


# %%
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy



client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}



# %%
EPOCH = 100

# %%
import os
# ���⼭�� 10���� ���� �� ��, �ϳ��� train ��.
import pandas as pd
df_final = pd.DataFrame()
print("Experiment on centralized manner.")

net = Net().to(DEVICE)
trained_path = "./dataset/trained_centralized.pkl"


if not (os.path.isfile(trained_path)):
    for epoch in range(EPOCH+1):
        train(net, full_split_trainloader, 1)
        loss, accuracy = test(net, full_valloader)
        print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")
        df_result = pd.DataFrame()
        df_result['round'] = epoch+1,
        df_result['strategy'] = 'Central',
        df_result['c_loss'] = loss,
        df_result['c_accuracy'] = accuracy,
        df_result['d_accuracy'] = 0.0

        df_final = pd.concat([df_final, df_result], axis=0)

    torch.save(net.state_dict(), trained_path)
else :
    net.load_state_dict(torch.load(trained_path))
loss, accuracy = test(net, full_testloader)
print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")

df_final

# %%
df_final

# %%
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

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
        print(f"Accuracy {accuracy}")
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    
def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader)
    
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# The `evaluate` function will be by Flower called after every round
def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = Net().to(DEVICE)
    valloader = valloaders[0]
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, valloader)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}


# %%
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

# %%

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
        config=fl.server.ServerConfig(num_rounds=EPOCH),
        strategy=strategy,
        client_resources=client_resources,
    )

    df_result = pd.DataFrame()
    df_result['round'] = [i for i in range(1, EPOCH + 1)]
    df_result['strategy'] = sname

    # centralized metrics
    metrics_cen = list(hist.metrics_centralized.keys())
    metrics_dis = list(hist.metrics_distributed.keys())

    for metric in metrics_cen:
        df_result[f"c_{metric}"] = [h[1] for h in hist.metrics_centralized[metric][1:]]
    for metric in metrics_dis:
        df_result[f"d_{metric}"] = [h[1] for h in hist.metrics_distributed[metric]]

    df_final = pd.concat([df_final, df_result], axis=0)

df_final.to_csv('./result/result2.csv', index=False)


