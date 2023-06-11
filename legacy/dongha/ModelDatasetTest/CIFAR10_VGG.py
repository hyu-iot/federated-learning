# %% [markdown]
# ## CIFAR10 DATASET SPLIT ##
# 
# Total_Trainset = 50000, Total_Testset = 10000
# 
# ### Centralized ###
# 
# Trainset = 45000, Valset= 5000, Testset = 10000
# 
# ### Federated ###
# 
# NUM_CLIENTS = 10
# Trainset = 4500, Valset= 500, (Each client)
# Testset = 10000 (Aggregated Model)
# 
# metrics_distributed �� �� client���� ���� ��� ��ճ� accuracy
# metrics_centralized �� ���� model test�� ��� ���� accuracy

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
import torchvision.models

from sklearn.metrics import f1_score, precision_score, recall_score

import os
# ���⼭�� 10���� ���� �� ��, �ϳ��� train ��.
import pandas as pd


import flwr as fl
from flwr.common import Metrics
import flwr.common

# %%

DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

NUM_CLIENTS = 10
EPOCH = 100
BATCH_SIZE = 32

client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

transform = transforms.Compose(
    [transforms.ToTensor(), 
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)
trainset = CIFAR10("../dataset", train=True, download=True, transform=transform)
testset = CIFAR10("../dataset", train=False, download=True, transform=transform)

# Split dataset
lengths = [45000, 5000]
split_trainset, valset = random_split(trainset, lengths, torch.Generator().manual_seed(42)) 
    
# 45000
full_split_trainloader = DataLoader(split_trainset, batch_size=BATCH_SIZE, shuffle=True)
# 5000
full_valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)
# 10000
full_testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)


def load_datasets():
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

        # Count Distribution of data
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


# %% [markdown]
# ### VGG 16 ###

# %%
# all_models = torchvision.models.list_models()
# all_models

# %%
# net = torchvision.models.densenet121(num_classes=10).to(DEVICE)
# net = torchvision.models.resnet18(num_classes=10).to(DEVICE)
# net = torchvision.models.resnet34(num_classes=10).to(DEVICE)
# net = torchvision.models.resnet50(num_classes=10).to(DEVICE)

# net = torchvision.models.googlenet(num_classes=10).to(DEVICE)
# net

# %%
def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
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
    pred = torch.Tensor([])
    target = torch.Tensor([])
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred = torch.cat([pred, predicted.cpu()], dim=0)
            target = torch.cat([target, labels.cpu()], dim=0)
    loss /= len(testloader.dataset)
    accuracy = correct / total
    recall = recall_score(pred, target, average='macro')
    precision = precision_score(pred, target, average='macro')
    f1 = f1_score(pred, target, average='macro')
    return loss, {"loss": loss, "accuracy": accuracy, "recall": recall, "precision": precision, "f1_score": f1}


# %% [markdown]
# ## Federated ##

# %%
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {
            k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
            for k, v in params_dict
        }
    )
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
        loss, metrics = test(self.net, self.valloader)
        print(f"Accuracy {metrics['accuracy']}")
        return float(loss), len(self.valloader), metrics
    
def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    # net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader)
    
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:

    averages = {}
    targets = ["accuracy", "recall", "precision", "f1_score"]

    total_examples = sum([num_examples for num_examples, _ in metrics])
    for target in targets:
        target_distributed = [num_examples * m[target] for num_examples, m in metrics]
        averages[target] = sum(target_distributed) / total_examples
    
    # Aggregate and return custom metric (weighted average)
    return averages

# The `evaluate` function will be by Flower called after every round
def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    # net = Net().to(DEVICE)
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, metrics = test(net, full_testloader) # Test total test_loader
    print(f"Server-side evaluation loss {loss} / accuracy {metrics['accuracy']}")
    return loss, metrics


# %%
models = {
    'Densenet121' : "densenet121",
    'Mobilenet_v2': "mobilenet_v2", # Odd results
    'Googlenet' : "googlenet", # Odd results
    'Resnet18' : "resnet18",
    'Resnet34' : "resnet34",
    'Resnet50' : "resnet50",
    'VGG11' : "vgg11", 
    'VGG13' : "vgg13", 
    'VGG16' : "vgg16", 
    'VGG19' : "vgg19",
}

# %%
df_final = pd.DataFrame()

for key, value in models.items():
    if key == 'Googlenet':
        net = torchvision.models.googlenet(num_classes=10, aux_logits=False).to(DEVICE)
    else:    
        net = getattr(torchvision.models, value)(num_classes=10).to(DEVICE)
    ######## Centralized #########
    print("Centralized: " + key)
    trained_path = "../dataset/trained_centralized_"+key+".pkl"
    df_centralized = pd.DataFrame()
    for epoch in range(EPOCH):
        df_temp = pd.DataFrame()
        train(net, full_split_trainloader, 1)
        loss, c_metrics = test(net, full_testloader) # Federated aggregation model tests full_testloader
        print(f"{key} Epoch {epoch+1}: validation loss {loss}, accuracy {c_metrics['accuracy']}")
        
        df_temp['round'] = epoch+1,
        df_temp['Centralized'] = 'Centralized',
        df_temp['model'] = key
        for metric in c_metrics:
            df_temp[f"c_{metric}"] = c_metrics[metric]
        df_centralized = pd.concat([df_centralized, df_temp], axis=0)
        df_final = pd.concat([df_final, df_temp], axis=0)
        df_centralized.to_csv('../result/result_c_'+key+'.csv', index=False)
    torch.save(net.state_dict(), trained_path)

    ######## Federated ######## 
    # Reset net
    if key == 'Googlenet':
        net = torchvision.models.googlenet(num_classes=10, aux_logits=False).to(DEVICE)
    else:    
        net = getattr(torchvision.models, value)(num_classes=10).to(DEVICE)
    print("Federated: " + key)
    fedavg = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=10,
        min_evaluate_clients=5,
        min_available_clients=10,
        evaluate_metrics_aggregation_fn=weighted_average,   # aggregate evaluation of local model
        evaluate_fn=evaluate,   # evaluate global model
    )  
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=EPOCH),
        strategy=fedavg,
        client_resources=client_resources,
    ) 

    df_federated = pd.DataFrame()
    df_federated['round'] = [i for i in range(1, EPOCH + 1)]
    df_federated['Centralized'] = "Federated" 
    df_federated['model'] = key

    # centralized metrics
    metrics_cen = list(hist.metrics_centralized.keys())
    metrics_dis = list(hist.metrics_distributed.keys())

    for metric in metrics_cen:
        df_federated[f"c_{metric}"] = [h[1] for h in hist.metrics_centralized[metric][1:]]
    for metric in metrics_dis:
        df_federated[f"d_{metric}"] = [h[1] for h in hist.metrics_distributed[metric]]
    
    df_federated.to_csv('../result/result_f_'+key+'.csv', index=False)

    df_final = pd.concat([df_final, df_federated], axis=0)

    df_final.to_csv('../result/result_total.csv', index=False)

