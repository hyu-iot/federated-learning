from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import flwr as fl
from flwr.common import Metrics
import logging

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

# The `evaluate` function will be by Flower called after every round
def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    # net = Net().to(DEVICE)
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, metrics = test(net, full_testloader) # Test total test_loader
    logging.info(f"Server-side evaluation loss {loss} / accuracy {metrics['accuracy']}")
    return loss, metrics

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
        logging.info(f"Accuracy {metrics['accuracy']}")
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

