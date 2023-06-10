from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch

from sklearn.metrics import f1_score, precision_score, recall_score


def train(device, net, trainloader, criterion, optimizer, epochs=1):
    """Train the network on the training set."""
    net.train()

    for epoch in range(epochs):

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            

def test(device, net, testloader, criterion):
    """Evaluate the network on the entire test set."""
    
    net.eval()
    test_loss, correct, total = 0, 0, 0    

    pred = torch.Tensor([])
    target = torch.Tensor([])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
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
