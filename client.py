# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import NN

import yaml
import datetime
import socket
import tqdm
import os

class Client:

    def __init__(self):

        self.config = {
            'id': 1,
            'timestamp': int(datetime.datetime.now().timestamp()),
            'device': "cuda" if torch.cuda.is_available() else "cpu",
            'data_path': './data',
            'dataset': 'FasionMNIST',
            'criterion': torch.nn.CrossEntropyLoss,
            'optimizer': torch.optim.SGD,
            'lr': 0.01,
            'batch_size': 64,
            'epoch': 5,
        }

        with open('./config.yaml') as c:
            self.config_yaml = list(yaml.load_all(c, Loader=yaml.FullLoader))

            self.global_configs = self.config_yaml[0]['global_config']
            self.data_configs = self.config_yaml[1]['data_config']
            self.train_configs = self.config_yaml[2]['train_config']
            self.optim_configs = self.config_yaml[3]['optim_config']

        for c in [self.global_configs, self.data_configs, self.train_configs, self.optim_configs]:
            self.update_config(c)

        print(self.config)

        
    def update_config(self, config):
        for k, v in config.items():
            if k in self.config.keys():
                self.config[k] = v



    def train(self, dataloader, model, loss_fn, optimizer):
        device = self.config['device']
        size = len(dataloader.dataset)
        model.train()
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # Compute prediction error
            pred = model(x)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(x)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

        
        return loss

    def test(self, dataloader, model, loss_fn):
        device = self.config['device']
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        return correct

    def run(self):

        # Download training data from open datasets.
        training_data = datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        # Download test data from open datasets.
        test_data = datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=ToTensor(),
        )
        
        # Create data loaders.
        train_dataloader = DataLoader(training_data, batch_size=self.config['batch_size'])
        test_dataloader = DataLoader(test_data, batch_size=self.config['batch_size'])

        for x, y in test_dataloader:
            print(f"Shape of x [N, C, H, W]: {x.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

        device = self.config['device']
        print(f"Using {device} device")

        model = NN().to(device)
        print(model)

        # Optimizing the Model Parameters
        loss_fn = eval(self.config['criterion'])()
        optimizer = eval(self.config['optimizer'])(model.parameters(), lr=1e-3)

        for t in range(self.config['epoch']):
            print(f"Epoch {t+1}\n-------------------")

            #TODO : Is it right to get loss from train and accuracy from test?
            loss = self.train(train_dataloader, model, loss_fn, optimizer)
            accuracy = self.test(test_dataloader, model, loss_fn)
        print("Done!")
        self.update_config({'timestamp': int(datetime.datetime.now().timestamp())})



        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())


        print("Optimizer's state_dict:")
        for param_tensor in optimizer.state_dict():
            print(param_tensor, "\t", optimizer.state_dict()[param_tensor])



        # Saving Models
        filename = f'model_{self.config["id"]}_{self.config["timestamp"]}.pt'
        filepath = f'./output/{filename}'

        torch.save({
            'id': self.config['id'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,

        }, filepath)
        print(f"Saved Pytorch Model State to {filepath}")


        # Loading Models
        model = NN()
        model.load_state_dict(torch.load(filepath)['model_state_dict'])

        classes = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]

        model.eval()
        x, y = test_data[0][0], test_data[0][1]
        with torch.no_grad():
            pred = model(x)
            print(f"Pred: {[pred]}")
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f"Predicted: {predicted}, Actual: {actual}")



if __name__ == "__main__":
    c = Client()
    c.run()
