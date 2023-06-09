import logging
import os
import torch

config = {
    "result_path"
}


class Simulation_Unit(object):
    
    def __init__(self, config):
        self.config = config

    def extract_config(self):
        config = self.config

        self.trainloader

    def load_model(self):
        




    def run(self):
        config = self.config
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    
        