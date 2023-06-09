import logging
import os
from datetime import datetime
from visualization import Visualization

class Server:
    def __init__(self, config):
        self.config = config

        logging.info('Starting...')
        
        self.create_directory()
        self.load_data()

    def run(self):
        # self.iterate_simulation()
        vis = Visualization(self.config, self.mydir)
        vis.run()

        pass


    def load_data(self):
        # print(self.config.models)
        pass

    # Make the dirctory named using the timestamp.
    def create_directory(self):
        if not os.path.exists("./result"):
            os.makedirs("./result")
        self.mydir = os.path.join(os.getcwd(), "./result/", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.mydir)

    def create_simulation(self):
        pass
    def iterate_simulation(self):
        for key in self.config.models:
            for strategy in self.config.strategies:
                # print(key)
                # print(strategy)
                self.create_simulation()

        pass


