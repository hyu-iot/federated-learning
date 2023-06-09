import logging

class Server:
    def __init__(self, config):
        self.config = config

        logging.info('Starting...')
        
        self.create_folder()
        self.load_data()

    def run(self):
        self.iterate_simulation()
        pass


    def load_data(self):
        print(self.config.models)
        pass

    def create_folder(self):
        pass
    def create_simulation(self):
        pass
    def iterate_simulation(self):
        for key in self.config.models:
            for strategy in self.config.strategies:
                # print(key)
                # print(strategy)
                self.create_simulation()

        pass


